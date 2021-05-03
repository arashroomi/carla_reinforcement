from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys
from utilities import actions
import shadow_mask
import time
import transforms3d
import torch.multiprocessing as mp
from torch.multiprocessing import set_start_method
import open3d as o3d
import scipy.cluster.hierarchy as hcluster
from sklearn.cluster import KMeans


try:
    sys.path.append(glob.glob('./carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


import open3d as o3d


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla
from carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions, BirdView, DEFAULT_HEIGHT,DEFAULT_WIDTH
from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.roaming_agent import RoamingAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error



import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
import torch


try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_F2
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_g
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_l
    from pygame.locals import K_i
    from pygame.locals import K_z
    from pygame.locals import K_x
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS

except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')
# ==============================================================================
# -- Internal imports ----------------------------------------------------------
# ==============================================================================
from birdeye2tensor import BirdEye
import lidar_lib
# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================
def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)

        for frame, intensity in self.history:
            history[frame] += intensity

        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        print(actor_type)
        #self.hud.notification('Collision with %r' % actor_type)

        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================
class LidarSensor(object):
    def __init__(self, x ,y, z, parent_actor, hud,semantic,lidar_noise):
        self.lidar_image_pixels = 84
        self.pixel2meter = 2
        lidar_max_distance = self.lidar_image_pixels / (self.pixel2meter * 2)
        self.sensor = None
        self.points= None
        self.surface_lidar = None
        self.lidar_img = None
        self.blocking_pointclouds = None
        self.depth = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()

        lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
        lidar_bp.set_attribute('points_per_second', '100000')
        lidar_bp.set_attribute('rotation_frequency', '20')
        lidar_bp.set_attribute('lower_fov','-50')
        lidar_bp.set_attribute('upper_fov', '30')
        lidar_bp.set_attribute('channels', str(64))
        lidar_bp.set_attribute('range', str(lidar_max_distance-.5))


        rad_deg = np.pi / float(180)
        rad_45 = 45 * rad_deg
        rad_135 = 135 * rad_deg
        rad_180 = np.pi
        rad_225 = 225 * rad_deg
        rad_315 = 315 * rad_deg

        user_offset = carla.Location(x, y, z)
        lidar_transform = carla.Transform(carla.Location(x=-0.5, z=1.8) + user_offset)

        self.lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=parent_actor)

        weak_self = weakref.ref(self)
        #lidar.listen(lambda data: sensor_callback(data, sensor_queue, "lidar01"))

        self.lidar.listen(lambda image: LidarSensor.semantic_lidar_callback(weak_self, image))

    def process_data(self,bird_array):

        if self.points is not None:
            lidar_data = np.array(self.points)
            # print(lidar_data.shape)

            lidar_data[:, :2] *= self.pixel2meter
            lidar_data[:, :2] += (0.5 * self.lidar_image_pixels, 0.5 * self.lidar_image_pixels)
            lidar_data[:, :1] = np.fabs(lidar_data[:, :1])  # pylint: disable=E1111
            # lidar_data = lidar_data.astype(np.int32)

            lidar_data = np.reshape(lidar_data, (-1, 3))

            lidar_img_size = (self.lidar_image_pixels, self.lidar_image_pixels, 3)
            lidar_img = np.zeros((lidar_img_size), dtype=int)
            depth = np.zeros((self.lidar_image_pixels, self.lidar_image_pixels), dtype=float)
            # print(np.shape(tuple((labels == 10).T)))
            # lidar_data[(labels != 10),:] = 0

            max_z = 4
            # print(np.shape(lidar_data))

            for i in range((np.shape(lidar_data))[0] - 1):
                x = lidar_data[i, 0]
                y = lidar_data[i, 1]
                z = lidar_data[i, 2]
                # print(z)
                lidar_img[int(x), int(y), 1] = 60
                # print(z)

                if 0.7 < z + self.lidar_z < 1.8:
                    lidar_img[int(x), int(y), 0] = 155*bird_array[x, y, 3] * 255
                    self.depth[int(x), int(y)] = z
                    if self.blocking_pointclouds is not None:
                        self.blocking_pointclouds = np.append(self.blocking_pointclouds, np.array([[x, y]]), 0)

                    else:
                        self.blocking_pointclouds = np.array([[x, y]])


            '''if self.blocking_pointclouds is not None:
                #clusters = hcluster.linkage(self.blocking_pointclouds, 'single')

                #clusters = hcluster.fclusterdata(self.blocking_pointclouds, thresh)
                clusters = KMeans(n_clusters=2, random_state=0, max_iter=10).fit(self.blocking_pointclouds).labels_
                n_clusters = np.nanmax(clusters)
                print('clusters',n_clusters)
                for i in range((np.shape(clusters))[0]-1):
                    place = clusters[i]
                    x = self.blocking_pointclouds[i, 0]
                    y = self.blocking_pointclouds[i, 1]
                    #z = self.blocking_pointclouds[i, 2]
                    #print(z)
                    lidar_img[int(x),int(y),1] = place*255/n_clusters
                    #print(z)'''

            # lidar_img[tuple(lidar_data.T)] = (0, 0, 200)
            # lidar_img[tuple((labels == 10).T)] = (255, 0, 0)
            # lidar_img[]
            return lidar_img

    def render(self, display,bird_array):
        if self.points is not None:
            lidar_data = np.array(self.points)
            # print(lidar_data.shape)

            lidar_data[:, :2] *= self.pixel2meter
            lidar_data[:, :2] += (0.5 * self.lidar_image_pixels, 0.5 * self.lidar_image_pixels)
            lidar_data[:, :1] = np.fabs(lidar_data[:, :1])  # pylint: disable=E1111
            # lidar_data = lidar_data.astype(np.int32)

            lidar_data = np.reshape(lidar_data, (-1, 3))

            lidar_img_size = (self.lidar_image_pixels, self.lidar_image_pixels, 3)
            lidar_img = np.zeros((lidar_img_size), dtype=int)
            depth = np.zeros((self.lidar_image_pixels, self.lidar_image_pixels), dtype=float)
            # print(np.shape(tuple((labels == 10).T)))
            # lidar_data[(labels != 10),:] = 0

            max_z = 4
            # print(np.shape(lidar_data))

            for i in range((np.shape(lidar_data))[0] - 1):
                x = lidar_data[i, 0]
                y = lidar_data[i, 1]
                z = lidar_data[i, 2]
                # print(z)
                lidar_img[int(x), int(y), 1] = 100
                # print(z)

                if 0.6 < z + self.lidar_z < 1.8:
                    if bird_array is not None:
                        lidar_img[int(x), int(y), 0] = bird_array[int(x), int(y), 3] * 180

                    #self.depth[int(x), int(y)] = z
                    if self.blocking_pointclouds is not None:
                        self.blocking_pointclouds = np.append(self.blocking_pointclouds, np.array([[x, y]]), 0)

                    else:
                        self.blocking_pointclouds = np.array([[x, y]])

            '''for i in range((np.shape(lidar_data))[0]-1):
                x = lidar_data[i, 0]
                y = lidar_data[i, 1]
                z = lidar_data[i, 2]
                #print(z)
                lidar_img[int(x),int(y),1] = 50
                #print(z)
                
                if 0.7<z+self.lidar_z<1.8:
                    lidar_img[int(x),int(y), 0] = 255
                    self.depth[int(x),int(y)] = z
                    if self.blocking_pointclouds is not None:
                        self.blocking_pointclouds = np.append(self.blocking_pointclouds, np.array([[x,y]]), 0)
                    else:
                        self.blocking_pointclouds = np.array([[x,y]])'''

            '''if self.blocking_pointclouds is not None:
                #clusters = hcluster.linkage(self.blocking_pointclouds, 'single')

                #clusters = hcluster.fclusterdata(self.blocking_pointclouds, thresh)
                clusters = KMeans(n_clusters=2, random_state=0, max_iter=10).fit(self.blocking_pointclouds).labels_
                n_clusters = np.nanmax(clusters)
                print('clusters',n_clusters)
                for i in range((np.shape(clusters))[0]-1):
                    place = clusters[i]
                    x = self.blocking_pointclouds[i, 0]
                    y = self.blocking_pointclouds[i, 1]
                    #z = self.blocking_pointclouds[i, 2]
                    #print(z)
                    lidar_img[int(x),int(y),1] = place*255/n_clusters
                    #print(z)'''

            #lidar_img[tuple(lidar_data.T)] = (0, 0, 200)
            #lidar_img[tuple((labels == 10).T)] = (255, 0, 0)
            #lidar_img[]
            #lidar_img = self.process_data(bird_array)
            self.lidar_img = np.rot90(lidar_img,3)
            self.surface_lidar = pygame.surfarray.make_surface(self.lidar_img)
            self.surface_lidar = pygame.transform.scale(self.surface_lidar,
                                                        (self.lidar_image_pixels * 4, self.lidar_image_pixels * 4)
                                                        )

            if self.surface_lidar is not None:
                display.blit(self.surface_lidar, (200, 460))

    @staticmethod
    def semantic_lidar_callback(weak_self, image):
        #print("lidar callback")
        self = weak_self()
        channels = image.channels
        self.lidar_z = image.transform.location.z
        x = 0
        pointcount = 0
        for i in range(0, channels):
            pointcount += image.get_point_count(i)
            totalpoints = pointcount * 3
        # print(totalpoints)

        complete = np.frombuffer(image.raw_data, dtype=np.dtype('u4'), count=-1)
        #print(totalpoints)
        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'), count=totalpoints)
        self.raw_points = points
        #labels = np.frombuffer(image.raw_data, dtype=np.dtype('u4'), count=totalpoints)

        labels = np.frombuffer(image.raw_data, dtype=np.uint32, offset=8*pointcount, count=pointcount)

        #bigstring = ''.join(chr(i) for i in labels_ascii)
        #bigstring = bigstring.split(',')
        #print('raw', np.shape(image.raw_data))

        #all_points = np.reshape(complete, (int(complete.shape[0] / 8), 8))
        #labels = all_points[:,7]

        self.points = np.reshape(points, (int(points.shape[0] / 3), 3))






class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))



class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self.init_labels()
        self.sensor_front =None
        self.sensor_top = None
        self.sensor_depth = None

        self.surface = None
        self.surface_top = None
        self.surface_depth = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self._camera_transforms = [carla.Transform(carla.Location(x=0, z=2), carla.Rotation(pitch=0,roll=90))]
        self.transform_index = 1
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        self.cam_bp = bp_library.find("sensor.camera.rgb")
        self.cam_bp.set_attribute("image_size_x", f"{500}")
        self.cam_bp.set_attribute("image_size_y", f"{500}")
        self.cam_bp.set_attribute("fov", "90")
        self.spawn_point_front_rgb = carla.Transform(carla.Location(x=2, z=1.2), carla.Rotation(pitch=-10, roll=0))

        self.cam_bp_depth = bp_library.find("sensor.camera.semantic_segmentation")
        self.cam_bp_depth.set_attribute("image_size_x", f"{500}")
        self.cam_bp_depth.set_attribute("image_size_y", f"{500}")
        self.cam_bp_depth.set_attribute("fov", "90")
        self.spawn_point_depth_cam = carla.Transform(carla.Location(x=2, z=1.2), carla.Rotation(pitch=-10, roll=0))

        self.cam_top = bp_library.find("sensor.camera.semantic_segmentation")
        self.cam_top.set_attribute("image_size_x", f"{500}")
        self.cam_top.set_attribute("image_size_y", f"{500}")
        self.cam_top.set_attribute("fov", "100")
        self.spawn_point_top_rgb = carla.Transform(carla.Location(x=11, z=10), carla.Rotation(pitch=-90, roll=0, yaw=0))

        weak_self = weakref.ref(self)
        weak_self2 = weakref.ref(self)
        weak_self_depth = weakref.ref(self)

        self.sensor_front = world.spawn_actor(self.cam_bp, self.spawn_point_top_rgb, attach_to=self._parent)
        self.sensor_top = world.spawn_actor(self.cam_top, self.spawn_point_front_rgb, attach_to=self._parent)
        self.sensor_depth = world.spawn_actor(self.cam_bp_depth, self.spawn_point_depth_cam, attach_to=self._parent)

        self.sensor_front.listen(lambda image: CameraManager._parse_front_image(weak_self, image))
        self.sensor_top.listen(lambda image: CameraManager._parse_top_image(weak_self2, image))
        self.sensor_depth.listen(lambda image: CameraManager._parse_depth_image(weak_self_depth, image))

    def init_labels(self):
        self.semantic_dict = {}
        self.semantic_dict[0] = (0, 0, 0)  # not labeled
        self.semantic_dict[1] = (70, 70, 70)  # building
        self.semantic_dict[2] = (190, 153, 153)  # fence
        self.semantic_dict[3] = (250, 170, 160)  # other
        self.semantic_dict[4] = (220, 20, 60)  # pedestrain
        self.semantic_dict[5] = (153, 153, 153)  # pole
        self.semantic_dict[6] = (157, 234, 50)  # road line
        self.semantic_dict[7] = (128, 64, 128)  # road
        self.semantic_dict[8] = (244, 35, 232)  # side walk
        self.semantic_dict[9] = (107, 142, 35)  # vegetation
        self.semantic_dict[10] = (0, 0, 142)  # car
        self.semantic_dict[11] = (102, 102, 156)  # wall
        self.semantic_dict[12] = (220, 220, 0)  # traffic sign




    def render(self, display):
        if self.surface is not None:
            #display.blit(self.surface, (200, 460))
            #display.blit(self.surface_depth, (200, 460))
            #display.blit(self.surface_top, (800, 460))
            pass

        #if self.surface_depth is not None:
            #display.blit(self.surface_depth, (200, 460))

        if self.surface_top is not None:
            display.blit(self.surface_top, (800, 460))


    @staticmethod
    def _parse_front_image(weak_self, image):
        self = weak_self()
        if not self:
            return

        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    @staticmethod
    def _parse_top_image(weak_self, image):
        self = weak_self()
        if not self:
            return

        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        seg_array = np.zeros((image.height, image.width,3),dtype=np.dtype("uint8"))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        array = array[:,:,0]
        for i in range(len(self.semantic_dict)):
            mask = array==i
            #print(mask.shape)
            seg_array[mask] = self.semantic_dict[i]

        self.surface_top = pygame.surfarray.make_surface(seg_array.swapaxes(0, 1))

        #if self.recording:
            #image.save_to_disk('_out/%08d' % image.frame)

    def _parse_depth_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        R = array[:, :, 0]
        G = array[:, :, 1]
        B = array[:, :, 2]

        normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
        in_meters = 1000 * normalized
        #print(np.shape(in_meters))
        depth = np.array(in_meters, dtype=np.dtype("uint8"))
        self.surface_depth = pygame.surfarray.make_surface(depth.swapaxes(0, 1))

        #self.surface_depth = pygame.surfarray.make_surface(in_meters.swapaxes(0, 1))