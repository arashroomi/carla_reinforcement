from __future__ import print_function
from keyboard import KeyboardControl


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
import pure_pursuit
import torch.multiprocessing as mp
from torch.multiprocessing import set_start_method

try:
    sys.path.append(glob.glob('./carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla
from carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions, BirdView, DEFAULT_HEIGHT,DEFAULT_WIDTH
#from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
#from agents.navigation.roaming_agent import RoamingAgent  # pylint: disable=import-error
#from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error

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

from pygame.locals import (

    K_UP,

    K_DOWN,

    K_LEFT,

    K_RIGHT,

    K_ESCAPE,

    KEYDOWN,

    QUIT,

)
# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_client, hud,n_memory_frames,frame_stack,state_dimention_height, state_dimention_width):
        self.camera_enabled = True
        self.camera_start = True
        self.state_dimention_height = state_dimention_height
        self.state_dimention_width = state_dimention_width
        self.hud = hud
        carla_world = carla_client.get_world()
        self.world = carla_client.load_world('Town01')
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.no_rendering_mode = not self.camera_start
        settings.fixed_delta_seconds = 1/20
        self.frame_stack = frame_stack

        self.world.apply_settings(settings)
        self.carla_client = carla_client
        self.world = carla_world
        self.actor_role_name = 'rlcarla'
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        weather = carla.WeatherParameters(
            cloudiness=20.0,
            precipitation=0.0,
            sun_altitude_angle=-30,
            fog_density=1000,
            fog_falloff=400,
            fog_distance=0 )

        '''weather = carla.WeatherParameters(
            cloudiness=20.0,
            precipitation=0.0,
            sun_altitude_angle=40,
            fog_density=0,
            fog_falloff=0,
            fog_distance=0 )'''

        self.world.set_weather(weather)
        settings.synchronous_mode = True
        settings.no_rendering_mode = not self.camera_start
        settings.fixed_delta_seconds = 1/20
        self.frame_stack = frame_stack

        self.world.apply_settings(settings)

        lmanager = self.world.get_lightmanager()
        mylights = lmanager.get_all_lights()
        lights = lmanager.get_light_group(mylights)
        ##print(lights)
        lmanager.turn_off(mylights)
        #lmanager.set_intensity(mylights, 0)


        # Custom a specific light
        for i in range(len(mylights)):
            light01 = mylights[i]
            light01.turn_off()
            light01.set_intensity(0)
            state01 = carla.LightState(active=False)
            light01.set_light_state(state01)



        self.carla_client = carla_client
        self.world = carla_world
        self.actor_role_name = 'rlcarla'
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.player = None
        self.collision_sensor = None

        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self.blueprint_library = self.world.get_blueprint_library()
        self.blueprint = self.blueprint_library.filter("model3")[0]
        self.blueprint.set_attribute('role_name', 'rlcarla')

        self.n_frames = n_memory_frames


        self.first_restart = True
        self.restart()



        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.SECONDS_PER_EPISODE = 120
        self.walkers = []
        self.pedestrian_relative_distance = None


        self.done = False




    def restart(self):
        torch.cuda.empty_cache()
        #self.destroy()
        self.player_max_speed = 50/3.6
        self.player_max_speed_fast = 60/3.6
        self.SumReward = 0
        self.player = None
        if self.camera_start == True:
            self.camera_enabled == True
            settings = self.world.get_settings()
            settings.no_rendering_mode = False
            self.world.apply_settings(settings)
        else:
            self.camera_enabled == False
            settings = self.world.get_settings()
            settings.no_rendering_mode = True
            self.world.apply_settings(settings)



        self.done = False

            #spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        while self.player is None:
            self.initial_pose = carla.Transform(carla.Location(x=212.764572, y=57.5, z=0.3),
                                        carla.Rotation(pitch=0.000000, yaw=-180, roll=0.000000))
            self.player = self.world.try_spawn_actor(self.blueprint, self.initial_pose)
            time.sleep(0.001)
        # Set up the sensors.

        self.player.set_light_state(carla.VehicleLightState.HighBeam)
        #self.player.set_autopilot(True)


        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        if self.camera_enabled:
            self.camera_manager = CameraManager(self.player, self.hud)

        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)
        #self.birds_eye_data = birds_eye(self.player,self.carla_client)
        self.begining = True

        #initial_control = carla.VehicleControl()
        #initial_control.throttle = 0.2
        #self.player.apply_control(initial_control)
        snap = self.world.get_snapshot()
        #self.player.set_autopilot(True)
        self.simulation_start_time = snap.timestamp.elapsed_seconds

        if self.first_restart:
            self.birds_eye_data = birds_eye(self.player, self.carla_client,width=self.state_dimention_width,
                                            height=self.state_dimention_height,memory_frames=self.n_frames, frame_stack=self.frame_stack)
            self.first_restart = False

        self.place_side_vehicles(1)
        print('reset')
        return self.player

    def place_side_vehicles(self,number_of_vehicles):
        blueprint_side = self.blueprint_library.filter("model3")[0]
        self.static_players_list = []
        e = random.randint(-20,20)
        ee = random.randint(2,5)

        for i in range(number_of_vehicles):
            role_name = 'static_' + str(i)
            blueprint_side.set_attribute('role_name', role_name)
            self.static_vehicle_pose = carla.Transform(carla.Location(x=180-e, y=54.3, z=0.3),
                                        carla.Rotation(pitch=0.000000, yaw=-179.999939, roll=0.000000))
            self.static_players_list.append(self.world.try_spawn_actor(blueprint_side, self.static_vehicle_pose))


            self.static_vehicle_pose2 = carla.Transform(carla.Location(x=130-e, y=54.3, z=0.3),
                                        carla.Rotation(pitch=0.000000, yaw=-179.999939, roll=0.000000))
            self.static_players_list.append(self.world.try_spawn_actor(blueprint_side, self.static_vehicle_pose2))

            self.static_vehicle_pose3 = carla.Transform(carla.Location(x=210+e, y=54.3, z=0.3),
                                        carla.Rotation(pitch=0.000000, yaw=-179.999939, roll=0.000000))
            self.static_players_list.append(self.world.try_spawn_actor(blueprint_side, self.static_vehicle_pose3))

            self.static_vehicle_pose = carla.Transform(carla.Location(x=800, y=54.3, z=0.3),
                                        carla.Rotation(pitch=0.000000, yaw=-179.999939, roll=0.000000))
            self.static_players_list.append(self.world.try_spawn_actor(blueprint_side, self.static_vehicle_pose))


            self.static_vehicle_pose2 = carla.Transform(carla.Location(x=60, y=54.3, z=0.3),
                                        carla.Rotation(pitch=0.000000, yaw=-179.999939, roll=0.000000))
            self.static_players_list.append(self.world.try_spawn_actor(blueprint_side, self.static_vehicle_pose2))

            self.static_vehicle_pose3 = carla.Transform(carla.Location(x=70, y=54.3, z=0.3),
                                        carla.Rotation(pitch=0.000000, yaw=-179.999939, roll=0.000000))
            self.static_players_list.append(self.world.try_spawn_actor(blueprint_side, self.static_vehicle_pose3))
    def load_walkers(self,walker_list):
        self.walkers = walker_list

    def walker_to_vehicle_distance(self):
        vehicle_transform = self.player.get_transform()
        vehicle_loc = vehicle_transform.location
        vehicle_xyz = np.array([vehicle_loc.x, vehicle_loc.y, vehicle_loc.z])
        vehicle_rot = vehicle_transform.rotation
        yaw = math.pi*vehicle_rot.yaw/180
        self.yaw = yaw
        i = 0
        near_collision = False
        near_car = False

        for car in self.static_players_list:
            if not car==None:
                car_transform = car.get_transform()
                loc = car_transform.location
                car_xyz = np.array([loc.x, loc.y, loc.z])
                R = transforms3d.euler.euler2mat(0, 0, yaw).T
                car_loc_relative = np.dot(R, car_xyz - vehicle_xyz)
                x_to_car = car_loc_relative[0]
                y_to_car = car_loc_relative[1]
                # self.pedestrian_relative_distance[i] = ped_loc_relative
                if ((3 < x_to_car < random.randrange(27,40)) & (-10 < y_to_car < 10)):
                    near_car = True
                i = i + 1
                x_to_ped = x_to_car


        for walker in self.walkers:
            walker_transform = walker.get_transform()
            loc = walker_transform.location
            walker_xyz = np.array([loc.x, loc.y, loc.z])
            R = transforms3d.euler.euler2mat(0, 0, yaw).T
            ped_loc_relative = np.dot(R, walker_xyz - vehicle_xyz)
            x_to_car = ped_loc_relative[0]
            y_to_car = ped_loc_relative[1]
            #self.pedestrian_relative_distance[i] = ped_loc_relative
            if ((0<x_to_car < random.randrange(15,20)/4 ) &(-random.randrange(2,3)<y_to_car<random.randrange(4,6)/1.5)):
                near_collision = True
            i = i + 1
            x_to_ped = x_to_car
            #print(x_to_ped)

        return near_collision, x_to_car,near_car

    def reward(self,action):
        #self.player.set_autopilot(True)

        colhist = self.collision_sensor.get_collision_history()

        v = self.player.get_velocity()
        mps = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        kmh = int(3.6 * mps)
        reward = 0
        obstacle_on_way, x_to_ped,near_car = self.walker_to_vehicle_distance()

        collision = 0


        snap = self.world.get_snapshot()
        simulation_time = snap.timestamp.elapsed_seconds
        loc = self.player.get_location()
        initial_loc = self.initial_pose.location
        self.distance = math.sqrt((initial_loc.x - loc.x)**2 + (initial_loc.y - loc.y)**2 + (initial_loc.z - loc.z)**2)
        self.distance = (simulation_time - self.simulation_start_time)*v





        distance = self.distance.x

        if len(colhist) > 1:
            self.done = True
            reward = -8_000
            #print("accident")
            self.collision = 1

        if len(colhist) < 1:
            self.done = False
            reward = ((2*int(kmh) - 6))

        if abs(distance) > 120:
            self.done = True
            reward = reward + 3000
        '''if x_to_ped < -3:
            self.done = True
            reward = reward + 3000
            print('passed')'''

        if self.SECONDS_PER_EPISODE < simulation_time - self.simulation_start_time:
            self.done = True
            reward = reward - 3000






        normalized_reward = (reward / 9000)

        self.SumReward = self.SumReward + normalized_reward





        self.SumReward = self.SumReward + normalized_reward
            # print("sum of rewards:", self.SumReward)
        self.hud.sum_reward = self.SumReward
        return normalized_reward, self.done, collision


    def tick(self, clock):

        self.world.tick()
        self.hud.tick(self, clock)
        framed_state,visibility_state = self.birds_eye_data.birdeyeData(self.player)
        snap = self.world.get_snapshot()
        if self.begining ==True:
            self.episode_start = snap.timestamp.elapsed_seconds
            self.begining = False
        return framed_state, visibility_state


    def render(self, display):
        self.hud.render(display)
        self.birds_eye_data.render(display)
        if self.camera_enabled: self.camera_manager.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor2.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.sensor2 = None
        self.camera_manager.index = None


    def destroy(self):
        sensors = [
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor]
        if self.camera_enabled:
            sensors = [
                self.collision_sensor.sensor,
                self.lane_invasion_sensor.sensor,
                self.camera_manager.sensor,
                self.camera_manager.sensor2]

        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()
        if self.static_players_list is not None:
            for vehicle in self.static_players_list:
                if vehicle is not None:
                    vehicle.destroy()


    def camera_toggle(self):

        for event in pygame.event.get():


            if event.type == pygame.KEYUP:
                if event.key == K_F1:
                    self.camera_start = True
                    print('pressed')
                if event.key == K_F2:
                    self.camera_start = False
                    print('pressed')


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================
class birds_eye(object):
    def __init__(self,parent_actor,client,memory_frames,frame_stack,width, height):
        self.frame_stack = frame_stack
        self.bird_eye_window_width = width
        self.bird_eye_window_height = height
        #self.buffer = torch.zeros(3, self.bird_eye_window_width, self.bird_eye_window_height).to(device)
        self.FRONT_FOV = 2 * (math.pi) / 3
        self._parent = parent_actor
        self.surface = None
        self._carla_client = client
        self.state_frames = memory_frames
        self.pedestrian_birdview = np.full((self.state_frames, self.bird_eye_window_width,self.bird_eye_window_height), 0, dtype=np.float32)
        self.pedestrian_birdview_stack = np.full((self.frame_stack * self.state_frames, self.bird_eye_window_width,self.bird_eye_window_height), 0, dtype=np.float32)
        self.pedestrian_birdview_embeded = np.full((self.bird_eye_window_width, self.bird_eye_window_height), 0, dtype=np.float32)

        self.birdview_producer = BirdViewProducer(
            self._carla_client,  # carla.Client
            target_size=PixelDimensions(width=self.bird_eye_window_width, height=self.bird_eye_window_height),
            pixels_per_meter=3,
            crop_type=BirdViewCropType.FRONT_AREA_ONLY, render_lanes_on_junctions=True,
        )
        #self.birdeyeData(parent_actor)


    def birdeyeData(self, player):
        self._parent = player
        birdview = self.birdview_producer.produce(agent_vehicle=self._parent)
        rgb = BirdViewProducer.as_rgb(birdview)
        copyrgb = rgb.copy()
        n, m = np.shape(birdview[:, :, 1])
        x, y = np.ogrid[0:n, 0:m]
        agent_pixel_pose_Y = int(self.bird_eye_window_width / 2)
        agent_pixel_pose_X = int(self.bird_eye_window_height-8)
        agent_pixel_pose = [agent_pixel_pose_X, agent_pixel_pose_Y]
        # print (agent_pixel_pose)

        fov_mask = (((agent_pixel_pose_Y - y) < ((agent_pixel_pose_X - x) * math.tan(self.FRONT_FOV / 2))) & (
                    (agent_pixel_pose_Y - y) > (-(agent_pixel_pose_X - x) * math.tan(self.FRONT_FOV / 2))))

        self.mask_shadow, uncertainty,d = shadow_mask.view_field(birdview[:, :, 9] + 2.7*birdview[:, :, 3], 1, agent_pixel_pose,30)
        #(array[self.mask_shadow & mask]) = 255

        #copyrgb = rgb.copy()
        shadow = np.zeros((n, m))
        visibility_matrix = fov_mask & self.mask_shadow
        shadow[visibility_matrix] = 1

        uncertainty = uncertainty * shadow
        uncertainty = uncertainty.astype(np.bool_)

        '''r= copyrgb[:, :, 0]
        g = copyrgb[:, :, 1]
        b = copyrgb[:, :, 2]
        r[uncertainty] = 0
        g[uncertainty] = 0
        b[uncertainty] = 0'''

        copyrgb[:, :, 1] = copyrgb[:, :, 1]
        #copyrgb[:,:,0] = copyrgb[:,:,0]
        n, m = np.shape(birdview[:, :, 1])
        embedded_birdseye = np.zeros((n, m))

        birdview[:, :, 8]=birdview[:, :, 8]*uncertainty


        for i in range(self.state_frames * self.frame_stack-1,-1,-1):
            if i == 0:
                self.pedestrian_birdview_stack[0, :, :] = np.multiply(shadow, birdview[:, :, 8])
            else:
                self.pedestrian_birdview_stack[i,:, :] = self.pedestrian_birdview_stack[i-1,:,:]

            #print(i)
            if i % self.frame_stack == 0:
                self.pedestrian_birdview[int(i//self.frame_stack), :, :] = self.pedestrian_birdview_stack[i,:, :]

                embedded_birdseye = self.pedestrian_birdview[int(i//self.frame_stack), :, :] \
                                    * (0.5**(int(i//self.frame_stack))) + embedded_birdseye



        '''copyrgb = copyrgb.swapaxes(0, 1)
        self.rgb_nyg_surf = pygame.surfarray.make_surface(copyrgb)
        self.rgb_nyg_surf = pygame.transform.scale(self.rgb_nyg_surf, (self.bird_eye_window_width*4, self.bird_eye_window_height*4))
        '''
        visibility_int = np.zeros((1, n, m))
        visibility_int[0, uncertainty] = 1
        cir = np.where(d<= 30, int(1), shadow)
        cir = np.where(d > 30, int(0), cir)






        self.pedestrian_birdview_embeded = np.add(255*birdview[:, :, 8], 0.5 * self.pedestrian_birdview_embeded)
        obsv= (0.01*copyrgb)
        obsv[:, :, 0] = obsv[:, :, 0] + 50 * uncertainty
        obsv[:,:,2]= obsv[:,:,2]+200*self.pedestrian_birdview_stack[0, :, :]
        obsv[:,:,1]= obsv[:,:,1]+200*self.pedestrian_birdview_stack[0, :, :]+100*(birdview[:, :, 3] * cir)
        birdview[:, :, 1] * fov_mask
        obsv = obsv.swapaxes(0, 1)
        self.rgb_obsv_surf = pygame.surfarray.make_surface(obsv)
        self.rgb_obsv_surf = pygame.transform.scale(self.rgb_obsv_surf, (self.bird_eye_window_width*4, self.bird_eye_window_height*4))


        copyrgb[:,:,0] = 255 * embedded_birdseye  + copyrgb[:,:,0]
        copyrgb = copyrgb.swapaxes(0, 1)
        self.rgb_nyg_surf = pygame.surfarray.make_surface(copyrgb)
        self.rgb_nyg_surf = pygame.transform.scale(self.rgb_nyg_surf, (self.bird_eye_window_width*4, self.bird_eye_window_height*4))

        self.tensor = pygame.surfarray.make_surface(embedded_birdseye)
        pygame.transform.flip(self.tensor,True,True)
        self.tensor = pygame.transform.rotozoom(self.tensor,90,4)


        return self.pedestrian_birdview, visibility_int


    def render(self, display):
        if self.rgb_nyg_surf is not None:
            display.blit(self.rgb_nyg_surf, (350, 0))
            display.blit(self.rgb_obsv_surf, (900, 0))

        if self.tensor is not None:
            display.blit(self.tensor, (350, 350))
        white = (255, 255, 255)
        green = (0, 255, 0)
        blue = (0, 0, 128)

        # set the pygame window name
        pygame.display.set_caption('Show Text')

        font = pygame.font.Font('freesansbold.ttf', 20)

        text2 = font.render('Augmented Partial Observation', True, green, blue)
        text = font.render('Full State' , True, green, blue)
        text3 = font.render('CARLA Front Camera' , True, green, blue)
        text4 = font.render('CARLA Front Camera' , True, green, blue)
        textRect = text.get_rect()

        textRect.center = (500, 360)
        display.blit(text, textRect)
        textRect = text.get_rect()

        textRect2 = text2.get_rect()
        textRect2.center = (1100, 360)
        display.blit(text2, textRect2)

        textRect3 = text3.get_rect()
        textRect3.center = (450, 1000)
        display.blit(text3, textRect3)

        textRect4 = text4.get_rect()
        textRect4.center = (1100, 1000)
        display.blit(text4, textRect4)


    def push_to_tensor(self, intensor, x):
        tensor = intensor.clone()
        tensor[:-1,:, :] = intensor[1:,:,:]
        tensor[:-1,:, :] = x[:,:]
        return tensor





class VehicleControl(object):
    """Class that handles keyboard input."""
    def __init__(self, world):
        self._world = world
        self._control = carla.VehicleControl()
        self._lights = carla.VehicleLightState.NONE
        world.player.set_light_state(self._lights)
        #self._steer_cache = 0.0
        # #self._control.throttle = 0.2
        #self._control.brake = 0.0
        # self._control_cache = 1
        self.vehicle = world.player
        self.t_previous = 0
        self._kP = 1
        self._kI = 0.2
        self._kD = .01
        self.i_term_previous = 0





    def speed_setpoint_actions(self, action,clock):

        obstacle_on_way, x_to_ped, near_car = self._world.walker_to_vehicle_distance()
        #action = random.choice((4, 3, 3,3,3,3,3))
        action = random.choice((0, 1, 1, 1, 1, 1, 1))


        if near_car:
            #action = random.choice((0,1,1,1,1,2,2, 2))
            pass
        if obstacle_on_way:
            action = 0
        self.action = action



        if (action == 0):
            v_desired = -1
        if (action == 1):
            v_desired = 8/3.6
        elif(action==2):
            v_desired = 20/3.6
        elif (action == 3):
            v_desired = 30/3.6
        elif (action == 4):
            v_desired = 40/3.6

        millisec = clock.get_time()
        sec = millisec/1000

        self.PI_controller(v_desired,sec)


    def set_throttle(self, input_throttle):
        # Clamp the throttle command to valid bounds
        throttle = np.fmax(np.fmin(input_throttle, 1.0), 0.0)
        return throttle
    def set_brake(self, input_brake):
        # Clamp the steering command to valid bounds
        brake = np.fmax(np.fmin(input_brake, 1.0), 0.0)
        return brake


    def PI_controller(self, v_desired,clock):

        v = self._world.player.get_velocity()
        mps = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        kmh = int(3.6 * mps)

        t = clock
        #print(clock)

        time_step = t - self.t_previous
        speed_error = v_desired - mps
        k_term = self._kP*speed_error
        i_term = float(self.i_term_previous) + self._kI*time_step*speed_error
        self.i_term_previous = i_term
        throttle_output = k_term + i_term

        self.throttle = self.set_throttle(throttle_output)  # in percent (0 to 1)
        if throttle_output<-0.0:
            self.brake = self.set_brake( abs(throttle_output))
            self.throttle = 0
            #print(brake_output)
        else:
            brake_output = 0
            self.brake = 0

        if v_desired == -1:
            self.brake = 1
            self.throttle = 0

        self._control.brake = self.brake
        self._control.throttle = self.throttle

        self._world.player.apply_control(self._control)



# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 16)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 16 if os.name == 'nt' else 16)
        self._notifications = FadingText(font, (width, 60), (0, height - 40))
        self.server_fps = 0
        self.client_fps = 0
        self.s_p_time = 0
        self.frame = 0
        self.simulation_time = 0
        self.computer_time = 0
        self._flag = False
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()
        self.SECONDS_PER_EPISODE = 90
        self.vhist = []

        self.sum_reward = 0

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds
        self._flag = False

    def tick(self, world, clock):

        self._notifications.tick(world, clock)
        world.camera_toggle()
        if not self._show_info:
            return
        if not self._flag:
            c = time.time()
            delta_time_c = c - self.computer_time
            delta_time_s = self.simulation_time - self.s_p_time
            self.x_faster = delta_time_s/delta_time_c
            self.computer_time = c
            self.s_p_time = self.simulation_time
            self._flag = True

        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]




        vehicles = world.world.get_actors().filter('vehicle.*')
        self.client_fps = clock.get_fps()
        self._info_text = [

            'Speed:   % 16.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),

            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision]


    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))


                v_offset += 18
        self._notifications.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """Helper class to handle text output using pygame"""
    def __init__(self, font, width, height):

        self.font = font
        self.line_space = 18

        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)



# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


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
        #print(actor_type)
        #self.hud.notification('Collision with %r' % actor_type)

        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


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



# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop():
    pygame.init()
    pygame.font.init()
    world = None
    action = 0
    args = bain()
    action = actions()
    action.longitudinal = actions().ACCELERATE

    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(2.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client, hud)
        #controller = VehicleControl(world)


        clock = pygame.time.Clock()
        #mpc_controller = pure_pursuit.mpc_controller(world.world, world.player, clock,hud.client_fps)
        #controller.actions(action, clock)
        #mpc_controller.step(clock)
        action = actions()
        #action.longitudinal = action.ACCELERATE
        controller = KeyboardControl(world, args.autopilot)
        world.restart()



        while True:
            clock.tick_busy_loop()
            world.tick(clock)
            if controller.actions(action, clock):
                return


            world.render(display)
            pygame.display.flip()
            pass


    finally:

        if (world and world.recording_enabled):
            client.stop_recorder()


        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def bain():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        default=False,
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1600x1600',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)
    return args

# ==============================================================================
# -- controllers() --------------------------------------------------------------------
# ==============================================================================

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.surface = None
        self.surface2 = None
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

        self.cam_top = bp_library.find("sensor.camera.rgb")
        self.cam_top.set_attribute("image_size_x", f"{500}")
        self.cam_top.set_attribute("image_size_y", f"{500}")

        self.cam_top.set_attribute("fov", "100")
        self.spawn_point = carla.Transform(carla.Location(x=11, z=10), carla.Rotation(pitch=-90, roll=0, yaw=0))

        self.spawn_point2 = carla.Transform(carla.Location(x=2, z=1.2), carla.Rotation(pitch=-10, roll=0))
        weak_self = weakref.ref(self)
        weak_self2 = weakref.ref(self)
        self.sensor = world.spawn_actor(self.cam_bp, self.spawn_point, attach_to=self._parent)
        self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))

        self.sensor2 = world.spawn_actor(self.cam_top, self.spawn_point2, attach_to=self._parent)
        self.sensor2.listen(lambda image: CameraManager._parse_image2(weak_self2, image))


    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (200, 460))
            display.blit(self.surface2, (800, 460))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    @staticmethod
    def _parse_image2(weak_self, image):
        self = weak_self()
        if not self:
            return
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface2 = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        #if self.recording:
            #image.save_to_disk('_out/%08d' % image.frame)
