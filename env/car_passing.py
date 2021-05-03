import objects
import random
import numpy as np
import math
import shadow_mask
import pygame
from enum import IntEnum, auto, Enum
from pathlib import Path
from typing import List
from filelock import FileLock
import time

from typing import NamedTuple, List, Tuple


class Dimensions(NamedTuple):
    width: int
    height: int


PixelDimensions = Dimensions


class World:
    def __init__(self):
        self.number_of_pedestrians = 1

        self.clock = 0
        self.SECONDS_PER_EPISODE = 25

        self.pedestrian_list = []
        self.actor_list = []
        self.building_list = []
        self.blocked_spwan_list = []
        self.visible_spwan_list = []

        self.ttc = [0.5,1,1.5, 2, 2.5, 3,3.5,4]
        self.vehicle_max_speed = 30 / 3.6
        self.ped_size = 1
        self.passage_len = 5
        self.side_walk_len = 1.5
        self.road_weidth = 3.5
        self.building_width = 10
        self.city_width = 50
        self.training_portion = 90
        self.min_building_len = 3
        self.block_len = 30
        self.n_blocks = 1
        self.fps = 10
        self.moving_ratio = 0.5
        self.speed_limit = 30

        self.max_ped_speed = 5
        self.min_ped_speed = 2
        self.min_ped_onroad = 10
        self.max_ped_onroad = 12

        self.built_space_x = [[0, 50]]
        self.road_len = 2*42
        self.col_data= np.full((2, len(self.ttc), 3), 0, dtype=np.float32)
        self.col_data[:,:,2] = 1
        self.start_time = 0
        self.done = False
        self.passing = True
        #self.start_world()
        self.avg_speed = 0
        self.vehicle_max_accelration = 5.8
        self.t_inevitable_crash = self.vehicle_max_speed/self.vehicle_max_accelration
        #print('time to inevitable accident:',self.t_inevitable_crash)
        self.camera_offset = 0



    def start_world(self):
        #print('new episode started')
        self.blocked = 0
        self.clock = 0
        self.SumReward = 0
        self.collision = 0
        self.to_ped = 1000
        self.vehicle = objects.Car(self.clock, x=0, y=0, v=self.vehicle_max_speed )


        if random.random()>0.3:

            self.building_present = 1
            self.create_map(start=int(self.road_len/4), end=2*int(self.road_len/4),height=self.building_present)
            if self.blocked_spwan_list is not None:
                if random.random() > 0.6:
                    blocked = 1
                    self.blocked = 1
                    self.create_walkers(self.blocked_spwan_list, len(self.blocked_spwan_list), self.moving_ratio,blocked)
                else:
                    blocked = 0
                    self.create_walkers(self.blocked_spwan_list, len(self.blocked_spwan_list), self.moving_ratio,
                                        blocked)

            #print('created blocked peds:', len(self.pedestrian_list))
        else:
            self.building_present = 0
            self.create_map(start=int(self.road_len/4), end=2*int(self.road_len/4),height=self.building_present)
            if self.blocked_spwan_list is not None:
                blocked = 0
                self.create_walkers(self.blocked_spwan_list, len(self.blocked_spwan_list), self.moving_ratio,blocked)

            #print('created non blocked peds:', len(self.pedestrian_list))

            #self.create_map(start=int(self.road_len / 4), end=2 * int(self.road_len / 4), height=1)
            #self.visible_spwan_list = [[random.uniform(self.road_len/4,2*self.road_len/4), self.side_walk_len + self.road_weidth -self.ped_size]]
            #self.create_walkers(self.visible_spwan_list, len(self.visible_spwan_list), self.moving_ratio, False)
            #print('not blocked:', len(self.visible_spwan_list))




    def tick(self):
        self.clock = self.clock + (1 / self.fps)
        self.elapsed_time = time.time() - self.start_time
        self.start_time = time.time()
        self.fps_game = 1/self.elapsed_time
        # self.vehicle.x = self.vehicle.x+1
        self.vehicle.calculate_location(self.clock)
        # print(self.vehicle.x,self.vehicle.v,self.vehicle.acceleration)
        for ped in self.pedestrian_list:
            self.ped_control(ped)
            ped.calculate_location(self.clock)
            self.to_ped = min(self.to_ped,ped.x_to_agent)

            if ped.collision:
                self.vehicle.col_history.append(ped)



    def render(self, display, surface):
        white = (255, 255, 255)
        green = (0, 255, 0)
        blue = (0, 0, 128)


        # set the pygame window name
        pygame.display.set_caption('Show Text')

        font = pygame.font.Font('freesansbold.ttf', 16)
        text = font.render('FPS: '+str(int(self.fps_game)), True, green, blue)
        textRect = text.get_rect()
        textRect.center = (110, 360)
        display.blit(text, textRect)

        display.blit(surface, (0, 0))

        pygame.display.update()

    def ped_control(self, ped):
        v_walker = ped.v_walking

        direction = [0, -1]

        x_to_vehicle = ped.x_to_agent
        y_to_vehicle = ped.y_to_agent
        self.y_to_vehicle = y_to_vehicle
        xtc = self.xtc

        v = 0
        #print(ped.x,x_to_vehicle,self.vehicle.x,xtc)
        if abs(x_to_vehicle) < xtc:
            v = -v_walker
            if self.passing == False:
                v = 0
            #print('herreee')

              # clock count standing in the middle of road

            if (y_to_vehicle) < 0.05 and ped.reserve_atr1 >= ped.reserve_atr2:
                v = 0
                ped.reserve_atr2 = ped.reserve_atr2 + 1 / self.fps



            if ped.reserve_atr1 < ped.reserve_atr2:
                v = -v_walker/4
                pass
        if abs(y_to_vehicle) >7  and ped.reserve_atr1 < ped.reserve_atr2:
            v = 0
            ped.z = 0
        ped.apply_control(v, direction)

    def create_walkers(self, spawn_points, n_ped, moving_ratio,blocked):

        ped_points = random.sample(spawn_points, n_ped)
        for i in range(n_ped):
            x, y = ped_points[i]
            # print('here')
            if blocked==0:
                x = random.uniform(self.road_len/4,2*self.road_len/4)
                y = self.side_walk_len + self.road_weidth -1.5*self.ped_size

            ped = objects.Walker(x, y,self.vehicle)
            choice = random.randint(0, len(self.ttc) - 1)
            ped.ttc = self.ttc[choice]
            self.chosen_ttc = ped.ttc

            ttc = self.t_inevitable_crash
            self.xtc = -0.5 * self.vehicle_max_accelration * (ttc ** 2) + self.vehicle_max_speed * ttc + (
                    ped.ttc * self.vehicle_max_speed)
            if random.random() < moving_ratio:
                ped.passing = True
                self.passing = True
                # waiting time
                ped.reserve_atr1 = random.uniform(self.min_ped_onroad, self.max_ped_onroad)
                ped.v_walking = random.uniform(self.min_ped_speed, self.max_ped_speed)

                # print('ttc:',ped.ttc)
            else:
                ped.passing = False
                self.passing = False
                ped.reserve_atr1 = 0


            ped.blocked = blocked
            self.SECONDS_PER_EPISODE = self.SECONDS_PER_EPISODE + ped.reserve_atr1

            self.pedestrian_list.append(ped)
            self.actor_list.append(ped)

    def create_map(self, start, end,height):
        block_width = 6
        min_openview_space = 10
        maped_x = 0
        block_places = range(int(self.road_len/4), 2*int(self.road_len/4), block_width)
        free_passage_place = random.choice(block_places)
        #print(free_passage_place)
        for i in  block_places:
            if i!=free_passage_place:
                building = objects.Building(x1=i,y1=self.side_walk_len + self.road_weidth,length=block_width,
                                            width=self.building_width)
                building.z = height
                self.building_list.append(building)

            else:
                self.blocked_spwan_list.append([i  + 2*self.ped_size, self.side_walk_len + self.road_weidth + 1*self.ped_size])



    def reward(self, action):

        colhist = self.vehicle.col_history

        mps = self.vehicle.v
        kmh = int(3.6 * mps)
        reward = 0

        distance = self.vehicle.x

        if len(colhist) > 0:
            self.done = True
            reward = -8_000
            #print("accident")
            self.collision = 1


        if len(colhist) < 1:
            self.done = False
            reward = ((2*int(kmh) - 6))
            if 1<self.to_ped < 10 and abs(self.y_to_vehicle)<0.5:
                reward = (20*(-int(kmh)))



        if distance > 1*self.road_len/2+4:
            self.done = True
            reward = reward + 3000
        if self.to_ped < -3:
            self.done = True
            reward = reward + 3000
            #reward = reward + 1000
            print('here')
        if self.SECONDS_PER_EPISODE < self.clock:
            self.done = True
            # reward = distance
        self.avg_speed = distance/self.clock


        normalized_reward = (reward / 8000)

        self.SumReward = self.SumReward + normalized_reward
        # print("sum of rewards:", self.SumReward)
        # self.hud.sum_reward = self.SumReward
        return normalized_reward, self.done, self.collision

    def data_metrics(self):
        for ped in self.vehicle.col_history:
            if ped.blocked:
                for i in range(len(self.ttc)):
                    if ped.ttc == self.ttc[i]:
                        self.col_data[0,i,0] = self.clock
                        self.col_data[0, i, 1] = self.col_data[0, i, 1] + 1

            if not ped.blocked:
                for i in range(len(self.ttc)):
                    if ped.ttc == self.ttc[i]:
                        self.col_data[1,i,0] = self.clock
                        self.col_data[1, i, 1] = self.col_data[0, i, 1] + 1







class BirdEye:
    def __init__(self, x_size_pixel: int, y_size_pixel: int, pixel_per_meter: int, world: World
                 , n_frame_dropout, frame_stack,fov_radians,max_detaction_distance):
        self.max_detaction_distance = max_detaction_distance
        self.n_frame_dropout = n_frame_dropout
        self.FRONT_FOV = 40*math.pi/180
        self.camera_offset = 0
        self.state_frames = frame_stack
        self.x_size_pixel = x_size_pixel
        self.y_size_pixel = y_size_pixel
        self.pixel_shift_x = 0
        self.n_layers = 2
        self.pixel_shift_y = int(y_size_pixel / 2)

        self.x_window_meters = int(x_size_pixel / pixel_per_meter)
        self.y_window_meters = int(y_size_pixel / pixel_per_meter)
        self.rendering_window = None

        self.pedestrian_birdview = np.full((self.state_frames, self.x_size_pixel, self.y_size_pixel), 0,
                                           dtype=np.float32)
        self.pedestrian_birdview_stack = np.full(
            (self.n_frame_dropout * self.state_frames, self.x_size_pixel, self.y_size_pixel), 0, dtype=np.float32)

        self.pixel_per_meter = pixel_per_meter
        self.world = world
        self.vehicle = world.vehicle
        self.building_list = world.building_list
        #canvas = self.produce_map()

        self.building_canvas = self.produce_map()


        self.mask = np.full((self.n_layers, x_size_pixel, y_size_pixel), 0, dtype=np.float32)

    def produce_mask(self):
        self.camera_offset = self.world.camera_offset

        self.mask[0, :, :] = self.update_pedestrians()
        self.mask[1, :, :] = self.update_building_mask(self.building_canvas)
        return self.mask

    def produce_map(self):
        canvas = self.make_empty_canvas()
        for building in self.world.building_list:
            canvas = self.plot_on_pixel(self.vehicle, building, canvas)
        return canvas
    def update_ego(self,rgb,d,w):
        x1 = -math.ceil(self.vehicle.length/2) * self.pixel_per_meter+d
        x2 = math.ceil(self.vehicle.length / 2) * self.pixel_per_meter
        y1 = math.ceil(self.vehicle.width / 2) * self.pixel_per_meter + int(w/2)
        y2 = -math.ceil(self.vehicle.width / 2) * self.pixel_per_meter + int(w/2)
        rgb[x1:d, y2:y1, 2] = 255
        return rgb

    def update_building_mask(self,canvas):

        mask = self.mask_crop(canvas)
        return mask





    def update_pedestrians(self):
        canvas = self.make_empty_canvas()
        for ped in self.world.pedestrian_list:
            canvas = self.plot_on_pixel(self.vehicle, ped, canvas)
            # print('here')

        mask = self.mask_crop(canvas)
        return mask


    def mask_crop(self, canvas):
        self.x_up = self.vehicle.x + self.x_window_meters
        self.x_down = self.vehicle.x
        self.y_positive = self.vehicle.y + self.y_window_meters / 2
        self.y_negative = self.vehicle.y - (self.y_window_meters / 2)

        x2 = int(self.x_up * self.pixel_per_meter) + self.pixel_shift_x
        x1 = -self.x_size_pixel + x2
        y1 = int(self.y_negative * self.pixel_per_meter) + self.pixel_shift_y
        y2 = self.y_size_pixel + y1
        # print(x1,x2,y1,y2)

        mask = canvas[x1:x2, y1:y2]
        return mask

    def make_empty_canvas(self):
        x, y = self.calculate_mask_size()

        return np.zeros((x, y), dtype=np.uint8)

    def plot_on_pixel(self, vehicle, actor, canvas):

        x1 = int((actor.x1 ) * self.pixel_per_meter) + self.pixel_shift_x + 1
        y1 = int((actor.y1) * self.pixel_per_meter) + self.pixel_shift_y + 1
        x2 = int((actor.x2) * self.pixel_per_meter) + self.pixel_shift_x + 1
        y2 = int((actor.y2) * self.pixel_per_meter) + self.pixel_shift_y + 1

        # corners = [(x1,y1),(x1,y2),(x2,y1),(x2,y2)]
        # pts = np.array(corners, 'int32')
        # print(np.fmin(x1, x2),np.fmax(x1, x2),np.fmin(y1, y2),np.fmax(y1, y2))
        # cv.fillConvexPoly(canvas, pts, 1)
        canvas[np.fmin(x1, x2):np.fmax(x1, x2), np.fmin(y1, y2):np.fmax(y1, y2)] = actor.z
        return canvas

    def calculate_mask_size(self):
        """Convert map boundaries to pixel resolution."""

        width_in_pixels = int(self.world.city_width * self.pixel_per_meter)
        height_in_pixels = int(self.world.road_len * self.pixel_per_meter)
        return height_in_pixels, width_in_pixels

    def birdeyeData(self):

        birdview = self.produce_mask()
        #birdview[:,:,0:int(self.y_size_pixel/3)]=0

        birdview = np.rot90(birdview, 2, (1, 2))

        rgb = self.as_rgb(birdview)
        rgb = self.update_ego(rgb, 84, 84)
        copyrgb = rgb.copy()
        n, m = np.shape(birdview[0, :, :])
        x, y = np.ogrid[0:n, 0:m]
        agent_pixel_pose_Y = int(self.y_size_pixel / 2)
        agent_pixel_pose_X = int(self.x_size_pixel)
        agent_pixel_pose = [agent_pixel_pose_X, agent_pixel_pose_Y]
        # print (agent_pixel_pose)

        fov_mask = (((agent_pixel_pose_Y - y) < ((agent_pixel_pose_X - x) * math.tan(self.camera_offset + self.FRONT_FOV / 2))) & (
                (agent_pixel_pose_Y - y) > (-(agent_pixel_pose_X - x) * math.tan(-self.camera_offset + self.FRONT_FOV / 2))))

        self.mask_shadow, uncertainty,d = shadow_mask.view_field(2.7 * birdview[1, :, :], 1, agent_pixel_pose,max_distance=self.max_detaction_distance)
        # (array[self.mask_shadow & mask]) = 255

        a = np.zeros(np.shape(d),dtype=bool)

        a = np.where(d <= self.max_detaction_distance, True, a)
        a = np.where(d > self.max_detaction_distance, False, a)
        fov_mask = fov_mask*a
        a = np.zeros(np.shape(d),dtype=bool)

        a = np.where(d <= self.max_detaction_distance, True, a)
        a = np.where(d > self.max_detaction_distance, False, a)
        fov_mask = fov_mask*a
        # copyrgb = rgb.copy()
        shadow = np.zeros((n, m))
        visibility_matrix = fov_mask & self.mask_shadow
        shadow[visibility_matrix] = 1





        uncertainty = uncertainty * shadow
        uncertainty = uncertainty.astype('int32')
        birdview[0, :, :] = birdview[0, :, :] * uncertainty
        copyrgb[:, :, 1] = copyrgb[:, :, 1] + 90 * uncertainty
        # copyrgb[:,:,0] = copyrgb[:,:,0]
        embedded_birdseye = np.zeros((n, m))

        for i in range(self.state_frames * self.n_frame_dropout - 1, -1, -1):
            if i == 0:
                self.pedestrian_birdview_stack[0, :, :] = birdview[0, :, :]  + 0.1*birdview[1, :, :]*fov_mask
            else:
                self.pedestrian_birdview_stack[i, :, :] = self.pedestrian_birdview_stack[i - 1, :, :]

            # print(i)
            if i % self.n_frame_dropout == 0:
                self.pedestrian_birdview[int(i // self.n_frame_dropout), :, :] = self.pedestrian_birdview_stack[i, :, :]

                embedded_birdseye = self.pedestrian_birdview[int(i // self.n_frame_dropout), :, :] \
                                    * (0.5 ** (int(i // self.n_frame_dropout))) + embedded_birdseye

        visibility_int = np.zeros((1, n, m))
        visibility_int[0, uncertainty] = 1

        # copyrgb[:,:,0] = 255 * embedded_birdseye  + copyrgb[:,:,0]
        copyrgb = copyrgb.swapaxes(0, 1)
        self.rgb_nyg_surf = pygame.surfarray.make_surface(copyrgb)
        self.rgb_nyg_surf = pygame.transform.scale(self.rgb_nyg_surf, (self.x_size_pixel * 4, self.y_size_pixel * 4))

        # pygame.transform.flip(self.tensor,True,True)
        # self.tensor = pygame.transform.rotozoom(self.tensor,90,4)

        return self.pedestrian_birdview, visibility_int

    def push_to_tensor(self, intensor, x):
        tensor = intensor.clone()
        tensor[:-1, :, :] = intensor[1:, :, :]
        tensor[:-1, :, :] = x[:, :]
        return tensor

    @staticmethod
    def as_rgb(birdview):
        RGB_BY_MASK = [(255, 0, 0), (100, 0, 100)]
        GRAY = [209, 209, 209]
        h, w, d = birdview.shape
        # print(h, w, d)
        rgb_canvas = np.zeros(shape=(w, d, 3), dtype=np.uint8)
        mask = np.zeros(shape=(w, d, 1), dtype=np.uint8)

        rgb_canvas[:, 20:60, 0:2] = 9
        rgb_canvas[:, :, 0] = 200 * birdview[0, :, :]
        rgb_canvas[:, :, 2] = 200 * birdview[1, :, :]


        # print(np.shape(mask))
        # If mask above contains 0, don't overwrite content of canvas (0 indicates transparency)
        return rgb_canvas
