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
from sensing import CollisionSensor, LaneInvasionSensor, CameraManager, LidarSensor
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
            cloudiness=50.0,
            precipitation=0.0,
            sun_altitude_angle=30,
            fog_density=0,
            fog_falloff=0,
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
        self.lidar_sensor = None
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
        self.camera_start = True
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



        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.lidar_sensor = LidarSensor(0, 0, 0, self.player, self.hud, True, False)
        if self.camera_enabled:
            self.camera_manager = CameraManager(self.player, self.hud)


        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)
        #self.birds_eye_data = birds_eye(self.player,self.carla_client)
        self.begining = True

        initial_control = carla.VehicleControl()
        initial_control.throttle = 1
        self.player.apply_control(initial_control)
        snap = self.world.get_snapshot()
        self.simulation_start_time = snap.timestamp.elapsed_seconds

        if self.first_restart:
            self.birds_eye_data = BirdEye(self.player, self.carla_client,width=self.state_dimention_width,
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
        for walker in self.walkers:
            walker_transform = walker.get_transform()
            loc = walker_transform.location
            walker_xyz = np.array([loc.x, loc.y, loc.z])
            R = transforms3d.euler.euler2mat(0, 0, yaw).T
            ped_loc_relative = np.dot(R, walker_xyz - vehicle_xyz)
            x_to_car = ped_loc_relative[0]
            y_to_car = ped_loc_relative[1]
            #self.pedestrian_relative_distance[i] = ped_loc_relative
            if ((0<x_to_car < 7) &(-3<y_to_car<3)):
                near_collision = True
            i = i + 1
        return near_collision

    def reward(self,action):

        colhist = self.collision_sensor.get_collision_history()

        v = self.player.get_velocity()
        mps = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        kmh = int(3.6 * mps)
        reward = 0
        obstacle_on_way = self.walker_to_vehicle_distance()


        snap = self.world.get_snapshot()
        simulation_time = snap.timestamp.elapsed_seconds
        loc = self.player.get_location()
        initial_loc = self.initial_pose.location
        distance = math.sqrt((initial_loc.x - loc.x)**2 + (initial_loc.y - loc.y)**2 + (initial_loc.z - loc.z)**2)
        collision = 0

        if len(colhist) > 0:
            self.done = True
            collision = 1
            reward = -7000
            print("accident")

        if kmh >= 20 and len(colhist) <1:
            self.done = False
            reward = int(kmh)-1
        elif kmh < 20 and len(colhist) <1:
            self.done = False
            if obstacle_on_way:
                reward = 10
            else:
                reward = int(kmh) - 1


        if distance > 120:
            self.done = True
            #reward = distance * 10 - (simulation_time - self.simulation_start_time) * 10
        if self.SECONDS_PER_EPISODE < simulation_time - self.simulation_start_time :
            self.done = True
            #reward = distance

        if action != actions.REMAIN:
            reward = reward - 1
        normalized_reward = reward/8000

        self.SumReward = self.SumReward + normalized_reward
            # print("sum of rewards:", self.SumReward)
        self.hud.sum_reward = self.SumReward
        return normalized_reward, self.done, collision


    def tick(self, clock):

        self.world.tick()
        self.hud.tick(self, clock)
        framed_state, visibility_state = self.birds_eye_data.birdeyeData(self.player,self.lidar_sensor)
        snap = self.world.get_snapshot()
        #print(clock.get_time())
        if self.begining ==True:
            self.episode_start = snap.timestamp.elapsed_seconds
            self.begining = False
        return framed_state, visibility_state


    def render(self, display):
        self.hud.render(display)
        self.birds_eye_data.render(display)
        if self.camera_enabled: self.camera_manager.render(display)
        self.lidar_sensor.render(display,self.birds_eye_data.bird_array)

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
                self.camera_manager.sensor_front,
                self.camera_manager.sensor_depth,
                self.camera_manager.sensor_top,
                self.lidar_sensor.lidar]

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
        self._info_text += [


            'fps:   % 16.0f ' % self.client_fps,

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










def initial_factors():
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


