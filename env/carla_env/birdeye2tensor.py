from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys
import shadow_mask

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
import math


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

class BirdEye(object):
    def __init__(self,parent_actor,client,memory_frames,frame_stack,width, height):
        self.frame_stack = frame_stack
        self.bird_eye_window_width = width
        self.bird_eye_window_height = height
        #self.buffer = torch.zeros(3, self.bird_eye_window_width, self.bird_eye_window_height).to(device)
        self.FRONT_FOV = 2 * (math.pi) / 3
        self._parent = parent_actor
        self.surface = None
        self.bird_array = None
        self._carla_client = client
        self.state_frames = memory_frames
        self.pedestrian_birdview = np.full((self.state_frames, self.bird_eye_window_width,self.bird_eye_window_height), 0, dtype=np.float32)
        self.pedestrian_birdview_stack = np.full((self.frame_stack * self.state_frames, self.bird_eye_window_width,self.bird_eye_window_height), 0, dtype=np.float32)
        self.pedestrian_birdview_embeded = np.full((self.bird_eye_window_width, self.bird_eye_window_height), 0, dtype=np.float32)

        self.birdview_producer = BirdViewProducer(
            self._carla_client,  # carla.Client
            target_size=PixelDimensions(width=self.bird_eye_window_width, height=self.bird_eye_window_height),
            pixels_per_meter=2,
            crop_type=BirdViewCropType.FRONT_AND_REAR_AREA, render_lanes_on_junctions=True,
        )
        #self.birdeyeData(parent_actor)
        '''    PEDESTRIANS = 8
        RED_LIGHTS = 7
        YELLOW_LIGHTS = 6
        GREEN_LIGHTS = 5
        AGENT = 4
        VEHICLES = 3
        CENTERLINES = 2
        LANES = 1
        ROAD = 0'''


    def birdeyeData(self, player, lidar_object):



        self._parent = player
        self.bird_array = self.birdview_producer.produce(agent_vehicle=self._parent)
        rgb = BirdViewProducer.as_rgb(self.bird_array)

        copyrgb = rgb.copy()
        n, m = np.shape(self.bird_array[:, :, 1])
        x, y = np.ogrid[0:n, 0:m]
        agent_pixel_pose_Y = int(self.bird_eye_window_width / 2)
        agent_pixel_pose_X = int(self.bird_eye_window_height / 2)
        agent_pixel_pose = [agent_pixel_pose_X, agent_pixel_pose_Y]
        # print (agent_pixel_pose)

        fov_mask = (((agent_pixel_pose_Y - y) < ((agent_pixel_pose_X - x) * math.tan(self.FRONT_FOV / 2))) & (
                    (agent_pixel_pose_Y - y) > (-(agent_pixel_pose_X - x) * math.tan(self.FRONT_FOV / 2))))

        self.mask_shadow, uncertainty,d = shadow_mask.view_field(self.bird_array[:, :, 9] + 2.7*self.bird_array[:, :, 3], 1.8, agent_pixel_pose,90)
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
        n, m = np.shape(self.bird_array[:, :, 1])
        embedded_birdseye = np.zeros((n, m))

        self.bird_array[:, :, 8]=self.bird_array[:, :, 8]*uncertainty


        for i in range(self.state_frames * self.frame_stack-1,-1,-1):
            if i == 0:
                self.pedestrian_birdview_stack[0, :, :] = np.multiply(shadow, self.bird_array[:, :, 8])
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






        self.pedestrian_birdview_embeded = np.add(255*self.bird_array[:, :, 8], 0.5 * self.pedestrian_birdview_embeded)
        obsv= (0.01*copyrgb)
        obsv[:, :, 0] = obsv[:, :, 0] #+ 50 * uncertainty
        obsv[:,:,2]= obsv[:,:,2]+200*self.pedestrian_birdview_stack[0, :, :]
        obsv[:,:,1] = obsv[:,:,1]+100*self.bird_array[:, :, 3]
        #obsv[:,:,1]= obsv[:,:,1]+200*self.pedestrian_birdview_stack[0, :, :]+100*(birdview[:, :, 3] * cir)
        obsv = obsv.swapaxes(0, 1)
        '''if lidar_data is not None:
            obsv= obsv + lidar_data
        birdview[:, :, 1] * fov_mask'''

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
        #pygame.display.set_caption('Show Text')

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