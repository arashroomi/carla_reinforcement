import random
import math
import torch as T
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
#import sync
import pygame
import carla_bridge as carenv
#import car_passing as carenv
import qrdqn

import numpy as np
#import dq_agent
import carla_bridge

import glob
import os
import sys
try:
    sys.path.append(glob.glob('./carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

# ==============================================================================
# -- Internal imports ----------------------------------------------------------
# ==============================================================================
import controls
import pedestrian_control
# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================

class trainer(object):
    def __init__(self, gamma, epsilon, lr, n_actions, uncertainty_enabled, mem_size, batch_size,
                 input_dimention, n_stacked_frames, n_frame_dropout,eps_min, eps_dec,replace,
                 control_dropout_enabled,load_network,train_network,n_episodes,remove_side,chkpt_name,
                 n_quants,ttc,logging_dir
                 ):
        self.ttc_list = ttc
        self.logging_dir = logging_dir
        self.uncertainty_enabled = uncertainty_enabled
        self.num_episodes = n_episodes
        self.best_score = -np.inf
        self.load_checkpoint = load_network
        self.training_mode = train_network
        self.n_frame_dropout = n_frame_dropout
        self.n_stacked_frames = n_stacked_frames
        self.input_dimention = input_dimention
        self.control_dropout_enabled = control_dropout_enabled
        self.remove_side = remove_side
        train_dimention = input_dimention
        self.n_quants = n_quants
        self.chkpt_name = chkpt_name
        self.txt_logging_name = chkpt_name


        '''self.create_env(n_stacked_frames=n_stacked_frames,
                        n_frame_dropout=n_frame_dropout,
                        input_dimention=input_dimention,
                        fov_radian=100*math.pi/180
                        )'''
        self.agent = qrdqn.QRAgent(gamma=gamma,
                               epsilon=epsilon,
                               lr=lr,
                               n_actions=n_actions,
                               input_dims=train_dimention,
                               mem_size=mem_size,
                               batch_size=batch_size,
                               chkpt_name=chkpt_name,
                               eps_min=eps_min,
                               eps_dec=eps_dec,
                               n_quant=n_quants,
                               replace=replace,
                               logging_dir=logging_dir

                               )
        if load_network:
            self.agent.load_models()
        #self.action_space = [0, 1, 2,3,4]#[ut.actions.ACCELERATE, ut.actions.REMAIN, ut.actions.DECELERATE]
        self.scores, self.eps_history, self.steps_array =[], [], []
        self.collisions = []
        self.building_preset = []
        self.episode_avg_speed = []
        self.ttcs = []
        self.fovs_deg = []
        self.n_steps = 0
        self.ped_passing = []
        self.max_distances = []
        self.ped_blockeds = []
        ti = datetime.now().strftime("%Y_%m_%d_%H_%M")
        self.tb_writer = SummaryWriter(f"logs/tensor_board/{self.logging_dir}-{chkpt_name}-{str(ti)}")
        #self.file_path = f"logs/{MODEL_NAME}-{str(ti)}.csv"

        self.input_dimention = input_dimention
        pygame.init()
        pygame.font.init()
        self.connect_to_carla()
        self.action_space = [0, 1, 2]
        #self.saver = data_process.SaveData(self.ttc_list,self.txt_logging_name,self.logging_dir)


    def create_env(self, n_stacked_frames, n_frame_dropout,input_dimention,fov_radian):
        self.x_size_pixel= input_dimention[1]
        self.y_size_pixel = input_dimention[2]
        self.world = carenv.World()
        self.world.ttc = self.ttc_list
        self.world.start_world()
        self.controller = carla_bridge.VehicleControl(self.world)
        self.world.tick()
        self.vehicle = self.world.vehicle
        fov_radian = 90*math.pi/180
        if self.uncertainty_enabled:
            self.max_detection_distance = random.randint(30,90)
        else:
            self.max_detection_distance = 60
        #self.max_detection_distance = random.randint(30, 90)
        self.max_detection_distance = 75
        #self.max_detection_distance = random.randint(50, 100)
        #self.bird_producer = carenv.BirdEye(self.x_size_pixel, self.y_size_pixel, 2, self.world, n_frame_dropout, n_stacked_frames,fov_radian,camera_offset_rad)
        self.bird_producer = carenv.BirdEye(self.x_size_pixel, self.y_size_pixel, 4, self.world, n_frame_dropout,
                                            n_stacked_frames, fov_radian,self.max_detection_distance)

        #self.action_space = [ut.actions.ACCELERATE, ut.actions.REMAIN, ut.actions.DECELERATE]

        first_episode =True

    def init_episode(self,fov_radian):
        self.create_env(n_stacked_frames=self.n_stacked_frames,
                        n_frame_dropout=self.n_frame_dropout,
                        input_dimention=self.input_dimention,
                        fov_radian=fov_radian,
                        )
        self.state = self.observe()


    def state_producer(self,observation):
        x = self.x_size_pixel
        y = int(3 * self.y_size_pixel/ 4)
        framed_state, visibility_state = observation
        if self.remove_side:
            framed_state= framed_state[:,0:x,0:y]
            visibility_state = visibility_state[:,0:x,0:y]

        if self.uncertainty_enabled:
            framed_state = framed_state.astype(np.float32)
            merged_array =np.concatenate((framed_state, visibility_state), axis=0)
             #np.array([framed_state, visibility_state])
            bird_eye_array = merged_array.astype(np.float32)
        else:
            bird_eye_array = framed_state.astype(np.float32)
        return bird_eye_array
    #T.from_numpy(bird_eye_array).unsqueeze(0).to(self.agent.q_eval.device)

    def connect_to_carla(self):

        self.x_size_pixel= self.input_dimention[1]
        self.y_size_pixel = self.input_dimention[2]
        self.world = None

        args = carla_bridge.initial_factors()

        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(2.0)

        self.display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        self.hud = carla_bridge.HUD(args.width, args.height)
        self.world = carla_bridge.World(self.client, self.hud, self.n_stacked_frames, self.n_frame_dropout, self.x_size_pixel,
                                        self.y_size_pixel)
        self.vehicle = self.world.player
        self.controller = controls.VehicleControl(self.world)


        self.clock = pygame.time.Clock()
        self.clock.tick()
        observation = self.world.tick(self.clock)
        self.state = self.state_producer(observation)
        self.world.render(self.display)
        pygame.display.flip()


    def train(self):
        best_avg_score = 0
        avg_score = 0
        avg_collisions = 0

        self.world.tick(self.clock)
        self.vehicle = self.world.player

        T.cuda.empty_cache()

        for i_episode in range(self.num_episodes):





            done = False
            score = 0
            self.running_reward = 0
            self.fov_deg = 90#random.randint(40,120)
            #print(self.fov_deg)
            #self.init_episode(fov_radian=math.pi*self.fov_deg/180)
            episode_step_count = int(0)
            self.pedestrians = pedestrian_control.CreateWalkers(client=self.client, number_of_walkers=random.randint(2, 3))
            walkers = self.pedestrians.SpawnWalkers(agent_vehicle=self.vehicle)
            self.world.load_walkers(walkers)


            while not done:

                # Select and perform an action
                if self.control_dropout_enabled:
                    if episode_step_count%self.n_frame_dropout == 0:
                        action = self.agent.select_action(self.state)

                    else:
                        pass
                else:
                    action = self.agent.select_action(self.state)
                episode_step_count += 1
                #print(action)
                self.controller.actions(self.action_space[action],self.clock)
                self.pedestrians.walker_collision_creator(self.vehicle)
                self.clock.tick()
                observation = self.world.tick(self.clock)
                self.next_state = self.state_producer(observation)
                self.world.render(self.display)
                pygame.display.flip()
                reward, done, collision = self.world.reward(self.action_space[action])
                score += reward
                reward = T.tensor([reward], device=self.agent.Z.device, dtype=T.float32)
                self.running_reward += reward
                #time.sleep(.010)
                #pygame.display.flip()


                if self.training_mode:
                    self.agent.store_transition(state=self.state,
                                                action=action,
                                                reward=reward,
                                                state_=self.next_state,
                                                done=int(done)
                                                )
                    self.agent.learn()

                self.state = self.next_state
                self.n_steps += 1
            #self.collisions.append(collision)
            self.scores.append(score)
            if done:
                self.world.destroy()
                print(f"episode {str(i_episode)} running reward: {str(self.running_reward.item())}")
                self.tb_writer.add_scalar('reward', self.running_reward, i_episode)


                running_reward = 0
                #self.pedestrians.kill_walkers()
                self.vehicle = self.world.restart()

            if i_episode>30:
                avg_score = np.mean(self.scores[-100:])
                avg_collisions = np.mean(self.collisions[-100:])
            if avg_score > best_avg_score :
                if self.training_mode:
                    self.agent.save_models()
                best_avg_score = avg_score

            print('episode: ', i_episode, 'score: ', score,
                  ' average score %.1f' % avg_score, 'best score %.2f' % best_avg_score,
                  'epsilon %.2f' % self.agent.epsilon, 'steps', self.n_steps, 'algorithm:',self.chkpt_name)


            self.tb_writer.add_scalars(f'avg_score/eps', {
                self.txt_logging_name+'avg_score': avg_score,
                'eps': self.agent.epsilon,
            }, i_episode)
            self.tb_writer.add_scalars(f'episode_score/eps', {
                'score': score,
                'eps': self.agent.epsilon,
            }, i_episode)
            self.tb_writer.add_scalars(f'avg_collision/eps', {
                self.txt_logging_name+'avg_collision': avg_collisions,
                'eps': self.agent.epsilon,
            }, i_episode)

        self.agent.save_models()
        print("end of episode")

