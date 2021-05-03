import random
import math
import torch as T
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
#import sync
import pygame
import car_env_one_ped as carenv

import numpy as np
#import dq_agent
import carla_bridge
import ddqn_agent

import data_process

class trainer(object):
    def __init__(self, gamma, epsilon, lr, n_actions, uncertainty_enabled, mem_size, batch_size,
                 input_dimention, n_stacked_frames, n_frame_dropout,eps_min, eps_dec,replace,
                 control_dropout_enabled,load_network,train_network,n_episodes,remove_side,chkpt_name,ttc,logging_dir):
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
        self.remove_side= remove_side
        self.txt_logging_name = chkpt_name
        train_dimention = input_dimention

        if self.remove_side:
            train_dimention[2] = int(3*train_dimention[2]/4)
        self.create_env(n_stacked_frames=n_stacked_frames,
                        n_frame_dropout=n_frame_dropout,
                        input_dimention=input_dimention,
                        fov_radian=100*math.pi/180
                        )
        self.agent = ddqn_agent.DDQNAgent(gamma=gamma,
                               epsilon=epsilon,
                               lr=lr,
                               n_actions=n_actions,
                               input_dims=train_dimention,
                               mem_size=mem_size,
                               batch_size=batch_size,
                               chkpt_name=chkpt_name,
                               eps_min=eps_min,
                               eps_dec=eps_dec,
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
        self.max_distances = []
        self.ped_blockeds = []
        self.n_steps = 0
        self.ped_passing = []
        ti = datetime.now().strftime("%Y_%m_%d_%H_%M")
        self.tb_writer = SummaryWriter(f"logs/tensor_board/{self.logging_dir}-{chkpt_name}-{str(ti)}")
        #self.file_path = f"logs/{MODEL_NAME}-{str(ti)}.csv"

        self.input_dimention = input_dimention

        pygame.init()
        pygame.font.init()
        self.world = None

        self.display = pygame.display.set_mode(
            (400, 400),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.action_space = [0, 1, 2,3,4]
        self.saver = data_process.SaveData(self.ttc_list,self.txt_logging_name,self.logging_dir)


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
            self.max_detection_distance = random.randint(40,100)
        else:
            self.max_detection_distance = 60
        self.max_detection_distance = 60
        #self.max_detection_distance = random.randint(30, 90)
        #self.max_detection_distance = random.randint(50, 100)

        self.bird_producer = carenv.BirdEye(self.x_size_pixel, self.y_size_pixel, 4, self.world, n_frame_dropout, n_stacked_frames,fov_radian,self.max_detection_distance)

        #self.action_space = [ut.actions.ACCELERATE, ut.actions.REMAIN, ut.actions.DECELERATE]

        first_episode =True

    def init_episode(self,fov_radian):
        self.create_env(n_stacked_frames=self.n_stacked_frames,
                        n_frame_dropout=self.n_frame_dropout,
                        input_dimention=self.input_dimention,
                        fov_radian=fov_radian
                        )
        self.state = self.observe()

    def observe(self):
        observation=self.bird_producer.birdeyeData()
        x = self.x_size_pixel
        y = int(3 * self.y_size_pixel/ 4)
        state = self.state_producer_embedded(observation,x,y)
        return state

    def state_producer_embedded(self,state,x,y):
        framed_state, visibility_state = state
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

    def train(self):
        best_avg_score = -1.01
        avg_score = -1
        freeze = False
        for i_episode in range(self.num_episodes):
            done = False
            score = 0
            self.fov_deg = 90#random.randint(40,120)
            #print(self.fov_deg)
            self.init_episode(fov_radian=math.pi*self.fov_deg/180)
            episode_step_count = int(0)

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
                self.controller.speed_setpoint_actions(self.action_space[action],self.world.clock)
                self.world.tick()
                self.next_state = self.observe()
                reward, done, collision = self.world.reward(self.action_space[action])
                score += reward
                reward = T.tensor([reward], device=self.agent.q_eval.device, dtype=T.float32)
                #time.sleep(.010)
                self.world.render(self.display,self.bird_producer.rgb_nyg_surf)
                #pygame.display.flip()


                if self.training_mode:
                    self.agent.store_transition(state=self.state,
                                                action=action,
                                                reward=reward,
                                                state_=self.next_state,
                                                done=int(done)
                                                )
                    self.agent.learn(freeze= freeze)

                self.state = self.next_state
                self.n_steps += 1
            self.scores.append(score)
            self.ped_passing.append(int(self.world.passing))
            self.steps_array.append(self.n_steps)

            self.collisions.append(collision)
            self.building_preset.append(self.world.building_present)
            self.ped_blockeds.append(self.world.blocked)
            self.episode_avg_speed.append(self.world.avg_speed)
            self.ttcs.append(self.world.chosen_ttc)
            self.fovs_deg.append(self.fov_deg)
            self.max_distances.append(self.max_detection_distance)

            ttc_list = self.world.ttc

            if i_episode > 200:
                freeze = False


            #self.world.destroy()
            #self.pedestrians.kill_walkers()
            #self.vehicle = self.world.restart()
            if i_episode>100:
                avg_score = np.mean(self.scores[-100:])
            avg_collisions = np.mean(self.collisions[-100:])


            if avg_score > best_avg_score :
                if self.training_mode:
                    self.agent.save_models()
                best_avg_score = avg_score
            print('episode: ', i_episode, 'score: ', score,
                  ' average score %.1f' % avg_score, 'best score %.2f' % best_avg_score,
                  'epsilon %.2f' % self.agent.epsilon, 'steps', self.n_steps,'algorithm:',self.txt_logging_name)


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

        #self.agent.save_models()
        print("end of episode")
        self.saver.save_data(passing=self.ped_passing,
                             n_steps=self.steps_array,
                             collisions=self.collisions,
                             building_present=self.building_preset,
                             episode_avg_speed=self.episode_avg_speed,
                             ped_ttc=self.ttcs,
                             fov_deg=self.fovs_deg,
                             max_distance=self.max_distances,
                             ped_blocked=self.ped_blockeds)


