from datetime import datetime
import os

import dqn_trainer
import side_trainer
# import side_trainer as car_trainer
import qr_trainer
import carla_dqn
import numpy as np
import data_process

import numpy as np




class Tester:
    def __init__(self, algo_n, n_episodes, uncertainty_enabled,logging_dir,train_network):

        self.algo_n = algo_n
        self.logging_dir = logging_dir
        self.n_episodes = n_episodes

        self.uncertainty_enabled = uncertainty_enabled
        if self.uncertainty_enabled:
            self.uncer_str = 'risk_aware'
        else:
            self.uncer_str = 'normal'

        if algo_n == 0:
            self.algo_name = 'ddqn'
        elif algo_n == 1:
            self.algo_name = 'qrdqn'
        elif algo_n == 2:
            self.algo_name = 'side_qrdqn'

        if train_network:
            self.epsilon = 1
            self.eps_min = 0.01
        else:
            self.epsilon = 0
            self.eps_min = 0

        self.set_meta_params()

        self.steps = []
        self.passing_peds = []
        self.collisions = []
        self.building_preset = []
        self.episode_durations = []
        self.ttcs = []
        self.fov_deg = []
        self.max_distance = []
        self.ped_blocked = []
        self.load_network = True
        self.train_network = train_network


    def set_meta_params(self):

        self.n_actions = 5
        self.n_stacked_frames = 4
        self.n_frame_dropout = 4
        if self.uncertainty_enabled:
            input_dims = [self.n_stacked_frames + 1, 84, 84]
        else:
            input_dims = [self.n_stacked_frames, 84, 84]

        self.gamma = 0.99
        self.lr = 0.0001

        self.batch_size = 64
        self.replace = 5_000
        self.mem_size = 50_000
        self.eps_dec = 1e-5
        self.n_quants = 32

        self.control_dropout_enabled = False



        if self.uncertainty_enabled:
            self.input_dims = [self.n_stacked_frames + 1, 84, 84]
        else:
            self.input_dims = [self.n_stacked_frames, 84, 84]


    def run_trainer(self,ttc):


        if self.algo_n == 0:

            self.trainer = dqn_trainer.trainer(gamma=self.gamma,
                                          epsilon=self.epsilon,
                                          lr=self.lr,
                                          n_actions=self.n_actions,
                                          uncertainty_enabled=self.uncertainty_enabled,
                                          mem_size=self.mem_size,
                                          batch_size=self.batch_size,
                                          input_dimention=self.input_dims,
                                          n_stacked_frames=self.n_stacked_frames,
                                          n_frame_dropout=self.n_frame_dropout,
                                          eps_min=self.eps_min,
                                          eps_dec=self.eps_dec,
                                          replace=self.replace,
                                          control_dropout_enabled=self.control_dropout_enabled,
                                          load_network=self.load_network,
                                          train_network=self.train_network,
                                          n_episodes=self.n_episodes,
                                          remove_side=False,
                                          chkpt_name=f"{self.algo_name}_{self.uncer_str}",
                                          ttc=ttc,
                                          logging_dir=self.logging_dir
                                          )
        elif self.algo_n == 1:

            self.trainer = qr_trainer.trainer(gamma=self.gamma,
                                          epsilon=self.epsilon,
                                          lr=self.lr,
                                          n_actions=self.n_actions,
                                         n_quants=self.n_quants,
                                         uncertainty_enabled=self.uncertainty_enabled,
                                         mem_size=self.mem_size,
                                         batch_size=self.batch_size,
                                         input_dimention=self.input_dims,
                                         n_stacked_frames=self.n_stacked_frames,
                                         n_frame_dropout=self.n_frame_dropout,
                                         eps_min=self.eps_min,
                                         eps_dec=self.eps_dec,
                                         replace=self.replace,
                                         control_dropout_enabled=self.control_dropout_enabled,
                                         load_network=self.load_network,
                                         train_network=self.train_network,
                                         n_episodes=self.n_episodes,
                                         remove_side=False,
                                         chkpt_name=f"{self.algo_name}_{self.uncer_str}",
                                         ttc=ttc,
                                         logging_dir=self.logging_dir
                                         )

        elif self.algo_n == 2:

            self.trainer = side_trainer.trainer(gamma=self.gamma,
                                          epsilon=self.epsilon,
                                          lr=self.lr,
                                          n_actions=self.n_actions,
                                         n_quants=self.n_quants,
                                         uncertainty_enabled=self.uncertainty_enabled,
                                         mem_size=self.mem_size,
                                         batch_size=self.batch_size,
                                         input_dimention=self.input_dims,
                                         n_stacked_frames=self.n_stacked_frames,
                                         n_frame_dropout=self.n_frame_dropout,
                                         eps_min=self.eps_min,
                                         eps_dec=self.eps_dec,
                                         replace=self.replace,
                                         control_dropout_enabled=self.control_dropout_enabled,
                                         load_network=self.load_network,
                                         train_network=self.train_network,
                                         n_episodes=self.n_episodes,
                                         remove_side=False,
                                         chkpt_name=f"{self.algo_name}_{self.uncer_str}",
                                         ttc=ttc,
                                         logging_dir=self.logging_dir
                                         )
        elif self.algo_n == 3:

            self.trainer = carla_dqn.trainer(gamma=self.gamma,
                                                epsilon=self.epsilon,
                                                lr=self.lr,
                                                n_actions=self.n_actions,
                                                n_quants=self.n_quants,
                                                uncertainty_enabled=self.uncertainty_enabled,
                                                mem_size=self.mem_size,
                                                batch_size=self.batch_size,
                                                input_dimention=self.input_dims,
                                                n_stacked_frames=self.n_stacked_frames,
                                                n_frame_dropout=self.n_frame_dropout,
                                                eps_min=self.eps_min,
                                                eps_dec=self.eps_dec,
                                                replace=self.replace,
                                                control_dropout_enabled=self.control_dropout_enabled,
                                                load_network=self.load_network,
                                                train_network=self.train_network,
                                                n_episodes=self.n_episodes,
                                                remove_side=False,
                                                chkpt_name=f"{self.algo_name}_{self.uncer_str}",
                                                ttc=ttc,
                                                logging_dir=self.logging_dir
                                                )

        self.trainer.train()

    def train_all(self,ttc_list):

        training_ttc = ttc_list
        self.run_trainer(training_ttc)
        #self.plot()



    def ttc_loop(self,ttc_list):

        for ttc in ttc_list:

            training_ttc = [ttc]
            self.run_trainer(training_ttc)
            self.append_for_plotting()
        self.trainer.saver.ttc_list = ttc_list
        #self.plot()


    def append_for_plotting(self):

        self.steps = self.steps + self.trainer.saver.steps
        self.passing_peds = self.passing_peds + self.trainer.saver.passing_peds
        self.collisions = self.collisions + self.trainer.saver.collisions
        self.building_preset = self.building_preset + self.trainer.saver.building_preset
        self.episode_durations = self.episode_durations + self.trainer.saver.episode_avg_speed
        self.ttcs = self.ttcs + self.trainer.saver.ttcs
        self.fov_deg = self.fov_deg + self.trainer.saver.fov_deg
        self.max_distance = self.max_distance + self.trainer.saver.max_distance
        self.ped_blocked = self.ped_blocked + self.trainer.saver.ped_blocked
        #print()

    def plot(self):
        self.trainer.saver.steps = self.steps
        self.trainer.saver.passing_peds = self.passing_peds
        self.trainer.saver.collisions = self.collisions
        self.trainer.saver.building_preset = self.building_preset
        self.trainer.saver.episode_avg_speed = self.episode_durations
        self.trainer.saver.ttcs =  self.ttcs
        self.trainer.saver.fov_deg = self.fov_deg
        self.trainer.saver.max_distance =  self.max_distance
        self.trainer.saver.ped_blocked = self.ped_blocked

        self.trainer.saver.plot()


