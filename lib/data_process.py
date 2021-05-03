from __future__ import division

import csv
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import math
import os


class SaveData:
    def __init__(self,ttc_list,txt_logging_name, logging_dir):

        self.ttc_list = ttc_list
        self.txt_logging_name = txt_logging_name
        self.logging_dir = logging_dir


    def save_data(self,passing,n_steps,collisions,building_present,episode_avg_speed,ped_ttc,fov_deg,max_distance,ped_blocked):


        self.file = open('logs/' + self.logging_dir + '/' + self.txt_logging_name + '.txt', '+a')

        self.passing_peds = passing
        self.steps = n_steps
        self.fov_deg = fov_deg

        self.collisions = collisions
        self.building_preset = building_present
        self.episode_avg_speed = episode_avg_speed
        self.ttcs = ped_ttc
        self.max_distance = max_distance
        self.ped_blocked = ped_blocked
        list_of_lists = [self.steps,
                         self.collisions,
                         self.passing_peds,
                         self.building_preset,
                         self.episode_avg_speed,
                         self.ttcs,
                         self.fov_deg,
                         self.max_distance,
                         self.ped_blocked]

        for i in range(len(self.steps)):
            j = 0
            for item in list_of_lists:
                if j>0:
                    self.file.write(",")
                j = 1
                self.file.write(str(item[i]))


            self.file.write("\n")
        self.file.close()
        self.plot()

    def read_text(self):
        self.file = open('dataset', 'r')
        self.passing_peds = []
        self.steps = []
        self.lines = [[],[],[],[],[],[],[]]
        #self.lines = []

        self.collisions = []
        self.building_preset = []
        self.episode_avg_speed =[]
        self.ttcs = []
        self.fov_deg = []
        s = 0
        #self.splited
        for line in self.file:
            print(line)
            for num in line.strip().split(','):
                try:

                    self.lines[s].append(float(num))
                except:
                    pass
            s += 1
        sampling = 2000






        self.steps = self.lines[0][-sampling:]
        self.passing_peds = self.lines[2][-sampling:]
        self.collisions = self.lines[1][-sampling:]
        self.building_preset = self.lines[3][-sampling:]
        self.episode_avg_speed = self.lines[4][-sampling:]
        self.ttcs = self.lines[5][-sampling:]
        self.fov_deg = self.deg[6][-sampling:]



    def plot(self):
        self.lines = [self.steps, self.collisions, self.passing_peds, self.building_preset, self.episode_avg_speed,
                         self.ttcs, self.fov_deg]
        sampling = 2000


        self.steps = self.lines[0][-sampling:]
        self.passing_peds = self.lines[2][-sampling:]
        self.collisions = self.lines[1][-sampling:]
        self.building_preset = self.lines[3][-sampling:]
        self.episode_avg_speed = self.lines[4][-sampling:]
        self.ttcs = self.lines[5][-sampling:]

        np.set_printoptions(precision=5)
        y = np.zeros((6, len(self.ttc_list),), dtype=np.float32)
        z = np.zeros((6, len(self.ttc_list),), dtype=np.float32)

        _mask_with_build = np.array(np.array(self.building_preset, dtype=int) == int(1), dtype=int)
        _mask_no_build = np.array(np.array(self.building_preset,dtype=int) == int(0),dtype=int)

        _mask_stationry_ped = np.array(np.array(self.passing_peds, dtype=int) == int(0), dtype=int)
        _mask_passing_ped = np.array(np.array(self.passing_peds, dtype=int) == int(1), dtype=int)

        _mask_urban_passing = np.multiply(_mask_with_build, _mask_passing_ped)
        _mask_clear_passing = np.multiply(_mask_no_build, _mask_passing_ped)

        _mask_urban_stationary = np.multiply(_mask_with_build, _mask_stationry_ped)
        _mask_clear_stationary = np.multiply(_mask_no_build, _mask_stationry_ped)

        _mask_collisions = np.array(np.array(self.collisions,dtype=int) == int(1), dtype=int)
        _mask_no_col = np.array(np.array(self.collisions,dtype=int) == int(0), dtype=int)

        _mask_urban_collided = np.multiply(_mask_urban_passing, _mask_collisions)
        _mask_clear_collided = np.multiply(_mask_clear_passing, _mask_collisions)

        avg_speeds = np.multiply(np.array(self.episode_avg_speed, dtype=float), _mask_stationry_ped)
        avg_speed_building_filtered = np.multiply(avg_speeds, _mask_with_build)
        avg_speed_no_building_filtered = np.multiply(avg_speeds, _mask_no_build)
        clear_avg_speed = 0
        urban_avg_speed = 0
        total_avg_speed = 0



        if _mask_stationry_ped.sum()>0:
            total_avg_speed = (avg_speeds).sum() / (_mask_stationry_ped.sum())
        if _mask_with_build.sum()>0:
            urban_avg_speed = (avg_speed_building_filtered).sum() / (_mask_urban_stationary.sum())
        if _mask_no_build.sum()>0:
            clear_avg_speed = (avg_speed_no_building_filtered).sum() / (_mask_clear_stationary.sum())





        i = int(0)
        #print('ttc length:',len(self.ttc_list))
        for i in range(len(self.ttc_list)):

            ttc = float(self.ttc_list[i])
            ttc_np = np.array(self.ttcs,dtype=float)

            _mask_ttc = np.array(ttc_np == ttc, dtype=int)
            n_ttc = float(np.sum(_mask_ttc))

            _urban_ttc = np.multiply(_mask_urban_passing, _mask_ttc)
            _clear_ttc = np.multiply(_mask_clear_passing, _mask_ttc)
            _urban_collided_ttc = np.multiply( _mask_urban_collided,  _mask_ttc)
            _clear_collided_ttc = np.multiply( _mask_clear_collided,  _mask_ttc)

            _passing_ttc = np.multiply( _mask_passing_ped,  _mask_ttc)



            if _passing_ttc.sum() > 0:
                y[0, i] =  100*np.float32((_urban_collided_ttc.sum()+_clear_collided_ttc.sum()) / _passing_ttc.sum())
            if _urban_ttc.sum() > 0:
                y1 = 100*np.float32(_urban_collided_ttc.sum() / _urban_ttc.sum())
                y[1,i] = y1
            if _clear_ttc.sum() > 0:
                y2 = 100*np.float32(_clear_collided_ttc.sum() / _clear_ttc.sum())
                y[2,i] = y2







        self.stacked_bar(y)

        #plt.savefig('logs/' + self.logging_dir + '/figures/' + self.txt_logging_name + '.png')

        self.file2 = open('logs/' + self.logging_dir + '/speeds/' + self.txt_logging_name + '.txt', '+a')
        self.file2.write("total avg speed: "+str(total_avg_speed)+'\n')
        self.file2.write("urban avg speed: "+str(urban_avg_speed)+'\n')
        self.file2.write("clear avg speed: "+str(clear_avg_speed)+'\n')
        self.file2.close()
    def stacked_bar(self,y):
        visible = (y[2,:])
        blocked = (y[1,:])
        ymax = np.amax(np.array(visible + blocked))
        fig, axs = plt.subplots(1)

        #fig.ti('collisions in: '+self.txt_logging_name)
        colors = {'Clear area': 'tab:blue','Urban area': 'tab:orange',  'Overall': 'tab:green'}
        labels = list(colors.keys())
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
        axs.legend(handles, labels)
        axs.set_ylim([0, 1.1*ymax])
        axs.set_ylabel('Collisions %')
        axs.set_xlabel('ttc pedestrian moves in front of the vehicle.')

        width = 0.06
        axs.bar(np.array(self.ttc_list), (y[0,:]), width=width, color='tab:green')
        axs.bar(np.array(self.ttc_list)-0.1, blocked, width=width, color='tab:orange')
        axs.bar(np.array(self.ttc_list)-0.2, visible , width=width, color='tab:blue')
        fig.tight_layout()
        #os.remove('logs/' + self.logging_dir + '/figures/' + self.txt_logging_name + '.png')
        plt.savefig('logs/' + self.logging_dir + '/figures/' + self.txt_logging_name + '.png')


    '''def cumm_plot(self,y):
        weeks = np.arange(1, 13, 1)
        visible = (y[2,:])
        blocked = (y[1,:])
        features = np.row_stack((features_dne, features_vfy, features_tst,
                                 features_dev, features_req))

        fig, ax = plt.subplots()
        ax.stackplot(weeks, features)

        # Add relevant y and x labels and text to the plot
        plt.title('Cumulative Flow Diagram')
        ax.set_ylabel('Features')
        ax.set_xlabel('Weeks')
        ax.set_xlim(1, 12)
        ax.set_ylim(0, 6)'''

