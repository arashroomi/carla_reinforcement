import glob
import os
import sys
import time
import random
import numpy as np
import cv2
import pygame
import math
import transforms3d
import logging
import matplotlib.pyplot as plt
#from multiprocessing import Process

import shadow_mask





try:
    sys.path.append(glob.glob('./carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


class CreateWalkers:


    walkers_list = []
    all_id = []
    actor_list = []


    def __init__(
        self,
        client: carla.Client,
        number_of_walkers
    ) -> None:

        self.world = client.get_world()
        self.number_of_walkers = number_of_walkers
        self.blueprintsWalkers = self.world.get_blueprint_library().filter('walker.pedestrian.*')
        self.spawn_location_calculated = False
        self.spawnable_points = []


        #print("pedestrian control is running")

    def SpawnWalkers(self,agent_vehicle):
        self.vehicle = agent_vehicle
        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        self.percentagePedestriansRunning = 1      # how many pedestrians will run
        self.percentagePedestriansCrossing = 1     # how many pedestrians will walk through the road
        # 1. take all the random locations to spawn
        spawn_points = []
        self.end_points = []
        vehicle_loc = carla.Location()
        vehicle_transform = self.vehicle.get_transform()
        vehicle_loc= vehicle_transform.location
        vehicle_rot = vehicle_transform.rotation
        #print(vehicle_transform)
        i = 0
        j = 0
        yaw = math.pi*vehicle_rot.yaw/180
        self.yaw = yaw
        spawn_point = carla.Transform()
        self.walker_speed = []
        self.walkers = []
        self.walker_actors = []
        self.walker_controllers = []
        walkers_list = []
        self.start_walking_distances = []
        self.wait_time = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')

        #if not self.spawn_location_calculated:
        while ((i < self.number_of_walkers)):
            try:

                loc = self.world.get_random_location_from_navigation()
                if loc!=None:

                    vehicle_xyz = np.array([vehicle_loc.x, vehicle_loc.y, vehicle_loc.z])
                    vehicle_xyz = np.array([vehicle_loc.x, vehicle_loc.y, vehicle_loc.z])
                    pedestrian_xyz = np.array([loc.x, loc.y, loc.z])
                    R = transforms3d.euler.euler2mat(0, 0, yaw).T
                    ped_loc_relative = np.dot(R, pedestrian_xyz-vehicle_xyz)

                    ped_velocity_vector = np.array([0, -1, 0])

                    self.ped_velocity_relative = np.dot(R, ped_velocity_vector)


                    pitch = 0.0
                    roll = 0.0
                    x_to_car =ped_loc_relative[0]
                    y_to_car =ped_loc_relative[1]

                    if ((4<x_to_car < 118) &(3<y_to_car<15)):
                        #print(x_to_car,y_to_car)

                        spawn_point.location = loc
                        #walker_bp = self.random_walker_bp()
                        walker_bp = random.choice(self.blueprintsWalkers)
                        if walker_bp.has_attribute('is_invincible'):
                            walker_bp.set_attribute('is_invincible', 'false')
                        # set the max speed
                        if walker_bp.has_attribute('speed'):
                            if (random.random() > self.percentagePedestriansRunning):
                                # walking
                                walker_speed = (walker_bp.get_attribute('speed').recommended_values[1])
                            else:
                                # running
                                walker_speed = (walker_bp.get_attribute('speed').recommended_values[2])
                        else:
                            print("Walker has no speed")
                            walker_speed = (0.0)
                        walker = self.world.try_spawn_actor(walker_bp, spawn_point)
                        if (walker == None):
                            pass
                        else:

                            end_spawn_point = spawn_point
                            end_spawn_point.location.x = loc.x-30
                            spawn_points.append(spawn_point)
                            self.walkers.append(walker)
                            self.wait_time.append(random.randint(30, 200))
                            start_walking_distance = random.uniform(2, 10)
                            self.start_walking_distances.append(start_walking_distance)



                            #print(loc)
                            i = i + 1


            finally:
                pass
        print("Spawn points initilized")
        self.spawnable_points = spawn_points
        self.spawn_location_calculated = True

        '''else:
            spawn_points = random.sample(self.spawnable_points, self.number_of_walkers)
            print('here')
            for spawn_point in spawn_points:
                walker_bp = self.random_walker_bp()
                walker = self.world.try_spawn_actor(walker_bp, spawn_point)
                print(walker)
                if (walker == None):
                    pass
                else:
                    self.walkers.append(walker)
                    start_walking_distance = random.uniform(2, 20)
                    self.start_walking_distances.append(start_walking_distance)'''


        print('spawned  %d walkers.' % (len(self.walkers)))
        self.world.set_pedestrians_cross_factor(self.percentagePedestriansCrossing)
        self.applied_control = np.full((1,self.number_of_walkers), False, dtype=bool)
        self.clock_counts = np.full((1,self.number_of_walkers), 0, dtype=np.uint8)
        self.finished_control = np.full((1,self.number_of_walkers), False, dtype=bool)
        print(self.applied_control.shape)



        return self.walkers



        #print(i)
        #print("we have the points!")
    def random_walker_bp(self):
        walker_bp = random.choice(self.blueprintsWalkers)
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        # set the max speed
        if walker_bp.has_attribute('speed'):
            if (random.random() > self.percentagePedestriansRunning):
                # walking
                walker_speed = (walker_bp.get_attribute('speed').recommended_values[1])
            else:
                # running
                walker_speed = (walker_bp.get_attribute('speed').recommended_values[2])
        else:
            print("Walker has no speed")
            walker_speed = (0.0)
        return walker_bp

    def ApplyControl(self):

        #while True:
        control = carla.WalkerControl()



        control.speed = 0.9
        control.direction.y = self.ped_velocity_relative[1]
        control.direction.x = self.ped_velocity_relative[0]
        control.direction.z = 0
        #print("here")
        # 3. we spawn the walker controller

        for i in range(len(self.walkers)):
            control.speed = random.uniform(3, 6)/3.6
            self.walkers[i].apply_control(control)
    def kill_walkers(self):
        for actor in self.walkers:
            actor.destroy()
    def walker_collision_creator(self,vehicle):

        vehicle_transform = vehicle.get_transform()
        v = vehicle.get_velocity()
        mps = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        vehicle_loc= vehicle_transform.location
        yaw = self.yaw
        vehicle_xyz = np.array([vehicle_loc.x, vehicle_loc.y, vehicle_loc.z])
        i = 0
        for walker in self.walkers:

            if not self.finished_control[0,i]:

                loc = walker.get_location()
                pedestrian_xyz = np.array([loc.x, loc.y, loc.z])
                R = transforms3d.euler.euler2mat(0, 0, yaw).T
                ped_loc_relative = np.dot(R, pedestrian_xyz - vehicle_xyz)
                x_to_car = ped_loc_relative[0]
                y_to_car = ped_loc_relative[1]

                if not self.applied_control[0,i]:
                    start_walking_distance = self.start_walking_distances[i]
                    if x_to_car<start_walking_distance+random.uniform(5, 20) and x_to_car>1:
                        control = carla.WalkerControl()
                        control.direction.y = self.ped_velocity_relative[1]
                        control.direction.x = self.ped_velocity_relative[0]
                        control.direction.z = 0
                        control.speed = random.uniform(3, 12)/3.6
                        walker.apply_control(control)
                        self.applied_control[0,i] = True

                if y_to_car<0.5 and self.clock_counts[0,i]<self.wait_time[i]:
                    control = carla.WalkerControl()
                    control.direction.y = 0
                    control.direction.x = 1
                    control.direction.z = 0
                    control.speed = 0
                    walker.apply_control(control)
                    self.clock_counts[0,i] = self.clock_counts[0,i]+1

                elif self.clock_counts[0,i]>=self.wait_time[i]:
                    control = carla.WalkerControl()
                    control.direction.y = self.ped_velocity_relative[1]
                    control.direction.x = self.ped_velocity_relative[0]
                    control.direction.z = 0
                    control.speed = 5/3.6
                    walker.apply_control(control)
                    self.finished_control[0, i] = True


            i = i + 1







def connect_loop():
    vehicle = None
    client = carla.Client("localhost", 2000)
    client.set_timeout(2.0)

    world = client.get_world()

    for i in range(10000):
        while (vehicle == None):
            for actor in world.get_actors().filter('vehicle.*'):
                if actor.attributes['role_name'] == 'rlcar':
                    vehicle = actor
                    print('found rlcar')
            time.sleep(0.1)

        try:
            pedestrians = CreateWalkers(client=client,number_of_walkers=10)
            pedestrians.SpawnWalkers(agent_vehicle=vehicle)

            pedestrians.ApplyControl()
        finally:
            pass
        time.sleep(7)
        pedestrians.kill_walkers()


'''def main():
    try:

        WalkerBatch = CreateWalkers()
        WalkerBatch.SpawnWalkers()
        # print(carla.ActorList)

    finally:
        WalkerBatch.kill_walkers()


        #print("all cleaned up!")

'''

if __name__ == "__main__":
    connect_loop()





