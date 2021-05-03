from objects import Car
from car_env_one_ped import World
import numpy as np
import math
import pygame
import glob
import os
import sys


import carla


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

class CarlaVehControl():
    """Class that handles keyboard input."""
    def __init__(self, world):
        self._world = world
        self._control = carla.VehicleControl()
        self._lights = carla.VehicleLightState.NONE
        world.player.set_light_state(self._lights)
        self._steer_cache = 0.0
        self._control.throttle = 1.0
        self._control.brake = 0.0
        self._control_cache = 1

        self.throttle = 1.0
        self.brake = 0.0
        self._control_cache = 1

        self._kP = 1
        self._kI = 0.1
        self._kD = .01
        self.simulation_time = 0
        self.vehicle_speed = 0
        self.i_term_previous = 0

    def update_variables(self,clock):
        v = self._world.vehicle.v
        self.vehicle_speed_previous = self.vehicle_speed
        self.vehicle_speed = v

        self.t_previous = self.simulation_time
        self.simulation_time = clock

    def set_throttle(self, input_throttle):
        # Clamp the throttle command to valid bounds
        throttle = np.fmax(np.fmin(input_throttle, 1.0), 0.0)
        return throttle
    def set_brake(self, input_brake):
        # Clamp the steering command to valid bounds
        brake = np.fmax(np.fmin(input_brake, 1.0), 0.0)
        return brake


    def speed_actions(self, action):
        snap = self.world.get_snapshot()
        clock = snap.timestamp.elapsed_seconds
        self.update_variables(clock)
        if (action == 0):
            v_desired = -1
            self._world.camera_offset = 0
        if (action == 1):
            v_desired = -1
            self._world.camera_offset = (math.pi) * 0.3
        elif(action==2):
            v_desired = 5/3.6
            self._world.camera_offset = 0
        elif (action == 3):
            v_desired = 5/3.6
            self._world.camera_offset = (math.pi) * 0.3
        elif (action == 4):
            v_desired = 10/3.6
            self._world.camera_offset = 0
        elif (action == 5):
            v_desired = 30/3.6
            self._world.camera_offset = (math.pi) * 0.3

        self.PI_controller(v_desired)






    def PI_controller(self, v_desired):

        t = self.simulation_time

        time_step = t - self.t_previous
        speed_error = v_desired - self.vehicle_speed
        k_term = self._kP*speed_error
        i_term = float(self.i_term_previous) + self._kI*time_step*speed_error
        self.i_term_previous = i_term
        throttle_output = k_term + i_term

        self.throttle = self.set_throttle(throttle_output)  # in percent (0 to 1)
        if throttle_output<-0.0:
            brake_output = abs(throttle_output)
            self.throttle = 0
            #print(brake_output)
        else:
            brake_output = 0

        if v_desired == -1:
            self.brake = 1
            self.throttle = 0
        self.brake = self.set_brake(brake_output)  # in percent (0 to 1)
        self.vehicle.apply_control(self.throttle, self.brake)





