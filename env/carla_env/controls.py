# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================
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


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla
import math





class VehicleControl(object):
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


    def actions(self, action, clock):
        v = self._world.player.get_velocity()
        mps = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        kmh = int(3.6 * mps)
        milliseconds = clock.get_time()
        if (action == 0 and (not (kmh > 30))):
            if self._control.throttle<0.4:
                self._control.throttle = 0.4
            self._control.throttle = min(self._control.throttle + 0.2, 1)
            self._control.brake = 0
        if action == 1:
            self._control.throttle = self._control.throttle #0
            self._control.brake = self._control.brake #min(self._control.brake + 0.02, 1)
        if action == 2:
            if self._control.brake<0.4:
                self._control.brake = 0.4
            self._control.brake = min(self._control.brake + 0.2, 1)
            #print(self._control.brake)
            self._control.throttle = 0.0
        #else:
            #self._control.brake = 0
        self._world.player.apply_control(self._control)