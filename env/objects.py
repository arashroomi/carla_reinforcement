import pygame
import random
class Car:
    def __init__(self, clock, x, y, v):
        self.x = x
        self.y = y
        self.v = v
        self.length = 3.5
        self.width = 2
        self.acceleration = 0
        self.decceleration = 0
        self.max_accelration = 3
        self.max_brake = 5.8
        self.col_history = []

        self.current_time = clock
        self.bounding_box()

    def bounding_box(self):
        self.x1 = self.length/2 + self.x
        self.x2 = -self.length / 2 + self.x
        self.y1 = self.width/2 + self.y
        self.y2 = -self.width / 2 + self.y



    def throttle(self,throttle_ratio):
        self.acceleration = self.max_accelration * throttle_ratio



    def brake(self, brake_ratio):
        self.decceleration = -self.max_brake * brake_ratio

    def apply_control(self,throttle,brake):
        self.throttle(throttle)
        self.brake(brake)
        ##print(self.decceleration)


    def calculate_location(self,clock):
        total_acc = self.decceleration + self.acceleration

        previous_time = self.current_time
        self.current_time = clock
        dt = (self.current_time - previous_time)


        if self.v<0:
            self.v = 0
            total_acc = 0
        #print(total_acc,self.v,self.decceleration , self.acceleration)

        delta_x = self.v * dt#0.5 * total_acc * (dt ** 2) + self.v * dt
        self.v = total_acc * dt + self.v
        #print(self.x, self.v,total_acc)
        if self.x<0:
            self.x =0
        if delta_x<=0:
            delta_x=0
        self.x = self.x + delta_x
        self.bounding_box()
        #print(self.v,total_acc)


class Walker:
    def __init__(self, x, y,agent):
        self.x = x
        self.y = y
        self.x_to_agent = self.x
        self.y_to_agent = self.y
        self.z = 1
        self.vx = 0
        self.vy = 0
        self.ped_size = 1
        self.length = self.ped_size
        self.width = self.ped_size
        self.collision = False
        self.agent = agent
        self.blocked = False


        self.v_walking = random.uniform(1.2,2.5)
        self.ttc = 0
        self.reserve_atr1 = 0
        self.reserve_atr2 = 0
        self.reserve_atr3 = 0
        self.reserve_atr4 = 0
        self.passing = True

        self.current_time = 0
        self.bounding_box()

    def bounding_box(self):
        self.x1 = self.length / 2 + self.x
        self.x2 = -self.length / 2 + self.x
        self.y1 = self.width / 2 + self.y
        self.y2 = -self.width / 2 + self.y



    def apply_control(self, v, direction):

        self.vx = 0#v*(direction[0]/abs(sum(direction)))
        self.vy = v#v*(direction[1]/abs(sum(direction)))
    def loc_to_agent(self,agent):
        x1_to_agent = self.x1 - agent.x
        x2_to_agent = self.x2 - agent.x
        y1_to_agent = self.y1 - agent.y
        y2_to_agent = self.y2 - agent.y
        self.x_to_agent = self.x - agent.x
        self.y_to_agent = self.y - agent.y

        corners = [[x1_to_agent,y1_to_agent],[x1_to_agent,y2_to_agent],[x2_to_agent,y1_to_agent],[x2_to_agent,y2_to_agent]]

        for corner in corners:
            x,y = corner
            if -agent.length/2<x<agent.length/2 and -agent.width/2<y<agent.width/2:
                self.collision = True
                #print('herreee')





    def calculate_location(self, clock):
        self.loc_to_agent(self.agent)
        previous_time = self.current_time
        self.current_time = clock
        dt = (self.current_time - previous_time)
        #self.x =  0#self.vx * dt+self.x
        self.y =  self.vy * dt+self.y
        self.bounding_box()


class Building:
    def __init__(self, x1, y1,length,width):
        self.x1 = x1
        self.y1 = y1
        self.z = 1
        self.length = length
        self.width = width
        #self.bounding_box()
        self.x2 = self.length + self.x1
        self.y2 = self.width + self.y1


    def bounding_box(self):
        self.x2 = self.length + self.x1

        self.y2 = -self.width  + self.y1
        if self.y1<0: self.y2 = self.width  + self.y1
        #return [[self.x1, self.y1], [self.x1, self.y2], [self.x2, self.y1], [self.x2, self.y2]]
