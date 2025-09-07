import numpy as np
import math
from collections import namedtuple
from util import collision_cir_cir
from util import num_seg_matrix

class radioactive_source:
    def __init__(self, id, init_state=[], radius=[], **kwargs):
        self.id = int(id)
        if isinstance(init_state, list): 
            init_state = np.array(init_state, ndmin=2).T
        self.source_state = init_state
        self.previous_state = init_state
        self.init_state = init_state
        self.radius = radius
        self.radius_collision = round(radius + kwargs.get('radius_exp', 0.1), 2)
        # 定义模型函数
        self.S = 3.14e-3
        self.p = 0.3
        # 混泥土对γ射线的线衰减系数,cm^-1
        self.u_i = 0.15
        self.world_map = kwargs.get('world_map', './map_image/map_100_100_6.png')
        self.point_step_weight = kwargs.get('point_step_weight', 2)

    def reset(self, random_activity=False):
        self.source_state = self.init_state
        self.previous_state = self.init_state

    def source_check(self, robot, components):
        circle = namedtuple('circle', 'x y r')
        
        self_circle = circle(robot.state[0][0],robot.state[1][0], robot.radius_collision)
        # if robot.source_flag == True:
        #     return True
        # check collision beetween robots and source
        for source in components['sources'].source_list:
            # if not robot.source_flag:
            temp_circle = circle(source.source_state[0][0], source.source_state[1][0], self.radius_collision)
            if collision_cir_cir(self_circle, temp_circle):
                robot.source_flag = True
                # print('collisions between robots and source')
                return True
        robot.source_flag = False
        return False
    
    def model(self, pose, components):
        start_point = np.array([pose.x, pose.y])
        end_point = self.source_state[0:2, 0]
        segment = [start_point, end_point]
        thickness = num_seg_matrix(segment, components['map_matrix'], components['xy_reso'])
        r = (self.source_state[0,0] - pose.x)**2 + (self.source_state[1,0] - pose.y)**2
        # if r < self.radius + pose.r:
        #     return 0
        return self.source_state[2,0] * math.exp(-thickness*self.point_step_weight*components['xy_reso']/0.01*self.u_i) * self.S * self.p / (4 * np.pi * (r))

    def source_detector(self, state, radius, components):
            circle = namedtuple('circle', 'x y r')

            self_circle = circle(state[0][0], state[1][0], radius)
            return math.log(self.model(self_circle, components) + 1)
