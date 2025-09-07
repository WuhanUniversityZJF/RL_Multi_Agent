import functools
import numpy as np
import random
import pygame
import yaml
import sys
from gym.spaces import Discrete, Box
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
# init
from world import env_plot, mobile_robot, obs_circle, obs_polygon
import cv2
from env.env_robot import env_robot
from env.env_obs_cir import env_obs_cir
from env.env_obs_line import env_obs_line
from env.env_source import env_source
from env.env_obs_poly import env_obs_poly

from scipy.interpolate import RectBivariateSpline
import gymnasium
from gymnasium.utils import seeding
import math
# Define constants
REWARD_FOUND_ITEM = 300

class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human, rgb_array"], "name": "multi_robot_env", "render_fps": 30}

    def __init__(self, render_mode="rgb_array", obs_type="coords", world_name=None, max_cycles=200, local_ratio=1, **kwargs):
        self.render_mode = render_mode
        self._obs_type = obs_type
        world_name = sys.path[0] + '/' + world_name
        self.local_ratio = local_ratio

        with open(world_name) as file:
            com_list = yaml.load(file, Loader=yaml.FullLoader)
            world_args = com_list['world']
            self._world_height = world_args.get('world_height', 1000)
            self._world_width = world_args.get('world_width', 1000)
            self._offset_x = world_args.get('offset_x', 0)
            self._offset_y = world_args.get('offset_y', 0)
            self._step_time = world_args.get('step_time', 0.1)
            self._world_map =  world_args.get('world_map', None)
            self._xy_reso = world_args.get('xy_resolution', 1)
            self._yaw_reso = world_args.get('yaw_resolution', 5)
            self._offset = np.array([self._offset_x, self._offset_y])

            self.robots_args = com_list.get('robots', dict())
            self._robot_number = kwargs.get('robot_number', self.robots_args.get('robot_number', 0) )
            
            # obs_cir
            self.obs_cirs_args = com_list.get('obs_cirs', dict())
            self.obs_cir_number = self.obs_cirs_args.get('number', 0)
            self.obs_step_mode = self.obs_cirs_args.get('obs_step_mode', 0)

            # obs line
            self.obs_lines_args = com_list.get('obs_lines', dict())

            # obs polygons
            self.obs_polygons_args = com_list.get('obs_polygons', dict())
            self.vertexes_list = self.obs_polygons_args.get('vertexes_list', [])
            self.obs_poly_num = self.obs_polygons_args.get('number', 0)

            # source
            self.source_args = com_list.get('source', dict())

        # world
        self._screen_height = int(self._world_height / self._xy_reso)
        self._screen_width = int(self._world_width / self._xy_reso)
        pygame.init()
        self.renderOn = False
        self.max_cycles = max_cycles
        self._screen = pygame.Surface([self._screen_width, self._screen_height])
            
        self.possible_agents = [f"agent_{i}" for i in range(self._robot_number)]
        self.agent_name_mapping = dict(zip(self.possible_agents, range(self._robot_number)))
        # init
        self.components = dict()
        self.init_environment(**kwargs)

    def init_environment(self, robot_class=mobile_robot, obs_cir_class=obs_circle, obs_polygon_class=obs_polygon,  **kwargs):

        if self._world_map != None:
            world_map_path = sys.path[0] + '/' + self._world_map
            map_img = cv2.imread(world_map_path, cv2.IMREAD_GRAYSCALE)
            map_matrix = cv2.resize(map_img, (self._screen_width, self._screen_height), interpolation=cv2.INTER_LINEAR)
            # cv2.imshow('Image', map_matrix)
            # cv2.waitKey(0)
            map_matrix[map_matrix>255/2] = 255
            map_matrix[map_matrix<=255/2] = 0
            self.map_matrix = map_matrix.T
        else:
            self.map_matrix = None

        self.components['map_matrix'] = self.map_matrix
        self.components['xy_reso'] = self._xy_reso
        self.components['offset'] = np.array([self._offset_x, self._offset_y])
        self.components['obs_lines'] = env_obs_line(**{**self.obs_lines_args, **kwargs})
        self.obs_line_states=self.components['obs_lines'].obs_line_states
        self.components['sources'] = env_source(components=self.components, **{**self.source_args, **kwargs})
        self.source_list = self.components['sources'].source_list
        
        self.components['obs_circles'] = env_obs_cir(obs_cir_class=obs_cir_class, obs_cir_num=self.obs_cir_number, step_time=self._step_time, components=self.components, **{**self.obs_cirs_args, **kwargs})
        self.obs_cir_list = self.components['obs_circles'].obs_cir_list

        self.components['obs_polygons'] = env_obs_poly(obs_poly_class=obs_polygon_class, vertex_list=self.vertexes_list, obs_poly_num=self.obs_poly_num, **{**self.obs_polygons_args, **kwargs})
        self.obs_poly_list = self.components['obs_polygons'].obs_poly_list 

        self.components['robots'] = env_robot(robot_class=robot_class, step_time=self._step_time, components=self.components, **{**self.robots_args, **kwargs})
        self.robot_list = self.components['robots'].robot_list

        self.world_plot = env_plot(self._world_width, self._world_height, self._xy_reso, self.components,map_matrix=self.map_matrix ,offset_x=self._offset_x, offset_y=self._offset_y, **kwargs)
        if self._robot_number > 0:
            self.robot = self.components['robots'].robot_list[0]

        self.components['robots'].init_transforms()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        if agent in self.possible_agents:
          if self._obs_type == "image":
              return Box(low=0, high=255, shape=(self._screen_width, self._screen_height, 3), dtype=np.uint8)
          else:
              return Box(low=-np.inf, high=np.inf, shape=(10 + 2*(len(self.robot_list) - 1) +2*len(self.obs_cir_list) + self.robot_list[0].lidar.data_num, ), dtype=np.float32)
        else:
            return None  # 处理未知的代理
        
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
      if agent in self.possible_agents:
          return Box(low=-self.robot_list[0].vel_max[0, 0], high=self.robot_list[0].vel_max[1, 0], shape=(2,), dtype=np.float32)  # 速度和角速度的动作空间
      else:
          return None  # 处理未知的代理
 
    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        self.enable_render(self.render_mode)
        _double_buffer = self.world_plot.draw_dyna_components()
        self._screen.blit(_double_buffer, (0, 0))
        if self.render_mode == "rgb_array":
            observation = np.array(pygame.surfarray.pixels3d(self._screen))
            return np.transpose(observation, axes=(1, 0, 2))
        elif self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            # pygame.image.save(self._screen, 'multi_map.png')
            return
        
    def enable_render(self, mode="human"):
        if not self.renderOn and mode == "human":
            self._screen = pygame.display.set_mode(self._screen.get_size())
            self.clock = pygame.time.Clock()
            self.renderOn = True

    def render_coordinate_array(self, robot, source):
        old_detector = source.source_detector(robot.previous_state, robot.radius, self.components)
        detector = source.source_detector(robot.state, robot.radius, self.components)
        r_d = detector - self.max_detector
        min_dis, min_angle = robot.lidar.nearest_obstacle(robot.state)
        entity_pos = []
        for obs_cir in self.obs_cir_list:  # world.entities:
            entity_pos.append(obs_cir.state[0:2,0] - robot.state[0:2,0])
        source_pos = []
        source_pos.append(source.source_state[0:2,0] - robot.state[0:2,0])
        other_pos = []
        for other in self.robot_list:
            if other.id == robot.id:
                continue
            other_pos.append(other.state[0:2,0] - robot.state[0:2,0])
        observation = np.concatenate([*source_pos, [-r_d], *entity_pos, *other_pos, robot.lidar.range_data.flatten(), [robot.lidar.range_max - min_dis], [min_angle], [robot.radius_collision], robot.vel_diff.flatten(), robot.state[0:2, 0]])
        return observation
    

    def close(self):
        if self.render_mode == "human":
          pygame.quit()  # 关闭 Pygame 窗口

    def reset(self, seed=None, options=None, reset_mode=4):
        self.agents = self.possible_agents[:]
        self.components['sources'].sources_reset(reset_mode=0, include_robot=True)
        self.components['robots'].robots_reset(reset_mode=reset_mode, include_robot=True, search_source=True)
        self.world_plot.robot_trajectories = {agent: [] for agent in range(self._robot_number)}
        for robot in self.robot_list:
            robot.cal_lidar_range(self.components)
        self.steps = 0 
        self.wait = 0
        self.max_detector = self.source_list[0].source_detector(self.robot_list[0].state, self.robot_list[0].radius, self.components)
        observations = {agent: self.render_coordinate_array(self.robot_list[i], self.source_list[0]) for i, agent in enumerate(self.agents)}
        self.state = observations
        infos = {agent: {} for agent in self.agents}
        detector = {agent:math.exp(self.source_list[0].source_detector(self.robot_list[i].state, self.robot_list[i].radius, self.components)) - 1 for i, agent in enumerate(self.agents)}
        return observations, infos, detector

    def step(self, actions):
        self.steps += 1
        for i, (agent, action) in enumerate(actions.items()):
            self.robot_list[i].move_forward(action)
            self.robot_list[i].cal_lidar_range(self.components)
            self.source_list[0].source_check(self.robot_list[i],self.components)
            self.robot_list[i].collision_check(self.components)
            
        rewards, infos, env_termination = self.cal_reward(self.robot_list, self.source_list)
        # 提前终止、游戏胜利和终止都应该env_termination = True而env_truncation表示截断
        if min(infos.values()) == True and not env_termination:
            self.wait += 1
            if self.wait > self.max_cycles*0.05:
              rewards = {key: value + (self.max_cycles - self.steps)*value for key, value in rewards.items()}
              env_termination = True
              # pygame.image.save(self._screen, 'multi_map.png')
        # 一荣俱荣、一损俱损
        terminations = {agent: env_termination for agent in self.agents}
        env_truncation = self.steps >= self.max_cycles
        truncations = {agent: env_truncation for agent in self.agents}
        observations = {agent: self.render_coordinate_array(self.robot_list[i], self.source_list[0]) for i, agent in enumerate(self.agents)} 
        self.state = observations
        
        detector = {agent:math.exp(self.source_list[0].source_detector(self.robot_list[i].state, self.robot_list[i].radius, self.components)) - 1 for i, agent in enumerate(self.agents)}
        if env_truncation or env_termination:
            if env_truncation and min(infos.values()) == False:
              rewards = {key: value - self.max_cycles * math.exp(-self.steps/self.max_cycles) for key, value in rewards.items()}
            self.agents = []

        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminations, truncations, infos, detector

    def cal_reward(self, robot_list, source_list, beta=1, theta=1, mu=1, gamma=1):
        rewards = {}
        infos = {}

        global_reward = 0.0
        if self.local_ratio is not None:
          for s in source_list:
              dists = [
                  np.sqrt(np.sum(np.square(a.state[0:2,0] - s.source_state[0:2,0])))
                  for a in robot_list
              ]
              global_reward -= sum(dists)

        agent_reward = {}
        find_num = sum(robot.source_flag for robot in robot_list)
        for robot in robot_list:
            detector = source_list[0].source_detector(robot.state, robot.radius, self.components)
            # 不能取等
            if detector - self.max_detector > 0.0:
                r_d = math.exp(detector - self.max_detector)
            else:
                r_d = - mu*max(self.max_detector - detector, 0)
            self.max_detector = detector if self.max_detector < detector else self.max_detector
            
            if robot.source_flag:
              r_d += 10*math.exp(find_num/self._robot_number)
            min_dis, _ = robot.lidar.nearest_obstacle(robot.state)
            if min_dis < robot.radius*1.5:
                r_c = - beta * max(robot.lidar.range_max - min_dis, 0)
            else:
                r_c = 0.0
            agent_reward[robot.id] = sum([r_d, r_c])

            if self.local_ratio is not None:
                reward = (
                    global_reward * (1 - self.local_ratio)
                    + agent_reward[robot.id] * self.local_ratio
                )
            else:
                reward = agent_reward[robot.id]
            rewards[f"agent_{robot.id}"] = reward
            infos[f"agent_{robot.id}"] = robot.source_flag

        env_termination = any(robot.collision_flag for robot in robot_list)
        if env_termination:
          rewards = {key: value - self.max_cycles * math.exp(-self.steps/self.max_cycles) for key, value in rewards.items()}
        return rewards, infos, env_termination
    
def env(render_mode=None):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def raw_env(render_mode=None):
    env = parallel_env(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env
