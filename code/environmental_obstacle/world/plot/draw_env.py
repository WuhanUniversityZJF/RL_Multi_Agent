
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pygame
import math
import cv2
class env_plot:
    def __init__(self, width=1000, height=1000, reso=0.01, components=dict(), map_matrix=None, offset_x = 0, offset_y=0, **kwargs):
        self.width = width
        self.height = height
        self.reso = reso
        self._double_buffer = pygame.Surface((self.width/self.reso, self.height/self.reso))
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.components = components

        self.map_matrix = map_matrix

        self.car_plot_list = []
        self.car_line_list = []
        self.robot_plot_list = []
        self.lidar_line_list = []
        self.car_img_show_list = []
        self.line_list = []
        self.dyna_obs_plot_list = []
        self.robot_trajectories = {agent: [] for agent in range(components['robots'].com['robots'].robot_number)}  # 用于保存机器人轨迹的字典

    # draw components
    def draw_components(self, **kwargs):

        if self.components['map_matrix'] is not None:
            self.ax.imshow(self.components['map_matrix'].T, cmap='Greys', origin='lower', extent=[self.offset_x, self.offset_x+self.width, self.offset_y, self.offset_y+self.height]) 
        
        self.draw_robots(self.components['robots'], **kwargs)
        self.draw_cars(self.components['cars'], **kwargs)
        self.draw_static_obs_cirs(self.components['obs_circles'], **kwargs)
        self.draw_obs_lines(self.components['obs_lines'], **kwargs)
    
    def draw_static_components(self, **kwargs):

        if self.components['map_matrix'] is not None:
            rgb = self.components['map_matrix'].T
            self.ax.imshow(self.components['map_matrix'].T, cmap='Greys', origin='lower', extent=[self.offset_x, self.offset_x+self.width, self.offset_y, self.offset_y+self.height]) 
            
        self.draw_static_obs_cirs(self.components['obs_circles'], **kwargs)
        self.draw_obs_lines(self.components['obs_lines'], **kwargs)
        self.draw_static_obs_polygons(self.components['obs_polygons'], **kwargs)

    def draw_dyna_components(self, **kwargs):
        
        robots = self.components.get('robots', None)
        source = self.components.get('sources', None) 
        obs_cirs = self.components.get('obs_circles', None) 
        self._double_buffer.fill((255, 255, 255))  # 清空双缓冲表面
        # bg_img = pygame.surfarray.make_surface(np.uint8(self.map_matrix))
        # self._double_buffer.blit(bg_img, (0, 0))

        if source is not None:
            self.draw_sources(source, **kwargs)

        if robots is not None:
            self.draw_robots(robots, **kwargs)
        
        if obs_cirs is not None:
            self.draw_static_obs_cirs(obs_cirs, **kwargs)



        return self._double_buffer
        

    def draw_robots(self, robots, **kwargs):
        for robot in robots.robot_list:
            self.draw_robot(robot, **kwargs)

    def draw_cars(self, cars, **kwargs):
        
        if len(cars.car_list) > 1:
            for car, color in zip(self.car_list, self.color_list):
                self.draw_car(car, car_color=color, text=True, **kwargs)
        else:
            for car in cars.car_list:
                self.draw_car(car, **kwargs)

    def draw_static_obs_cirs(self, obs_cirs, **kwargs):

        for obs_cir in obs_cirs.obs_cir_list:
            self.draw_static_obs_cir(obs_cir, **kwargs)

    def draw_sources(self, sources, **kwargs):
        
        for source in sources.source_list:
            self.draw_source(source, **kwargs)

    def draw_static_obs_polygons(self, obs_polygons, **kwargs):

        for obs in obs_polygons.obs_poly_list:
            self.draw_static_obs_polygon(obs, **kwargs)

    def draw_dyna_obs_cirs(self, obs_cirs, **kwargs):

        for obs_cir in obs_cirs.obs_cir_list:
            self.draw_dyna_obs_cir(obs_cir, **kwargs)

    def draw_obs_lines(self, obs_lines, **kwargs):
        
        for obs_line in obs_lines.obs_line_states:
            self.ax.plot([obs_line[0], obs_line[2]], [obs_line[1], obs_line[3]], 'k-')

    def draw_robot(self, robot, robot_color = (0, 125, 0), goal_color=(255, 0, 0), show_lidar=True, show_text=True, show_traj=True, traj_type='-g', **kwargs):
        
        x = robot.state[0][0]
        y = robot.state[1][0]
        theta = robot.state[2][0]
        # lidar
        if robot.lidar is not None and show_lidar:
            for i, point in enumerate(robot.lidar.inter_points[:, :]):
                pygame.draw.line(self._double_buffer,  (0, 0, 255, 128), (x/self.reso, y/self.reso), (point[0]/self.reso, point[1]/self.reso), 5)

        # 绘制机器人
        pygame.draw.circle(self._double_buffer, robot.color, (x/self.reso, y/self.reso), robot.radius/self.reso)
        arrow_length = 30*self.reso
        arrow_x = x + arrow_length * math.cos(theta)
        arrow_y = y + arrow_length * math.sin(theta)
        pygame.draw.line(self._double_buffer, robot.color, (x/self.reso, y/self.reso), (arrow_x/self.reso, arrow_y/self.reso), 5)

        # TODO
        # goal_x = robot.goal[0, 0]
        # goal_y = robot.goal[1, 0]
        # pygame.draw.circle(self._double_buffer, goal_color, (int(goal_x), int(goal_y), robot.radius))

        if show_text:
            font = pygame.font.Font(None, 30)
            text_surface = font.render('r' + str(robot.id), True, (0, 0, 0))
            self._double_buffer.blit(text_surface, ((x - 3* robot.radius * math.cos(theta + math.pi/4))/self.reso, (y - 3* robot.radius * math.sin(theta + math.pi/4))/self.reso))

            # self._double_buffer.blit(text_surface, ((x+sign*4*robot.radius*math.cos(theta) )/self.reso, (y+sign*4*robot.radius*math.cos(theta))/self.reso))  # 显示机器人名称
        # TODO
        if show_traj:
            # 添加当前机器人的轨迹点到字典
            self.robot_trajectories[robot.id].append((int(x / self.reso), int(y / self.reso)))
            if len(self.robot_trajectories[robot.id]) > 1:
                pygame.draw.lines(self._double_buffer, robot.color, False, self.robot_trajectories[robot.id], 4)

    def draw_static_obs_cir(self, obs_cir, obs_cir_color=(0,0,0), show_text=False, **kwargs):

        if obs_cir.obs_model == 'static':

            x = obs_cir.state[0,0]
            y = obs_cir.state[1,0]
            pygame.draw.circle(self._double_buffer, obs_cir_color, (x/self.reso, y/self.reso), obs_cir.radius/self.reso)

    def draw_source(self, source, source_color=(255,0,0), show_text=True, **kwargs):
        if source is not None:
            color_matrix = cv2.imread(source.world_map)
            color_matrix = cv2.resize(color_matrix, (int(self.width/self.reso), int(self.height/self.reso)), interpolation=cv2.INTER_LINEAR)
            color_matrix = cv2.cvtColor(color_matrix, cv2.COLOR_BGR2RGB)
            color_matrix_surface = pygame.surfarray.make_surface(np.transpose(color_matrix, axes=(1, 0, 2)))
            self._double_buffer.blit(color_matrix_surface, (0,0))
            # font = pygame.font.Font(None, 30)
            # goal_text_surface = font.render("s1", True, (0, 0, 0))
            # pygame.draw.circle(self._double_buffer, source_color, (x/self.reso, y/self.reso), source.radius/self.reso)
            # self._double_buffer.blit(goal_text_surface, ((x -source.radius)/self.reso, (y-4.2*source.radius)/self.reso))
    
    def draw_dyna_obs_cir(self, obs_cir, obs_cir_color='k', **kwargs):

        if obs_cir.obs_model != 'static':
            x = obs_cir.state[0,0]
            y = obs_cir.state[1,0]
            
            obs_circle = mpl.patches.Circle(xy=(x, y), radius = obs_cir.radius, color = obs_cir_color)
            obs_circle.set_zorder(2)
            self.ax.add_patch(obs_circle)

            self.dyna_obs_plot_list.append(obs_circle)

    def draw_static_obs_polygon(self, obs_polygon, obs_polygon_color='k', **kwargs):
        
        p = Polygon(obs_polygon.vertexes.T, facecolor = obs_polygon_color)
        self.ax.add_patch(p)

    # def draw_obs_line_list(self, **kwargs):
        
    #     for line in self.obs_line_list:
    #         # self.ax.plot(   line[0:2], line[2:4], 'k-')
    #         self.ax.plot( [line[0], line[2]], [line[1], line[3]], 'k-')

    def draw_vector(self, x, y, dx, dy, color='r'):
        arrow = mpl.patches.Arrow(x, y, dx, dy, width=0.2, color=color) 
        self.ax.add_patch(arrow)

    def draw_trajectory(self, traj, style='g-', label='line', show_direction=False, refresh=False, markersize=2):

        if isinstance(traj, list):
            path_x_list = [p[0, 0] for p in traj]
            path_y_list = [p[1, 0] for p in traj]

        elif isinstance(traj, np.ndarray):
            # raw*column: points * num
            path_x_list = [p[0] for p in traj.T]
            path_y_list = [p[1] for p in traj.T]
        
        line = self.ax.plot(path_x_list, path_y_list, style, label=label, markersize=markersize)

        if show_direction:

            if isinstance(traj, list):
                u_list = [cos(p[2, 0]) for p in traj]
                y_list = [sin(p[2, 0]) for p in traj]
            elif isinstance(traj, np.ndarray):
                u_list = [cos(p[2]) for p in traj.T]
                y_list = [sin(p[2]) for p in traj.T]

            self.ax.quiver(path_x_list, path_y_list, u_list, y_list)

        if refresh:
            self.line_list.append(line)

    def draw_point(self, point, label='point', markersize=2, color='k'):

        point = self.ax.plot(point[0], point[1], marker='o', markersize=markersize, color=color, label=label)
        # self.ax.legend()
        return point
        
        
    def com_cla(self):
        # self.ax.patches = []
        # self.ax.texts.clear()
        # self.ax.texts.clear()
        for text in self.ax.texts:
            text.remove()

        for robot_plot in self.robot_plot_list:
            robot_plot.remove()

        for car_plot in self.car_plot_list:
            car_plot.remove()

        # for line in self.car_line_list:
        #     line.pop(0).remove()
        for line in self.line_list:
            line.pop(0).remove()

        for lidar_line in self.lidar_line_list:
            lidar_line.pop(0).remove()
        
        for car_img in self.car_img_show_list:
            car_img.remove()

        for obs in self.dyna_obs_plot_list:
            obs.remove()
            

        self.car_plot_list = []
        self.robot_plot_list = []
        self.lidar_line_list = []
        self.car_img_show_list=[]
        self.line_list = []
        self.dyna_obs_plot_list = []

    # animation method 1
    def animate(self):

        self.draw_robot_diff_list()

        return self.ax.patches + self.ax.texts + self.ax.artists

    def show_ani(self):
        ani = animation.FuncAnimation(
        self.fig, self.animate, init_func=self.init_plot, interval=100, blit=True, frames=100, save_count=100)
        plt.show()
    
    def save_ani(self, name='animation'): 
        ani = animation.FuncAnimation(
        self.fig, self.animate, init_func=self.init_plot, interval=1, blit=False, save_count=300)
        ani.save(name+'.gif', writer='pillow')

    # # animation method 2
    def save_gif_figure(self, path, i, format='png'):

        if path.exists():
            order = str(i).zfill(3)
            plt.savefig(str(path)+'/'+order+'.'+format, format=format)
        else:
            path.mkdir()
            order = str(i).zfill(3)
            plt.savefig(str(path)+'/'+order+'.'+format, format=format)

    def create_animate(self, image_path, ani_path, ani_name='animated', keep_len=30, rm_fig_path=True):

        if not ani_path.exists():
            ani_path.mkdir()

        images = list(image_path.glob('*.png'))
        images.sort()
        image_list = []
        for i, file_name in enumerate(images):

            if i == 0:
                continue

            image_list.append(imageio.imread(file_name))
            if i == len(images) - 1:
                for j in range(keep_len):
                    image_list.append(imageio.imread(file_name))

        imageio.mimsave(str(ani_path)+'/'+ ani_name+'.gif', image_list)
        print('Create animation successfully')

        if rm_fig_path:
            shutil.rmtree(image_path)

    # old             
    def point_arrow_plot(self, point, length=0.5, width=0.3, color='r'):

        px = point[0, 0]
        py = point[1, 0]
        theta = point[2, 0]

        pdx = length * cos(theta)
        pdy = length * sin(theta)

        point_arrow = mpl.patches.Arrow(x=px, y=py, dx=pdx, dy=pdy, color=color, width=width)

        self.ax.add_patch(point_arrow)

    def point_list_arrow_plot(self, point_list=[], length=0.5, width=0.3, color='r'):

        for point in point_list:
            self.point_arrow_plot(point, length=length, width=width, color=color)

    
    def point_plot(self, point, markersize=2, color="k"):
        
        if isinstance(point, tuple):
            x = point[0]
            y = point[1]
        else:
            x = point[0,0]
            y = point[1,0]
    
        self.ax.plot([x], [y], marker='o', markersize=markersize, color=color)

    # plt 
    def cla(self):
        self.ax.cla()

    def pause(self, time=0.001):
        plt.pause(time)
    
    def show(self):
        plt.show()