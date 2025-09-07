import math
import numpy as np
import random

def sample_motion_model_velocity(x, y, teta, v, w, dt, a= [0.02, 0.002, 0.02, 0.2, 0.2, 0.2]):
    a1, a2, a3, a4, a5, a6 = a
    v1 = v + random.gauss(0, a1 * abs(v) + a2 * abs(w))
    w1 = w + random.gauss(0, a3 * abs(v) + a4 * abs(w))
    gama1 = random.gauss(0, a5 * abs(v) + a6 * abs(w))

    if abs(w1) < 1e-5:
        x1 = x + v1 * dt * math.cos(teta)
        y1 = y + v1 * dt * math.sin(teta)
        teta1 = teta
    else:
        x1 = x - (v1 / w1) * math.sin(teta) + (v1 / w1) * math.sin(teta + w1 * dt)
        y1 = y + (v1 / w1) * math.cos(teta) - (v1 / w1) * math.cos(teta + w1 * dt)
        teta1 = teta + w1 * dt + gama1 * dt

    return x1, y1, teta1

# reference: probabilistic robotics[book], motion model P127
def motion_diff(current_state, vel, sampletime, noise = True, alpha = [0.02, 0.002, 0.02, 0.2, 0.2, 0.2]):

    # vel: np.array([[linear vel],[angular vel]])
    # current_state: np.array([[x], [y], [theta]])
    # alpha: control noise includes linear, angular, orientation

    if noise == True:
        std_linear = np.sqrt(alpha[0] * (vel[0, 0] ** 2) + alpha[1] * (vel[1, 0] ** 2))
        std_angular = np.sqrt(alpha[2] * (vel[0, 0] ** 2) + alpha[3] * (vel[1, 0] ** 2))
        gamma = alpha[4] * (vel[0, 0] ** 2) + alpha[5] * (vel[1, 0] ** 2)

        vel_noise = vel + np.array([std_linear, std_angular]).T
    else:
        vel_noise = vel

    vt = float(vel_noise[0, 0])
    omegat = float(vel_noise[1,0])

    theta = float(wraptopi(current_state[2, 0]))
    next_state = np.zeros(current_state.shape)
    if omegat >= 0.01 or omegat <= -0.01:
        next_state[0] = current_state[0] - (vt / omegat) * math.sin(theta) + (vt / omegat) * math.sin(theta + omegat * sampletime)
        next_state[1] = current_state[1] + (vt / omegat) * math.cos(theta) - (vt / omegat) * math.cos(theta + omegat * sampletime)
        next_state[2] = current_state[2] + omegat * sampletime + gamma * sampletime
    else:
        next_state[0] = current_state[0] + vt * sampletime * math.cos(theta)
        next_state[1] = current_state[1] + vt * sampletime * math.sin(theta)
        next_state[2] = current_state[2] + omegat * sampletime
    # if omegat >= 0.01 or omegat <= -0.01:
    #     ratio = vt/omegat
    #     next_state = current_state + np.array([ [-ratio * sin(theta) + ratio * sin(theta + omegat * sampletime)], 
    #                                    [ratio * cos(theta) - ratio * cos(theta + omegat * sampletime)], 
    #                                    [omegat * sampletime]])

    # else:
    #     next_state = current_state + np.array([[vt * sampletime * cos(theta)], [vt * sampletime * sin(theta)], [0]])

    next_state[2, 0] =  float(wraptopi(next_state[2, 0])) 
    
    return next_state 


def motion_omni(current_state, vel, sampletime, noise = False, control_std = [0.01, 0.01]):

    # vel: np.array([[vel x], [vel y]])
    # current_state: np.array([[x], [y]])

    if noise == True:
        vel_noise = vel + np.random.normal([[0], [0]], scale = [[control_std[0]], [control_std[1]]])
    else:
        vel_noise = vel

    next_state = current_state + vel_noise * sampletime
    
    return next_state 


# reference: Modern Robotics: Mechanics, Planning, and Control[book], 13.3.1.3 car-like robot
def motion_ackermann(state, wheelbase=1, vel=np.zeros((2, 1)), steer_limit=math.pi/2, step_time=0.1, ack_mode='default', theta_trans=True):
    
    # l: wheel base
    # state: 0, x
    #        1, y
    #        2, phi, heading direction
    #        3, psi, steering angle
    # motion_mode: default: vel: linear velocity, angular velocity of steer
    #              steer:   vel：linear velocity, steer angle
    #              simplify: vel: linear velocity, rotation rate, do not consider the steer angle    
    
    phi = state[2, 0]  
    psi = state[3, 0] 
    
    if ack_mode == 'default':
        co_matrix = np.array([ [math.cos(phi), 0],  [math.sin(phi), 0], [math.tan(psi) / wheelbase, 0], [0, 1] ])
        d_state = co_matrix @ vel
        new_state = state + d_state * step_time
    
    elif ack_mode == 'steer':
        co_matrix = np.array([ [math.cos(phi), 0],  [math.sin(phi), 0], [math.tan(psi) / wheelbase, 0], [0, 0] ])
        d_state = co_matrix @ vel
        new_state = state + d_state * step_time
        new_state[3, 0] = np.clip(vel[1, 0], -steer_limit, steer_limit)

    elif ack_mode == 'simplify':

        new_state = np.zeros((4, 1))
        co_matrix = np.array([ [math.cos(phi), 0],  [math.sin(phi), 0], [0, 1] ])
        d_state = co_matrix @ vel
        new_state[0:3] = state[0:3] + d_state * step_time

    if theta_trans:
        new_state[2, 0] = wraptopi(new_state[2, 0]) 
        
    new_state[3, 0] = np.clip(new_state[3, 0], -steer_limit, steer_limit) 

    return new_state

def motion_acker_pre(state, wheelbase=1, vel=1, psi=0, steer_limit=math.pi/4, pre_time=2, time_step=0.1):
    
    # l: wheel base
    # vel: linear velocity, steer
    # state: 0, x
    #        1, y
    #        2, phi, heading direction
    #        3, psi, steering angle
    
    cur_time = 0

    while cur_time < pre_time:
        phi = state[2, 0] 
        d_state = np.array([ [vel*math.cos(phi)], [vel*math.sin(phi)], [vel*math.tan(psi) / wheelbase], [0] ])
        print(d_state * time_step)
        new_state = state + d_state * time_step
        new_state[2, 0] = wraptopi(new_state[2, 0]) 
        new_state[3, 0] = np.clip(psi, -steer_limit, steer_limit) 

        cur_time = cur_time + time_step
        state = new_state

    return new_state

def motion_acker_step(state, gear=1, steer=0, step_size=0.5, min_radius=1, include_gear=False):
    
    # psi: steer angle
    # state: 0, x
    #        1, y
    #        2, phi, heading direction
    # gear: -1, 1
    # steer: 1, 0, -1, left, straight, right
    if not isinstance(state, np.ndarray):
        state = np.array([state]).T

    cur_x = state[0, 0]
    cur_y = state[1, 0]
    cur_theta = state[2, 0]

    curvature = steer * 1/min_radius
    
    rot_theta = abs(steer) * step_size * curvature * gear
    trans_len = (1 - abs(steer)) * step_size * gear

    rot_matrix = np.array([[math.math.cos(rot_theta), -math.math.sin(rot_theta)], [math.math.sin(rot_theta), math.math.cos(rot_theta)]])
    trans_matrix = trans_len * np.array([[math.math.cos(cur_theta)], [math.math.sin(cur_theta)]]) 

    center_x = cur_x + math.math.cos(cur_theta + steer * math.pi/2) * min_radius
    center_y = cur_y + math.math.sin(cur_theta + steer * math.pi/2) * min_radius
    center = np.array([[center_x], [center_y]])

    if include_gear:
        new_state = np.zeros((4, 1))
        new_state[0:2] = np.around(rot_matrix @ (state[0:2] - center) + center + trans_matrix, 4)
        new_state[2, 0] = np.around(mod(cur_theta + rot_theta), 4)
        new_state[3, 0] = gear
    else:
        new_state = np.zeros((3, 1))
        new_state[0:2] = np.around(rot_matrix @ (state[0:2] - center) + center + trans_matrix, 4)
        new_state[2, 0] = np.around(mod(cur_theta + rot_theta), 4)
    
    return new_state


# def motion_acker_step(state, gear=1, steer=0, step_size=0.5, min_radius=1, include_gear=False):
    
#     # psi: steer angle
#     # state: 0, x
#     #        1, y
#     #        2, phi, heading direction
#     # gear: -1, 1
#     # steer: 1, 0, -1, left, straight, right
#     if not isinstance(state, np.ndarray):
#         state = np.array([])

#     cur_x = state[0, 0]
#     cur_y = state[1, 0]
#     cur_theta = state[2, 0]

#     curvature = steer * 1/min_radius
    
#     rot_theta = abs(steer) * step_size * curvature * gear
#     trans_len = (1 - abs(steer)) * step_size * gear

#     rot_matrix = np.array([[cos(rot_theta), -sin(rot_theta)], [sin(rot_theta), cos(rot_theta)]])
#     trans_matrix = trans_len * np.array([[cos(cur_theta)], [sin(cur_theta)]]) 

#     center_x = cur_x + cos(cur_theta + steer * math.pi/2) * min_radius
#     center_y = cur_y + sin(cur_theta + steer * math.pi/2) * min_radius
#     center = np.array([[center_x], [center_y]])

#     if include_gear:
#         new_state = np.zeros((4, 1))
#         new_state[0:2] = np.around(rot_matrix @ (state[0:2] - center) + center + trans_matrix, 4)
#         new_state[2, 0] = np.around(mod(cur_theta + rot_theta), 4)
#         new_state[3, 0] = gear
#     else:
#         new_state = np.zeros((3, 1))
#         new_state[0:2] = np.around(rot_matrix @ (state[0:2] - center) + center + trans_matrix, 4)
#         new_state[2, 0] = np.around(mod(cur_theta + rot_theta), 4)
    
#     return new_state


def wraptopi(radian):
    # -math.pi to math.pi

    if radian > math.pi:
        radian2 = radian - 2 * math.pi
    elif radian < -math.pi:
        radian2 = radian + 2 * math.pi
    else:
        radian2 = radian

    return radian2

def mod(theta):

    theta = theta % (2*math.pi)
    if theta < - math.pi: 
        return theta + 2*math.pi
    if theta >= math.pi: 
        return theta - 2*math.pi

    return theta


