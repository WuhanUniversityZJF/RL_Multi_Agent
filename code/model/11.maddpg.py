import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import multirobotenv
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import os
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from datetime import datetime
from tqdm import tqdm
def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


class QNet(nn.Module):
    def __init__(self, dim_state, dim_action, args):
        super().__init__()
        self.fc1 = nn.Linear(args.num_agents * (dim_state + dim_action), args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)
        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, state, action):
        sa = torch.cat(state + action, 1)
        x = F.relu(self.fc1(sa))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PolicyNet(nn.Module):
    def __init__(self, dim_state, dim_action, args):
        super().__init__()
        self.max_action = args.max_action
        self.fc1 = nn.Linear(dim_state, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, dim_action)
        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.max_action * torch.tanh(self.fc3(x))
        return x
    
class ReplayBuffer(object):
    def __init__(self, num_agents, obs_dim_n, action_dim_n, args):
        self.N = num_agents  # The number of agents
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.count = 0
        self.current_size = 0
        self.device = args.device

        self.buffer_obs_n = {f'agent_{i}': np.empty((self.buffer_size, obs_dim_n)) for i in range(self.N)}
        self.buffer_a_n = {f'agent_{i}': np.empty((self.buffer_size, action_dim_n)) for i in range(self.N)}
        self.buffer_r_n = {f'agent_{i}': np.empty((self.buffer_size, 1)) for i in range(self.N)}
        self.buffer_s_next_n = {f'agent_{i}': np.empty((self.buffer_size, obs_dim_n)) for i in range(self.N)}
        self.buffer_done_n = {f'agent_{i}': np.empty((self.buffer_size, 1)) for i in range(self.N)}


    def store_transition(self, obs_n, a_n, r_n, obs_next_n, done_n):
        for i in range(self.N):
            agent_key = f'agent_{i}'
            self.buffer_obs_n[agent_key][self.count] = obs_n[agent_key]
            self.buffer_a_n[agent_key][self.count] = a_n[agent_key]
            self.buffer_r_n[agent_key][self.count] = r_n[agent_key]
            self.buffer_s_next_n[agent_key][self.count] = obs_next_n[agent_key]
            self.buffer_done_n[agent_key][self.count] = done_n[agent_key]
        self.count = (self.count + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(self, ):
        index = np.random.choice(self.current_size, size=self.batch_size, replace=False)
        batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = {}, {}, {}, {}, {}
        for i in range(self.N):
            agent_key = f'agent_{i}'
            batch_obs_n[agent_key] = torch.tensor(self.buffer_obs_n[agent_key][index], dtype=torch.float).to(self.device)
            batch_a_n[agent_key] = torch.tensor(self.buffer_a_n[agent_key][index], dtype=torch.float).to(self.device)
            batch_r_n[agent_key] = torch.tensor(self.buffer_r_n[agent_key][index], dtype=torch.float).to(self.device)
            batch_obs_next_n[agent_key] = torch.tensor(self.buffer_s_next_n[agent_key][index], dtype=torch.float).to(self.device)
            batch_done_n[agent_key] = torch.tensor(self.buffer_done_n[agent_key][index], dtype=torch.float).to(self.device)

        return batch_obs_n, batch_a_n, batch_r_n, batch_done_n, batch_obs_next_n

class MADDPG(nn.Module):
    def __init__(
        self,
        max_action=1,
        num_states=6,
        num_actions=5,
        args=None,
        agent_id=0,
    ):
        super().__init__()
        self.id = agent_id
        self.max_action = max_action
        self.num_states = num_states
        self.num_actions = num_actions
        self.device = args.device
        self.use_grad_clip = args.use_grad_clip

        self.Q1 = QNet(self.num_states, self.num_actions, args).to(self.device)
        self.Mu = PolicyNet(self.num_states, self.num_actions, args).to(self.device)
        self.target_Q1 = QNet(self.num_states, self.num_actions, args).to(self.device)
        self.target_Mu = PolicyNet(self.num_states, self.num_actions, args).to(self.device)
        self.target_Q1.load_state_dict(self.Q1.state_dict())
        self.target_Mu.load_state_dict(self.Mu.state_dict())
        self.Q1_optimizer = torch.optim.Adam(self.Q1.parameters(), lr=args.lr_c)
        self.Mu_optimizer = torch.optim.Adam(self.Mu.parameters(), lr=args.lr_a)

    def get_action(self, state, noise_std=0):
        state = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        action = self.Mu(state).cpu().data.numpy().flatten()
        noise =  np.random.normal(0, noise_std, size=self.num_actions)
        action = (action + noise).clip(-self.max_action, self.max_action)
        return action.astype(np.float32)

    def compute_value_loss(self, args, agents, s_batch, a_batch, r_batch, d_batch, ns_batch):
        # Compute target_Q
        with torch.no_grad():
            # 让目标策略网络做预测。
            batch_a_next = {agent_id: agents[agent_id].target_Mu(ns_batch[agent_id]) for agent_id in agents}

            # 让两个目标价值网络做预测。
            q1 = self.target_Q1(list(ns_batch.values()), list(batch_a_next.values()))
            # 计算 TD 目标。
            y = r_batch[self.id] + args.gamma * q1 * (1 - d_batch[self.id])

        # 让两个价值网络做预测。
        qvals1 = self.Q1(list(s_batch.values()), list(a_batch.values()))
        value_loss1 = F.mse_loss(y, qvals1)
        return value_loss1

    def compute_policy_loss(self, s_batch, a_batch):
        a_batch[self.id] = self.Mu(s_batch[self.id])
        policy_loss = -self.Q1(list(s_batch.values()), list(a_batch.values())).mean()
        return policy_loss

    def soft_update(self, tau=0.01):
        def soft_update_(target, source, tau_=0.01):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau_) + param.data * tau_)

        soft_update_(self.target_Q1, self.Q1, tau)
        soft_update_(self.target_Mu, self.Mu, tau)

    def save_model(self, output_dir):
        save_path = os.path.join(output_dir, f"{self.id}_model.bin")
        torch.save(self.Mu.state_dict(), save_path)

    def load_model(self, output_dir):
        save_path = os.path.join(output_dir, f"{self.id}_model.bin")
        self.Mu.load_state_dict(torch.load(save_path))

    def learn(self, replay_buffer: ReplayBuffer, args, agents: dict, infos: dict):
        s_batch, a_batch, r_batch, d_batch, ns_batch = replay_buffer.sample()
        # Optimize the critic
        value_loss1 = self.compute_value_loss(args, agents, s_batch, a_batch, r_batch, d_batch, ns_batch)
        infos[self.id].log["value_loss1"].append(value_loss1.item())
        self.Q1_optimizer.zero_grad()
        value_loss1.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.Q1.parameters(), 10.0)
        self.Q1_optimizer.step()

        # Optimize the actor
        policy_loss = self.compute_policy_loss(s_batch, a_batch)
        # Optimize the actor
        self.Mu_optimizer.zero_grad()
        policy_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.Mu.parameters(), 10.0)
        self.Mu_optimizer.step()
        self.soft_update(args.tau)
        infos[self.id].log["policy_loss"].append(policy_loss.item())

class INFO:
    def __init__(self):
        self.log = defaultdict(list)
        self.episode_length = 0
        self.episode_reward = 0
        self.max_episode_reward = -float("inf")

    def put(self, done, reward, detector):
        if done is True:
            self.episode_length += 1
            self.episode_reward += reward
            self.log["episode_length"].append(self.episode_length)
            self.log["episode_reward"].append(self.episode_reward)

            if self.episode_reward > self.max_episode_reward:
                self.max_episode_reward = self.episode_reward

            self.episode_length = 0
            self.episode_reward = 0
        else:
            self.episode_length += 1
            self.episode_reward += reward
            self.log["detector"].append(detector)
            
    def reset(self):
        self.log = defaultdict(list)

def train(args, env, agents: dict, replay_buffer: ReplayBuffer, writer: SummaryWriter):
    if args.add_model:
      for agent in agents.values():
          agent.load_model(args.output_dir)
    step = 0  # global step counter
    max_reward = 300
    sum_reward = 0
    episode_data = []
    infos = {agent_id: INFO() for agent_id in env.possible_agents}
    for episode in tqdm(range(args.episode_num), desc="Training Episodes"):
        state, _, detector = env.reset()
        for agent_id in env.possible_agents:
            infos[agent_id].put(False, 0, detector[agent_id])
        while env.agents:
            step += 1
            if step < args.warmup_steps:
              # action = {agent: env.components['robots'].robot_list[i].cal_des_vel().reshape(2,) for i, agent in enumerate(env.agents)}
              action = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
            else:
              action = {agent_id:agents[agent_id].get_action(state[agent_id], noise_std=args.noise_std) for agent_id in env.agents}
            next_state, reward, terminated, truncated, _, detector = env.step(action)
            done = {agent: terminated[agent] or truncated[agent] for agent in terminated}
            replay_buffer.store_transition(state, action, reward, next_state, done)
            for agent_id in env.possible_agents:
              infos[agent_id].put(done[agent_id], reward[agent_id], detector[agent_id])
            state = next_state

            if step >= args.warmup_steps:
                for agent_id in env.possible_agents:
                    agents[agent_id].learn(replay_buffer, args, agents, infos)

        # episode finishes
        if (episode + 1) % 20 == 0:  # print info every 20 episodes
            sum_reward = 0
            current_time = datetime.now()
            for agent_id, info in infos.items():  # record reward
                avg_reward = np.mean(info.log["episode_reward"][-20:])
                value_loss1 = np.mean(info.log["value_loss1"][-20:]) if len(info.log["value_loss1"]) > 0 else 0
                value_loss2 = np.mean(info.log["value_loss2"][-20:]) if len(info.log["value_loss2"]) > 0 else 0
                policy_loss = np.mean(info.log["policy_loss"][-20:]) if len(info.log["policy_loss"]) > 0 else 0
                episode_data.append({"time": current_time.strftime("%Y-%m-%d %H:%M:%S"),"episode": episode + 1,"agent_id": agent_id, "reward": avg_reward, "value_loss1": value_loss1, "value_loss2": value_loss2, "policy_loss": policy_loss})
                # print(f"episode={episode+1:d}, {agent_id} reward={avg_reward:.2f}, value loss1={value_loss1:.4f}, value loss2={value_loss2:.4f}, policy loss={policy_loss:.4f}")
                sum_reward +=avg_reward
                infos[agent_id].reset()
            writer.add_scalar('train_step_rewards:',sum_reward, global_step=episode)
            writer.add_scalar(f'{agent_id}\'s value_loss1:',value_loss1, global_step=episode)
            writer.add_scalar(f'{agent_id}\'s value_loss2:',value_loss2, global_step=episode)
            writer.add_scalar(f'{agent_id}\'s policy_loss:',policy_loss, global_step=episode)
            
            # 将信息保存到DataFrame
            if (episode + 1) % 100 == 0:
              df = pd.DataFrame(episode_data)
              csv_file_path = f"{args.output_dir}/episode_info.csv"
              df.to_csv(csv_file_path, mode='a', header=not os.path.exists(csv_file_path), index=False)
              episode_data.clear()
              
            if sum_reward > max_reward:
                max_reward = sum_reward
                for agent_id in env.possible_agents:
                  agents[agent_id].save_model(args.output_dir)
            
def eval(args, env, agents: dict, writer: SummaryWriter):
    episode_data = []
    detector_info = []
    infos = {agent_id: INFO() for agent_id in env.possible_agents}
    gif_dir = os.path.join(args.output_dir, 'gif' + str(args.seed))
    os.makedirs(gif_dir, exist_ok=True)
    gif_num = len([file for file in os.listdir(gif_dir)])  # current number of gif
    for agent in agents.values():
        agent.load_model(args.output_dir)
    for episode in tqdm(range(args.evaluate_times), desc="Evaluating Episodes"):
        state, _, detector = env.reset()
        for agent_id in env.possible_agents:
            infos[agent_id].put(False, 0, detector[agent_id])
        frame_list = []  # used to save gif
        while env.agents:
            action = {agent_id:agents[agent_id].get_action(state[agent_id], noise_std=0) for agent_id in env.agents}
            next_state, reward, terminated, truncated, _, detector = env.step(action)
            if env.render_mode == "rgb_array" and args.evaluate_times < 30:
              frame_list.append(Image.fromarray(env.render()))
            done = {agent: terminated[agent] or truncated[agent] for agent in terminated}
            for agent_id in env.possible_agents:
              infos[agent_id].put(done[agent_id], reward[agent_id], detector[agent_id])
            state = next_state

        # episode finishes
        sum_reward = 0
        current_time = datetime.now()
        for i in range(env.steps):
          detector_info.append({"steps": i+1, "agent_0":infos["agent_0"].log["detector"][i], "agent_1":infos["agent_1"].log["detector"][i], "agent_2":infos["agent_2"].log["detector"][i]})
        detector_info.append({"steps": "avg_reward", "agent_0":infos["agent_0"].log["episode_reward"][-1], "agent_1":infos["agent_1"].log["episode_reward"][-1], "agent_2":infos["agent_2"].log["episode_reward"][-1]})
        for agent_id, info in infos.items():  # record reward
          avg_reward = info.log["episode_reward"][-1]
          # print(f"episode={episode+1:d} steps={env.steps}, {agent_id} reward={avg_reward:.2f}")
          sum_reward += avg_reward
        episode_data.append({"time": current_time.strftime("%Y-%m-%d %H:%M:%S"),"episode":episode+1, "steps": env.steps, "sum_reward": sum_reward})
        if env.render_mode == "rgb_array" and args.evaluate_times < 30:
          frame_list[0].save(os.path.join(gif_dir, f'out{gif_num + episode + 1}.gif'),
                             save_all=True, append_images=frame_list[1:], duration=1, loop=0)
        # writer.add_scalar('evaluate_step_rewards:',sum_reward, global_step=episode)
    # 将信息保存到DataFrame
    df = pd.DataFrame(detector_info)
    csv_file_path = f"{args.output_dir}/detector_info.csv"
    df.to_csv(csv_file_path, mode='a', header=not os.path.exists(csv_file_path), index=False)
    episode_data.clear()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MADDPG and MATD3 in MPE environment")
    parser.add_argument("--env", default="multi_robot_world_10_3", type=str, help="Environment name.")
    parser.add_argument("--max_action", default=1.0, type=float, help="Action scale, [-max, max].")
    parser.add_argument('--episode_num', type=int, default=50000, help='total episode num during training procedure')
    parser.add_argument("--evaluate_freq", type=float, default=20, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=1, help="Evaluate times")
    parser.add_argument("--warmup_steps", default=10000, type=int, help="Warmup steps without training.")
    parser.add_argument("--algorithm", type=str, default="MADDPG", help="MATD3 or MATD3_LSTM or MADDPG")
    parser.add_argument("--buffer_size", type=int, default=int(1e6), help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--noise_std", type=float, default=0.1, help="The std of Gaussian noise for exploration")
    parser.add_argument("--lr_a", type=float, default=1e-3, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=1e-3, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--seed", default=14, type=int, help="Random seed.")
    parser.add_argument("--tau", type=float, default=0.01, help="Softly update the target network")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
    parser.add_argument("--output_dir", default="output", type=str, help="Output directory.")
    parser.add_argument("--do_train", action="store_true", help="Train policy.")
    parser.add_argument("--do_eval", action="store_true", help="Evaluate policy.")
    parser.add_argument("--add_model", type=bool, default=False, help="Loading model")

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir = os.path.join(args.output_dir, '{}/env_{}_seed_{}'.format(args.algorithm, args.env, args.seed))
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    world_name = args.env + '.yaml'
    env = multirobotenv.parallel_env(render_mode="rgb_array", world_name=world_name)
    env_eval = multirobotenv.parallel_env(render_mode="human", world_name=world_name)
    args.num_agents = len(env.possible_agents)
    num_actions = env.action_space(env.possible_agents[0]).shape[0]
    args.max_action = env.action_space(env.possible_agents[0]).high[0]
    observation_size = env.observation_space(env.possible_agents[0]).shape[0]

    agents = {agent_id:MADDPG(max_action=args.max_action, num_states=observation_size, num_actions=num_actions, args=args, agent_id=agent_id) for agent_id in env.possible_agents}
    replay_buffer = ReplayBuffer(args.num_agents, observation_size, num_actions, args)
            
    if args.do_train:
        writer = SummaryWriter(log_dir='runs/{}/env_{}_seed_{}'.format(args.algorithm, args.env, args.seed))
        writer.add_text("args", str(args))
        train(args, env, agents, replay_buffer, writer)
        writer.close()
    if args.do_eval:
        eval(args, env_eval, agents, None)