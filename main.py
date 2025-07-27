import torch
import numpy as np
import wandb
import argparse
from pathlib import Path
from replay_buffer import ReplayBuffer
from maddpg import MADDPG
from tqdm import tqdm
import copy
import os
import yaml
from environment.env import Env


def choose_device():
    gpu_count = torch.cuda.device_count()  # 获取 GPU 数量
    if gpu_count > 0:
        min_utilization = 100
        device = 0
        for i in range(gpu_count):
            utilization = torch.cuda.utilization(i)
            if utilization < min_utilization:
                device = i
                min_utilization = utilization
        return torch.device(f"cuda:{device}")
    else:
        return torch.device("cpu")


class Runner:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.device = args.device
        self.env_name = env_name
        self.number = number
        self.seed = seed
        self.args.n_agents = 3
        self.env = Env(self.args)
        self.args.N = self.args.n_agents  # The number of agents
        self.args.obs_dim_n = [57, 57, 57]  # obs dimensions of N agents
        # actions dimensions of N agents
        # self.args.action_dim_n = [2+3, 2+3, 2+3]
        self.args.action_dim_n = [2, 2, 2]
        print("obs_dim_n={}".format(self.args.obs_dim_n))
        print("action_dim_n={}".format(self.args.action_dim_n))

        ################ PPO hyperparameters ################
        has_continuous_action_space = False  # continuous action space; else discrete
        self.env_change_freq = 1           # env init_state change frequency
        self.update_timestep = 64
        action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
        action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
        min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
        action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)

        K_epochs = 80               # update policy for K epochs in one PPO update

        eps_clip = 0.2          # clip parameter for PPO
        gamma = 0.99            # discount factor

        lr_actor = 0.0001       # learning rate for actor network
        lr_critic = 0.001       # learning rate for critic network
        #####################################################
        from PPO import PPO
        self.offload_agent = PPO(42, 3, 256, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std, device, multi_buffer=True) # PPO2

        # Create N agents
        print(f"Algorithm: {args.algorithm}")
        if self.args.algorithm == "MADDPG":
            self.agent_n = [MADDPG(args, agent_id)
                            for agent_id in range(args.N)]
        elif args.algorithm == "matd3":
            from matd3 import MATD3
        elif args.algorithm == "matd3_attn":
            from matd3_attn import MATD3
        elif args.algorithm == "matd3_d2l_attn":
            from matd3_d2l_attn import MATD3
        elif args.algorithm == "matd3_mlla" or args.algorithm == "matd3_global_ppo":
            from matd3_mlla import MATD3
        elif args.algorithm == "matd3_divide_reward":
            from matd3 import MATD3
        else:
            print("Wrong!!!")
            exit(0)

        self.attn = None
        self.agent_n = [MATD3(args, agent_id, self.attn).to(self.device)
                        for agent_id in range(args.N)]

        self.replay_buffer = ReplayBuffer(self.args)
        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0

        self.noise_std = self.args.noise_std_init  # Initialize noise_std

    def run(self, pre_train=False, pre_episode=0):
        experiment_dir = self.args.run_dir / self.args.experiment_name
        gifs_dir = str(experiment_dir / 'gifs')
        if not os.path.exists(gifs_dir):
            os.makedirs(gifs_dir)
        self.best_model_dir = str(experiment_dir / 'best_model')
        if not os.path.exists(self.best_model_dir):
            os.makedirs(self.best_model_dir)

        self.checkpoint_file = f'{self.best_model_dir}/checkpoint.yaml'
        self.checkpoint = {
            'lastest': {
                'reward': 0,
                'episode': -1
            },
            'best': {
                'reward': 0,
                'episode': -1
            }
        }
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                self.checkpoint = yaml.unsafe_load(f)
        print(f"{self.checkpoint = }")
        pre_episode, max_reward = self.load_model(tag='lastest')
        max_rewards = [0] * self.args.n_agents
        metrics = []
        for episode in tqdm(range(self.args.num_episodes)):
            obs_n, avail_action_mask = self.env.reset(save_trajectory = (episode+pre_episode) % 50 == 0)
            episode_reward = 0
            average_move_reward, average_offload_reward = 0, 0
            sum_rewards = np.zeros((self.args.n_agents, 2))
            collision_cnt = np.zeros(self.args.N)
            average_cost = np.zeros(self.args.N)
            for _ in range(self.args.episode_limit):
                offload_obs = self.env.offload_obs()
                offload_actions, action_log_probs = [], []
                for i in range(self.args.N):
                     offload_action, action_log_prob = self.offload_agent.select_action(offload_obs[i]) 
                     offload_actions.append(offload_action)
                     action_log_probs.append(action_log_prob)
                from collections import namedtuple
                Transition = namedtuple('Transition', ['state', 'action', 'reward', 'a_log_prob', 'next_state'])

                # Each agent selects actions based on its own local observations(add noise for exploration)
                move_actions = [agent.choose_action(obs, noise_std=self.noise_std, mask=mask)
                       for agent, obs, mask in zip(self.agent_n, obs_n, avail_action_mask)]
                
                a_n = [move_actions, offload_actions]

                obs_next_n, avail_action_mask, r_n, done_n, info = self.env.step(
                    copy.deepcopy(a_n))
                collision_cnt += info['collision_cnt']
                average_cost += info['average_cost']

                sum_rewards += r_n
                for r in r_n:
                    average_move_reward += r[0]
                    average_offload_reward += r[1]

                offload_obs_next = self.env.offload_obs()
                for i in range(self.args.N):
                    trans = Transition(offload_obs[i], offload_actions[i], r_n[i][1], action_log_probs[i], offload_obs_next[i])
                    if self.offload_agent.store_transition(trans, i):
                        self.offload_agent.update(i)

                r_n = [sum(r) for r in r_n]
                episode_reward += sum(r_n)
                # Store the transition
                self.replay_buffer.store_transition(
                    obs_n, move_actions,  r_n, obs_next_n, done_n)
                obs_n = obs_next_n
                self.total_steps += 1

                # Decay noise_std
                if self.args.use_noise_decay:
                    self.noise_std = self.noise_std - self.args.noise_std_decay if self.noise_std - \
                        self.args.noise_std_decay > self.args.noise_std_min else self.args.noise_std_min

                if self.replay_buffer.current_size > self.args.batch_size:
                    # Train each agent individually
                    for agent_id in range(self.args.N):
                        self.agent_n[agent_id].train(
                            self.replay_buffer, self.agent_n)

                if all(done_n):
                    break

            if (episode+pre_episode) == 0 or episode_reward > max_reward:
                max_reward = episode_reward
                self.env.save_video(f'{gifs_dir}/best.gif')
                self.save_model('best', episode_reward,
                                episode + pre_episode)
            for i in range(self.args.n_agents):
                if (episode+pre_episode) == 0 or sum_rewards[i].sum() > max_rewards[i]:
                    max_rewards[i] = sum_rewards[i].sum()

            metric  = dict( episode = episode + pre_episode,
                            episode_reward = episode_reward,
                            average_move_reward = average_move_reward / self.args.episode_limit,
                            average_offload_reward = average_offload_reward / self.args.episode_limit,
                            collision_cnt = sum(collision_cnt),
                            average_cost = sum(average_cost) / (self.args.N * self.args.episode_limit),
                            max_reward = max_reward)
            
            for i in range(self.args.n_agents):
                metric[f'reward/agent{i}'] = sum_rewards[i].sum()
                metric[f'max_reward/agent{i}'] = max_rewards[i]
                metric[f'collision_cnt/agent{i}'] = collision_cnt[i]
                metric[f'average_cost/agent{i}'] = average_cost[i] / self.args.episode_limit
                # Agent
                actor_metrics = self.agent_n[i].actor_metrics
                if len(actor_metrics) > 0:
                    metric[f'actor_loss/agent{i}'] = sum(actor_metrics)/len(actor_metrics)
                    self.agent_n[i].actor_metrics.clear()
                    critic_metrics = self.agent_n[i].critic_metrics
                    metric[f'critic_loss/agent{i}'] = sum(critic_metrics)/len(critic_metrics)
                    self.agent_n[i].critic_metrics.clear()
                
            metrics.append(metric)
            if episode % self.args.wandb_upload_step == 0:
                print(f"wandb {episode = } {self.args.wandb_upload_step = }")
                for metric in metrics:
                    wandb.log(metric)
                metrics.clear()

            print("episode:{} \t episode_reward:{}".format(
                episode+pre_episode, episode_reward))
            if (episode+pre_episode) % 50 == 0:
                self.env.save_video(f'{gifs_dir}/{episode+pre_episode}.gif')
            if (episode+pre_episode) % 50 == 0:
                self.save_model('lastest', episode_reward,
                                episode + pre_episode)
        self.env.close()

    def test(self):
        self.noise_std = 0  # noise_std 0
        experiment_dir = self.args.run_dir / self.args.experiment_name
        gifs_dir = str(experiment_dir / 'test')
        if not os.path.exists(gifs_dir):
            os.makedirs(gifs_dir)
        self.best_model_dir = str(experiment_dir / 'best_model')
        if not os.path.exists(self.best_model_dir):
            os.makedirs(self.best_model_dir)

        self.checkpoint_file = f'{self.best_model_dir}/checkpoint.yaml'
        self.checkpoint = {
            'lastest': {
                'reward': 0,
                'episode': -1
            },
            'best': {
                'reward': 0,
                'episode': -1
            }
        }
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                self.checkpoint = yaml.unsafe_load(f)
        print(f"{self.checkpoint = }")
        self.load_model(tag='lastest')
        env_files = get_test_envs("test_env")
        for episode, file in enumerate(env_files):
            obs_n, avail_action_mask = self.env.reset(save_trajectory = True, mode = "test")

            load_env(self.env, f"test_env/{file}")
            obs_n, avail_action_mask = self.env.obs()
            episode_reward = 0
            average_move_reward, average_offload_reward = 0, 0
            collision_cnt = np.zeros(self.args.N)
            average_cost = np.zeros(self.args.N)
            for _ in range(self.args.episode_limit):
                a_n = [agent.choose_action(obs, noise_std=self.noise_std, mask=mask)
                        for agent, obs, mask in zip(self.agent_n, obs_n, avail_action_mask)]
                obs_next_n, avail_action_mask, r_n, done_n, info = self.env.step(
                    copy.deepcopy(a_n))
                collision_cnt += info['collision_cnt']
                average_cost += info['average_cost']
                for r in r_n:
                    average_move_reward += r[0]
                    average_offload_reward += r[1]
                r_n = [sum(r) for r in r_n]
                episode_reward += sum(r_n)
                obs_n = obs_next_n
                self.total_steps += 1

                if all(done_n):
                    break
            if sum(collision_cnt) == 0:
                print(f"{episode = } no collision!!!")

            self.env.save_video(f'{gifs_dir}/{file.split(".")[0]}.gif')

            print("episode:{} episode_reward:{}".format(episode, episode_reward))

        self.env.close()

    def save_model(self, tag, reward, episode):
        self.checkpoint[tag]['reward'] = reward
        self.checkpoint[tag]['episode'] = episode
        with open(self.checkpoint_file, 'w') as f:
            yaml.dump(self.checkpoint, f)
        episode = tag
        for agent_id in range(self.args.N):
            self.agent_n[agent_id].save_model(
                self.best_model_dir, episode, agent_id)

        # offload PPO
        if self.args.algorithm == "matd3_global_ppo":
            torch.save(self.offload_agent.actor_net.state_dict(), f"{self.best_model_dir}/global_ppo_actor.pkl")
            torch.save(self.offload_agent.critic_net.state_dict(), f"{self.best_model_dir}/global_ppo_critic.pkl")

        # attn
        path_checkpoint = "{}/checkpoint_epoch_{}_attn.pkl".format(
            self.best_model_dir, episode)
        if self.attn is not None:
            torch.save(self.attn.state_dict(), path_checkpoint)

    def load_model(self, tag='no',  episode=-1):
        if tag == 'no':  # 读取指定的模型
            for agent_id in range(self.args.N):
                self.agent_n[agent_id].load_model(
                    self.best_model_dir, episode, agent_id)
            return episode+1, 0

        episode = self.checkpoint[tag]['episode']
        if episode != -1:  # lastest/best
            for agent_id in range(self.args.N):
                self.agent_n[agent_id].load_model(
                    self.best_model_dir, tag, agent_id)

            # offload PPO
            if self.args.algorithm == "matd3_global_ppo":
                self.offload_agent.actor_net.load_state_dict(torch.load(f"{self.best_model_dir}/global_ppo_actor.pkl"))
                self.offload_agent.critic_net.load_state_dict(torch.load(f"{self.best_model_dir}/global_ppo_critic.pkl"))
            elif self.args.algorithm == "matd3_global_ppo":
                for i in range(self.args.N):
                    self.offload_agents[i].actor_net.load_state_dict(torch.load(f"{self.best_model_dir}/global_ppo_agent{i}_actor.pkl"))
                    self.offload_agents[i].critic_net.load_state_dict(torch.load(f"{self.best_model_dir}/global_ppo_agent{i}_critic.pkl"))

            # reward为max_reward
            return episode+1, self.checkpoint['best']['reward']

        return 0, 0


def wandb_init(args):
    wandb.init(config=args,
                        project="multi-robots-offloading-master",
                        name=args.experiment_name,
                        group=args.algorithm,
                        dir=str(run_dir),
                        job_type="training",
                        reinit=True)
    wandb.define_metric("episode")
    wandb.define_metric("*", step_metric="episode")

    # arti_code = wandb.Artifact(name=args.experiment_name, type="code")
    # arti_code.add_dir('../code')
    # ignore_files = ['__pycache__/']
    # for ignore_file in ignore_files:
    #     arti_code.remove(ignore_file) # 忽略上面添加文件夹中的子文件或子文件夹(末尾加/)
    # wandb.log_artifact(arti_code)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Hyperparameters Setting")
    parser.add_argument('--n_training_threads', type=int,
                        default=4, help="Number of torch threads for training")
    parser.add_argument("--algorithm", type=str, default="matd3_global_ppo", choices=["matd3","matd3_mlla","mappo"])
    parser.add_argument("--num_episodes", type=int,
                        default=20000, help=" Maximum number of training steps")
    parser.add_argument('--use_wandb', action='store_false', default=True,
                        help="Whether to use weights&biases, if not, use tensorboardX instead")
    parser.add_argument('--wandb_upload_step', action='store_false', default=1,
                        help="wandb upload log per 10 steps")
    parser.add_argument("--episode_limit", type=int, default=100,
                        help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=100,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--test_times", type=float,
                        default=100, help="Test times")
    parser.add_argument("--max_action", type=float,
                        default=1.0, help="Max action")

    parser.add_argument("--buffer_size", type=int, default=int(1e6),
                        help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int,
                        default=1024, help="Batch size 1024")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--noise_std_init", type=float, default=0.2,
                        help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_std_min", type=float, default=0.05,
                        help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_decay_steps", type=float, default=5e5,
                        help="How many steps before the noise_std decays to the minimum 5e6")
    parser.add_argument("--use_noise_decay", type=bool,
                        default=False, help="Whether to decay the noise_std")
    parser.add_argument("--lr_a", type=float, default=0.001,
                        help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=0.0001,
                        help="Learning rate of critic")
    parser.add_argument("--gamma", type=float,
                        default=0.95, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.01,
                        help="Softly update the target network")
    parser.add_argument("--use_orthogonal_init", type=bool,
                        default=True, help="Orthogonal initialization")
    parser.add_argument("--use_attention", type=bool,
                        default=False, help="use Attention")
    parser.add_argument("--use_grad_clip", type=bool,
                        default=True, help="Gradient clip")
    # --------------------------------------MATD3--------------------------------------------------------------------
    parser.add_argument("--policy_noise", type=float,
                        default=0.2, help="Target policy smoothing")
    parser.add_argument("--noise_clip", type=float,
                        default=0.5, help="Clip noise")
    parser.add_argument("--policy_update_freq", type=int,
                        default=2, help="The frequency of policy updates")

    parser.add_argument("--train", type=bool,
                        default=True, help="train or test")

    args = parser.parse_args()

    device = choose_device()
    torch.set_num_threads(args.n_training_threads)
    torch.backends.cudnn.benchmark = True
    print(f"{device = }")
    args.device = device

    args.noise_std_decay = (args.noise_std_init -
                            args.noise_std_min) / args.noise_decay_steps

    args.experiment_name = 'matd3_global_ppo1'

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / args.algorithm
    args.run_dir = run_dir
    print(f"{run_dir = }")
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    env_names = ["simple_speaker_listener", "simple_spread"]
    env_index = 0
    runner = Runner(args, env_name=env_names[env_index], number=1, seed=0)
    if args.train:
        if args.use_wandb:
            # init wandb
            wandb_init(args)
        else:
            if not run_dir.exists():
                curr_run = 'run1'
            else:
                exst_run_nums = [int(str(folder.name).split('run')[
                                    1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
                if len(exst_run_nums) == 0:
                    curr_run = 'run1'
                else:
                    curr_run = 'run%i' % (max(exst_run_nums) + 1)
            run_dir = run_dir / curr_run
            if not run_dir.exists():
                os.makedirs(str(run_dir))
        runner.run()
    else:
        runner.test()
