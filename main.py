#! /usr/bin/env python3.6

import argparse
import torch
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.buffer import ReplayBuffer
from algorithms.attention_sac import AttentionSAC
from mapf_env import Environment
import rospkg


def run(config):
    pkg_dir   = os.path.dirname(os.path.abspath(__file__))
    model_dir = Path(os.path.join(pkg_dir, 'models', config.model_name))
    if not model_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1
    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(run_num)
    np.random.seed(run_num)

    # construct environment
    pkg_path = rospkg.RosPack().get_path('mapf_environment')
    img_path = os.path.join(pkg_path, 'maps', config.map)
    env = Environment(map_path=img_path, number_of_agents=config.no_agents,
        max_steps=config.episode_length, seed=run_num)

    os_shape = env.get_observation_space()
    as_shape = env.get_action_space()
    model = AttentionSAC(
        [{'num_in_pol': os_shape[0], 'num_out_pol': as_shape[0]}] * config.no_agents,
        [(os_shape[0], as_shape[0])] * config.no_agents,
        tau=config.tau,
        pi_lr=config.pi_lr,
        q_lr=config.q_lr,
        gamma=config.gamma,
        pol_hidden_dim=config.pol_hidden_dim,
        critic_hidden_dim=config.critic_hidden_dim,
        attend_heads=config.attend_heads,
        reward_scale=config.reward_scale)
    replay_buffer = ReplayBuffer(config.buffer_length, model.nagents, os_shape, as_shape)
    t = 0
    for ep_i in range(0, config.n_episodes):
        print("Episode %i of %i" % (ep_i + 1, config.n_episodes))
        obs = env.reset()
        obs = np.array(obs)
        model.prep_rollouts(device='cpu')

        for et_i in range(config.episode_length):
            torch_obs = Variable(torch.Tensor(obs), requires_grad=False).unsqueeze(0)

            # get actions as torch Variables
            torch_actions = model.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            actions = torch_actions[0].data.numpy()
            next_obs, rewards, dones = env.step(actions)
            next_obs = np.array(next_obs)
            replay_buffer.push(obs, actions, rewards, next_obs, dones)
            obs = next_obs
            t += 1
            if (len(replay_buffer) >= config.batch_size and (t % config.steps_per_update) == 0):
                if config.use_gpu:
                    model.prep_training(device='gpu')
                else:
                    model.prep_training(device='cpu')
                for u_i in range(config.num_updates):
                    sample = replay_buffer.sample(config.batch_size,
                                                  to_gpu=config.use_gpu)
                    model.update_critic(sample, logger=logger)
                    model.update_policies(sample, logger=logger)
                    model.update_all_targets()
                model.prep_rollouts(device='cpu')
        ep_rews = replay_buffer.get_average_rewards(config.episode_length)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i,
                              a_ep_rew * config.episode_length, ep_i)

        if ep_i % config.save_interval == 0:
            model.prep_rollouts(device='cpu')
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            model.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            model.save(run_dir / 'model.pt')

    model.save(run_dir / 'model.pt')
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--map",               default='empty_4x4.jpg', type=str)
    parser.add_argument("--no_agents",         default=3,               type=int)
    parser.add_argument("--buffer_length",     default=int(1e5),        type=int)
    parser.add_argument("--n_episodes",        default=5000,            type=int)
    parser.add_argument("--episode_length",    default=50,              type=int)
    parser.add_argument("--steps_per_update",  default=10,              type=int)
    parser.add_argument("--num_updates",       default=3,               type=int)
    parser.add_argument("--batch_size",        default=1024,            type=int)
    parser.add_argument("--save_interval",     default=1000,            type=int)
    parser.add_argument("--pol_hidden_dim",    default=128,             type=int)
    parser.add_argument("--critic_hidden_dim", default=128,             type=int)
    parser.add_argument("--attend_heads",      default=1,               type=int)
    parser.add_argument("--pi_lr",             default=0.001,           type=float)
    parser.add_argument("--q_lr",              default=0.001,           type=float)
    parser.add_argument("--tau",               default=0.001,           type=float)
    parser.add_argument("--gamma",             default=0.99,            type=float)
    parser.add_argument("--reward_scale",      default=100.,            type=float)
    parser.add_argument("--use_gpu",           action='store_true')

    config = parser.parse_args()

    run(config)
