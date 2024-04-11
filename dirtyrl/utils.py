import numpy as np
import gymnasium as gym

import optree
import torch


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, has_lives=False):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None
        self.has_lives = has_lives

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations, infos

    def step(self, action):
        return self.update_stats_and_infos(*super().step(action))

    def update_stats_and_infos(self, *args):
        observations, rewards, terminated, truncated, infos = args
        dones = np.logical_or(terminated, truncated)
        self.episode_returns += infos.get("reward", rewards)
        self.episode_lengths += 1
        # self.returned_episode_returns = np.where(
        #     dones, self.episode_returns, self.returned_episode_returns
        # )
        # self.returned_episode_lengths = np.where(
        #     dones, self.episode_lengths, self.returned_episode_lengths
        # )
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        if self.has_lives:
            all_lives_exhausted = infos["lives"] == 0
            self.episode_returns *= 1 - all_lives_exhausted
            self.episode_lengths *= 1 - all_lives_exhausted
        else:
            self.episode_returns *= 1 - dones
            self.episode_lengths *= 1 - dones
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths

        return (
            observations,
            rewards,
            dones,
            infos,
        )

    def async_reset(self):
        self.env.async_reset()
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
    
    def recv(self):
        return self.update_stats_and_infos(*self.env.recv())

    def send(self, action):
        return self.env.send(action)


def to_tensor(x, device, dtype=None):
    return optree.tree_map(lambda x: torch.from_numpy(x).to(device=device, dtype=dtype, non_blocking=True), x)
