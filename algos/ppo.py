import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional


import envpool
import numpy as np
import tyro

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast

from dirtyrl.utils import RecordEpisodeStatistics, to_tensor
from dirtyrl.dist import reduce_gradidents, setup, fprint, mp_start
from dirtyrl.rl import bootstrap_value, train_step as train_step_


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    # Algorithm specific arguments
    env_id: str = "Breakout-v5"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    local_num_envs: int = 8
    """the number of parallel game environments in the local rank"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    minibatch_size: int = 256
    """the mini-batch size"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""

    compile: Optional[str] = None
    """Compile mode of torch.compile, None for no compilation"""
    local_torch_threads: Optional[int] = None
    """the number of threads to use for torch in the local rank, defaults to ($OMP_NUM_THREADS or 2)"""
    local_env_threads: Optional[int] = None
    """the number of threads to use for envpool in the local rank, defaults to `local_num_envs`"""

    fp16_train: bool = False
    """if toggled, training will be done in fp16 precision"""
    fp16_eval: bool = False
    """if toggled, evaluation will be done in fp16 precision"""

    tb_dir: str = "./runs"
    """tensorboard log directory"""
    ckpt_dir: str = "./checkpoints"
    """checkpoint directory"""
    save_interval: int = 500
    """the number of iterations to save the model"""
    log_p: float = 1.0
    """the probability of logging"""
    port: int = 12355
    """the port to use for distributed training"""

    # to be filled in runtime
    num_envs: int = 0
    """the number of parallel game environments (computed in runtime)"""
    local_batch_size: int = 0
    """the local batch size in the local rank (computed in runtime)"""
    local_minibatch_size: int = 0
    """the local mini-batch size in the local rank (computed in runtime)"""
    num_minibatches: int = 4
    """the number of mini-batches (computed in runtime)"""
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    world_size: int = 0
    """the number of processes (computed in runtime)"""


def make_env(args, num_envs, num_threads):
    envs = envpool.make(
        args.env_id,
        env_type="gymnasium",
        num_envs=num_envs,
        num_threads=num_threads,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
    )
    envs.num_envs = args.local_num_envs
    envs = RecordEpisodeStatistics(envs, has_lives=True)
    return envs

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def forward(self, x):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        return logits, self.critic(hidden)


def run(local_rank, world_size):
    rank = local_rank
    args = tyro.cli(Args)
    args.world_size = world_size
    args.num_envs = args.local_num_envs * args.world_size
    args.local_batch_size = int(args.local_num_envs * args.num_steps)
    args.local_minibatch_size = int(args.minibatch_size // args.world_size)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.num_iterations = args.total_timesteps // args.batch_size
    args.num_minibatches = args.local_batch_size // args.local_minibatch_size
    args.local_env_threads = args.local_env_threads or args.local_num_envs
    args.local_torch_threads = args.local_torch_threads or int(os.getenv("OMP_NUM_THREADS", "2"))

    torch.set_num_threads(args.local_torch_threads)
    torch.set_float32_matmul_precision('high')

    if args.world_size > 1:
        setup('nccl', local_rank, args.world_size, args.port)

    timestamp = int(time.time())
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{timestamp}"
    writer = None
    if rank == 0:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(args.tb_dir, run_name))
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        ckpt_dir = os.path.join(args.ckpt_dir, run_name)
        os.makedirs(ckpt_dir, exist_ok=True)


    # TRY NOT TO MODIFY: seeding
    # CRUCIAL: note that we needed to pass a different seed for each data parallelism worker
    args.seed += rank
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed - rank)
    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = make_env(args, args.local_num_envs, args.local_env_threads)
    n_actions = envs.action_space.n
    action_shape = envs.action_space.shape
    observation_shape = envs.observation_space.shape
    if local_rank == 0:
        fprint(f"obs_shape={observation_shape}, action_shape={action_shape}, n_actions={n_actions}")

    agent = Agent(n_actions).to(device)
    torch.manual_seed(args.seed)
    optim_params = list(agent.parameters())
    optimizer = optim.Adam(optim_params, lr=args.learning_rate, eps=1e-5)

    scaler = GradScaler(enabled=args.fp16_train, init_scale=2 ** 8)

    def predict_step(agent: Agent, next_obs):
        with torch.no_grad():
            with autocast(enabled=args.fp16_eval):
                logits, value = agent(next_obs)
        return logits, value

    if args.compile:
        # It seems that using torch.compile twice cause segfault at start, so we use torch.jit.trace here
        # predict_step = torch.compile(predict_step, mode=args.compile)
        example_obs = torch.zeros((1,) + observation_shape, device=device)
        with torch.no_grad():
            agent_r = torch.jit.trace(agent, (example_obs,), check_tolerance=False, check_trace=False)

        train_step = torch.compile(train_step_, mode=args.compile)
    else:
        agent_r = agent
        train_step = train_step_

    obs = torch.zeros((args.num_steps, args.local_num_envs) + observation_shape, device=device)
    actions = torch.zeros((args.num_steps, args.local_num_envs) + action_shape, dtype=torch.long, device=device)
    logprobs = torch.zeros((args.num_steps, args.local_num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.local_num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.local_num_envs), device=device)
    values = torch.zeros((args.num_steps, args.local_num_envs), device=device)
    avg_returns = deque(maxlen=20)

    global_step = 0
    warmup_steps = 0
    start_time = time.time()
    next_obs, info = envs.reset()
    next_obs = to_tensor(next_obs, device)
    next_done = torch.zeros(args.local_num_envs, device=device, dtype=torch.float32)

    for iteration in range(args.num_iterations):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - iteration / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        model_time = 0
        env_time = 0
        collect_start = time.time()
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            _start = time.time()
            logits, value = predict_step(agent_r, next_obs)
            value = value.flatten()
            probs = Categorical(logits=logits)
            action = probs.sample()
            logprob = probs.log_prob(action)

            values[step] = value
            actions[step] = action
            logprobs[step] = logprob
            action = action.cpu().numpy()
            model_time += time.time() - _start

            _start = time.time()
            next_obs, reward, next_done_, info = envs.step(action)
            env_time += time.time() - _start
            rewards[step] = to_tensor(reward, device)
            next_obs, next_done = to_tensor(next_obs, device), to_tensor(next_done_, device, dtype=torch.float32)

            if not writer:
                continue

            for idx, d in enumerate(next_done_):
                if d and info["lives"][idx] == 0:
                    episode_length = info['l'][idx]
                    episode_reward = info['r'][idx]
                    avg_returns.append(episode_reward)

                    if random.random() < args.log_p:
                        n = 100
                        if random.random() < 10/n or iteration <= 1:
                            writer.add_scalar("charts/episodic_return", info["r"][idx], global_step)
                            writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)
                            fprint(f"global_step={global_step}, e_ret={episode_reward}, e_len={episode_length}")

                        if random.random() < 10/n:
                            writer.add_scalar("charts/avg_ep_return", np.mean(avg_returns), global_step)

        collect_time = time.time() - collect_start
        if local_rank == 0:
            fprint(f"collect_time={collect_time:.4f}, model_time={model_time:.4f}, env_time={env_time:.4f}")

        _start = time.time()
        # bootstrap value if not done
        nextvalues = predict_step(agent_r, next_obs)[1].reshape(-1)
        advantages = bootstrap_value(
            values, rewards, dones, nextvalues, next_done, args.gamma, args.gae_lambda)
        bootstrap_time = time.time() - _start

        _start = time.time()
        # flatten the batch
        b_obs = obs.reshape((-1,) + obs.shape[2:])
        b_actions = actions.reshape((-1,) + action_shape)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_values = values.reshape(-1)
        b_returns = b_advantages + b_values

        # Optimizing the policy and value network
        b_inds = np.arange(args.local_batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.local_batch_size, args.local_minibatch_size):
                end = start + args.local_minibatch_size
                mb_inds = b_inds[start:end]

                old_approx_kl, approx_kl, clipfrac, pg_loss, v_loss, entropy_loss = \
                    train_step(agent, optimizer, scaler, b_obs[mb_inds], b_actions[mb_inds], b_logprobs[mb_inds], b_advantages[mb_inds],
                            b_returns[mb_inds], b_values[mb_inds], args)
                reduce_gradidents(optim_params, args.world_size)
                nn.utils.clip_grad_norm_(optim_params, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                clipfracs.append(clipfrac.item())

        train_time = time.time() - _start

        if local_rank == 0:
            fprint(f"train_time={train_time:.4f}, collect_time={collect_time:.4f}, bootstrap_time={bootstrap_time:.4f}")

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if rank == 0:
            if iteration % args.save_interval == 0:
                torch.save(agent.state_dict(), os.path.join(ckpt_dir, f"agent.pt"))

            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)

        SPS = int((global_step - warmup_steps) / (time.time() - start_time))

        # Warmup at first few iterations for accurate SPS measurement
        SPS_warmup_iters = 10
        if iteration == SPS_warmup_iters:
            start_time = time.time()
            warmup_steps = global_step
        if iteration > SPS_warmup_iters:
            if local_rank == 0:
                fprint(f"SPS: {SPS}")
            if rank == 0:
                writer.add_scalar("charts/SPS", SPS, global_step)

    if args.world_size > 1:
        dist.destroy_process_group()
    envs.close()
    if rank == 0:
        torch.save(agent.state_dict(), os.path.join(ckpt_dir, f"agent_final.pt"))
        writer.close()


if __name__ == "__main__":
    mp_start(run)
