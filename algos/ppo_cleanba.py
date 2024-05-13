import os
import random
import time
from collections import deque
from queue import Queue
from dataclasses import dataclass, field
from typing import Optional, List


import envpool
import numpy as np
import tyro

import optree
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.cuda.amp import GradScaler, autocast
import torch.multiprocessing as mp

from dirtyrl.utils import RecordEpisodeStatistics, to_tensor
from dirtyrl.dist import reduce_gradidents, setup, fprint, SummaryWriterProxy, MpSummaryWriterManager
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
    """the number of parallel game environments per actor"""
    num_actor_threads: int = 2
    "the number of actor threads to use"
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    gae_length: Optional[int] = None
    """the length of the trajectory to calculate the GAE (if None, defaults to `num_steps`)"""
    anneal_lr: bool = False
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

    actor_device_ids: List[int] = field(default_factory=lambda: [0])
    "the device ids that actor workers will use"
    learner_device_ids: List[int] = field(default_factory=lambda: [0])
    "the device ids that learner workers will use"

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
    num_minibatches: int = 0
    """the number of mini-batches (computed in runtime)"""
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    world_size: int = 0
    """the number of processes (computed in runtime)"""
    group_mode: int = 0
    """the group mode for the learner and actor (computed in runtime)"""


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


def actor(
    args,
    a_rank,
    rollout_queues: List[Queue],
    param_queue: Queue,
    learner_devices,
    tb_queue,
    device_thread_id,
):
    assert len(rollout_queues) == len(learner_devices)
    if a_rank == 0:
        writer = SummaryWriterProxy(tb_queue)
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    else:
        writer = None
    torch.set_num_threads(args.local_torch_threads)
    torch.set_float32_matmul_precision('high')

    device = torch.device(f"cuda:{device_thread_id}" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = make_env(args, args.local_num_envs, args.local_env_threads)
    n_actions = envs.action_space.n
    action_shape = envs.action_space.shape
    observation_shape = envs.observation_space.shape
    if a_rank == 0:
        fprint(f"obs_shape={observation_shape}, action_shape={action_shape}, n_actions={n_actions}")

    agent = Agent(n_actions).to(device)

    def predict_step(agent: Agent, next_obs):
        with torch.no_grad():
            with autocast(enabled=args.fp16_eval):
                logits, value = agent(next_obs)
        return logits, value

    if args.compile:
        predict_step = torch.compile(predict_step, mode=args.compile)

    obs = torch.zeros((args.gae_length, args.local_num_envs) + observation_shape, device=device)
    actions = torch.zeros((args.gae_length, args.local_num_envs) + action_shape, dtype=torch.long, device=device)
    logprobs = torch.zeros((args.gae_length, args.local_num_envs), device=device)
    rewards = torch.zeros((args.gae_length, args.local_num_envs), device=device)
    dones = torch.zeros((args.gae_length, args.local_num_envs), device=device)
    values = torch.zeros((args.gae_length, args.local_num_envs), device=device)
    avg_returns = deque(maxlen=20)

    global_step = 0
    warmup_steps = 0
    start_time = time.time()
    next_obs, info = envs.reset()
    next_obs = to_tensor(next_obs, device)
    next_done = torch.zeros(args.local_num_envs, device=device, dtype=torch.float32)
    step = 0
    params_buffer = param_queue.get()[1]

    next_done_t = next_done.clone()
    next_obs_t = next_obs.clone()

    for iteration in range(1, args.num_iterations + 2):
        if iteration > 2:
            param_queue.get()
        agent.load_state_dict(params_buffer)

        model_time = 0
        env_time = 0
        collect_start = time.time()
        while step < args.gae_length:
            obs[step] = next_obs
            dones[step] = next_done

            _start = time.time()
            logits, value = predict_step(agent, next_obs)
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
            step += 1

            global_step += args.num_envs

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
        if a_rank == 0:
            fprint(f"collect_time={collect_time:.4f}, model_time={model_time:.4f}, env_time={env_time:.4f}")

        _start = time.time()

        next_done_t[:] = next_done
        next_obs_t[:] = next_obs

        step = args.gae_length - args.num_steps

        def prepare_data(xs, device):
            return optree.tree_map(lambda x: x.to(device=device, non_blocking=True, copy=True), xs)

        for iq, (rq, device_id) in enumerate(zip(rollout_queues, learner_devices)):
            n_e = args.local_num_envs // len(rollout_queues)
            start = iq * n_e
            end = start + n_e
            data = [
                *optree.tree_map(lambda x: x[:, start:end],
                    (obs, actions, logprobs, rewards, dones)),
                *optree.tree_map(lambda x: x[start:end],
                    (next_done_t, next_obs_t)),
            ]
            data = [global_step, *prepare_data(data, device_id)]
            rq.put(data)

        if step > 0:
            # TODO: use cyclic buffer to avoid copying
            for v in [obs, actions, logprobs, rewards, dones, values]:
                v[:step] = v[args.num_steps:].clone()

        SPS = int((global_step - warmup_steps) / (time.time() - start_time))
        if a_rank == 0:
            fprint(f"SPS: {SPS}")

        # Warmup at first few iterations for accurate SPS measurement
        SPS_warmup_iters = 10
        if iteration == SPS_warmup_iters:
            start_time = time.time()
            warmup_steps = global_step
        if iteration > SPS_warmup_iters:
            if a_rank == 0:
                writer.add_scalar("charts/SPS", SPS, global_step)

def learner(
    args: Args,
    l_rank,
    rollout_queues: List[Queue],
    param_queues: List[Queue],
    tb_queue,
    ckpt_dir,
    device_thread_id,    
):
    num_learners = len(args.learner_device_ids)
    if len(args.learner_device_ids) > 1:
        setup('nccl', l_rank, num_learners, args.port)
    local_batch_size = args.local_batch_size // num_learners
    local_minibatch_size = args.local_minibatch_size // num_learners

    torch.set_num_threads(args.local_torch_threads)
    torch.set_float32_matmul_precision('high')

    writer = SummaryWriterProxy(tb_queue) if l_rank == 0 else None

    args.seed += l_rank
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed - l_rank)

    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True

    envs = make_env(args, 1, 1)
    n_actions = envs.action_space.n

    device = torch.device(f"cuda:{device_thread_id}" if torch.cuda.is_available() and args.cuda else "cpu")
    agent = Agent(n_actions).to(device)
    torch.manual_seed(args.seed)

    example_obs = torch.zeros((1,) + envs.observation_space.shape, device=device)
    with torch.no_grad():
        agent_r = torch.jit.trace(agent, (example_obs,), check_tolerance=False, check_trace=False)

    if args.compile:
        train_step = torch.compile(train_step_, mode=args.compile)
    else:
        train_step = train_step_

    optim_params = list(agent.parameters())
    optimizer = optim.Adam(optim_params, lr=args.learning_rate, eps=1e-5)

    scaler = GradScaler(enabled=args.fp16_train, init_scale=2 ** 8)

    if args.group_mode == 1:
        first_in_group = l_rank % (num_learners // (len(args.actor_device_ids) * args.num_actor_threads)) == 0
        if first_in_group:
            param_queues[0].put(("Init", agent.state_dict()))
    else:
        for pq in param_queues:
            pq.put(("Init", agent.state_dict()))
    
    learner_policy_version = 0
    while True:
        learner_policy_version += 1

        if args.anneal_lr:
            frac = 1.0 - (learner_policy_version - 1) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        bootstrap_start = time.time()

        _start = time.time()
        
        if args.group_mode == 1:
            global_step, obs, actions, logprobs, rewards, dones, next_done, next_obs = \
                rollout_queues[0].get()
        else:
            data_list1 = []
            data_list2 = []
            for rq in rollout_queues:
                global_step, *data = rq.get()
                data_list1.append(data[:5])
                data_list2.append(data[5:])
            obs, actions, logprobs, rewards, dones = \
                optree.tree_map(lambda *x: torch.cat(x, dim=1), *data_list1)
            next_done, next_obs = optree.tree_map(
                lambda *x: torch.cat(x, dim=0), *data_list2)
        wait_time = time.time() - _start

        with torch.no_grad():
            with autocast(enabled=args.fp16_eval):
                nextvalues = agent_r(next_obs)[1].reshape(-1)

        values = torch.zeros_like(rewards)
        n_mbs = args.num_minibatches // 4
        for start in range(0, args.gae_length, args.gae_length // n_mbs):
            end = start + args.gae_length // n_mbs
            v_obs = obs[start:end].flatten(0, 1)
            with torch.no_grad():
                with autocast(enabled=args.fp16_eval):
                    value = agent_r(v_obs)[1].reshape(end - start, -1)
            values[start:end] = value

        advantages = bootstrap_value(
            values, rewards, dones, nextvalues, next_done, args.gamma, args.gae_lambda)
        bootstrap_time = time.time() - bootstrap_start

        _start = time.time()

        # flatten the batch
        b_obs = obs[:args.num_steps].flatten(0, 1)
        b_actions = actions[:args.num_steps].flatten(0, 1)
        b_logprobs = logprobs[:args.num_steps].flatten(0, 1)
        b_advantages = advantages[:args.num_steps].reshape(-1)
        b_values = values[:args.num_steps].reshape(-1)
        b_returns = b_advantages + b_values

        # Optimizing the policy and value network
        b_inds = np.arange(local_batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, local_batch_size, local_minibatch_size):
                end = start + local_minibatch_size
                mb_inds = b_inds[start:end]

                old_approx_kl, approx_kl, clipfrac, pg_loss, v_loss, entropy_loss = \
                    train_step(agent, optimizer, scaler, b_obs[mb_inds], b_actions[mb_inds], b_logprobs[mb_inds], b_advantages[mb_inds],
                            b_returns[mb_inds], b_values[mb_inds], args)
                reduce_gradidents(optim_params, num_learners)
                nn.utils.clip_grad_norm_(optim_params, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                clipfracs.append(clipfrac.item())

        if args.group_mode == 1:
            if first_in_group:
                param_queues[0].put(("Done", None))
        else:
            for pq in param_queues:
                pq.put(("Done", None))

        if l_rank == 0:
            train_time = time.time() - _start
            fprint(f"train_time={train_time:.4f}, bootstrap_time={bootstrap_time:.4f}, wait_time={wait_time:.4f}")

            if learner_policy_version % args.save_interval == 0:
                torch.save(agent.state_dict(), os.path.join(ckpt_dir, f"agent.pt"))

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            fprint(f"global_step={global_step}, value_loss={v_loss.item():.4f}, policy_loss={pg_loss.item():.4f}, entropy_loss={entropy_loss.item():.4f}")

            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)

        if learner_policy_version >= args.num_iterations:
            break


if __name__ == "__main__":
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    args = tyro.cli(Args)
    args.world_size = 1
    args.gae_length = args.gae_length or args.num_steps
    assert args.gae_length >= args.num_steps, "gae_length must be greater than or equal to num_steps"
    args.local_batch_size = int(args.local_num_envs * args.num_steps * args.num_actor_threads * len(args.actor_device_ids))
    args.local_minibatch_size = args.minibatch_size // args.world_size
    args.num_minibatches = args.local_batch_size // args.local_minibatch_size
    assert (
        args.local_num_envs % len(args.learner_device_ids) == 0
    ), "local_num_envs must be divisible by len(learner_device_ids)"
    assert (
        int(args.local_num_envs / len(args.learner_device_ids)) * args.num_actor_threads % args.num_minibatches == 0
    ), "int(local_num_envs / len(learner_device_ids)) must be divisible by num_minibatches"

    args.num_envs = args.local_num_envs * args.world_size * args.num_actor_threads * len(args.actor_device_ids)
    args.batch_size = args.local_batch_size * args.world_size
    args.num_iterations = args.total_timesteps // args.batch_size
    args.local_env_threads = args.local_env_threads or args.local_num_envs
    args.env_threads = args.local_env_threads * args.world_size * args.num_actor_threads * len(args.actor_device_ids)
    args.local_torch_threads = args.local_torch_threads or int(os.getenv("OMP_NUM_THREADS", "2"))

    timestamp = int(time.time())
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{timestamp}"

    tb_manager = MpSummaryWriterManager(
        os.path.join(args.tb_dir, run_name)
    )
    tb_queue = tb_manager.get_queue()

    ckpt_dir = os.path.join(args.ckpt_dir, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    rollout_queues = []
    param_queues = []

    actor_processes = []
    learner_processes = []

    num_actors = len(args.actor_device_ids) * args.num_actor_threads
    num_learners = len(args.learner_device_ids)
    if num_learners >= num_actors:
        args.group_mode = 1
        assert num_learners % num_actors == 0, "num_learners must be divisible by num_actors"
        group_size = num_learners // num_actors
    else:
        args.group_mode = 2
        assert num_actors % num_learners == 0, "num_actors must be divisible by num_learners"
        group_size = num_actors // num_learners


    for i, device_id in enumerate(args.actor_device_ids):
        for j in range(args.num_actor_threads):
            a_rank = i * args.num_actor_threads + j
            param_queues.append(mp.Queue(maxsize=1))
            learner_devices = []
            rollout_queues_ = []
            if args.group_mode == 1:
                for k in range(group_size):
                    rollout_queues_.append(mp.Queue(maxsize=1))
                    learner_devices.append(args.learner_device_ids[a_rank * group_size + k])
            else:
                rollout_queues_.append(mp.Queue(maxsize=1))
                learner_devices.append(args.learner_device_ids[a_rank // group_size])
            rollout_queues.extend(rollout_queues_)
            p = mp.Process(
                target=actor,
                args=(args, a_rank, rollout_queues_, param_queues[-1], learner_devices, tb_queue, device_id),
            )
            actor_processes.append(p)
            p.start()

    for i, device_id in enumerate(args.learner_device_ids):
        if args.group_mode == 1:
            param_queues_ = [param_queues[i // group_size]]
            rollout_queues_ = [rollout_queues[i]]
        else:
            param_queues_ = [
                param_queues[i * group_size + j] for j in range(group_size)
            ]
            rollout_queues_ = [
                rollout_queues[i * group_size + j] for j in range(group_size)
            ]
        p = mp.Process(
            target=learner,
            args=(args, i, rollout_queues_, param_queues_, tb_queue, ckpt_dir, device_id),
        )
        learner_processes.append(p)
        p.start()

    for p in actor_processes + learner_processes:
        p.join()

    tb_manager.close()
