import torch
from torch.cuda.amp import autocast

from dirtyrl.dist import reduce_gradidents


def entropy_from_logits(logits):
    min_real = torch.finfo(logits.dtype).min
    logits = torch.clamp(logits, min=min_real)
    p_log_p = logits * torch.softmax(logits, dim=-1)
    return -p_log_p.sum(-1)


def train_step(agent, optimizer, scaler, mb_obs, mb_actions, mb_logprobs, mb_advantages, mb_returns, mb_values, args):
    with autocast(enabled=args.fp16_train):
        logits, newvalue = agent(mb_obs)
        logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        newlogprob = logits.gather(-1, mb_actions[:, None]).squeeze(-1)
        entropy = entropy_from_logits(logits)
    logratio = newlogprob - mb_logprobs
    ratio = logratio.exp()

    with torch.no_grad():
        # calculate approx_kl http://joschu.net/blog/kl-approx.html
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()

    if args.norm_adv:
        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

    # Policy loss
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2)
    pg_loss = pg_loss.mean()

    # Value loss
    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        v_loss_unclipped = (newvalue - mb_returns) ** 2
        v_clipped = mb_values + torch.clamp(
            newvalue - mb_values,
            -args.clip_coef,
            args.clip_coef,
        )
        v_loss_clipped = (v_clipped - mb_returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max
    else:
        v_loss = 0.5 * ((newvalue - mb_returns) ** 2)
    v_loss = v_loss.mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
    optimizer.zero_grad()
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    return old_approx_kl, approx_kl, clipfrac, pg_loss, v_loss, entropy_loss


def bootstrap_value(values, rewards, dones, nextvalues, next_done, gamma, gae_lambda):
    num_steps = rewards.size(0)
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = nextvalues
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    return advantages