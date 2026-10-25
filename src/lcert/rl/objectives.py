from __future__ import annotations

import math

import torch
import torch.nn.functional as functional


def _finite_same_shape(*tensors: torch.Tensor) -> None:
    if not tensors or tensors[0].numel() == 0:
        raise ValueError('Objective inputs cannot be empty')
    shape = tensors[0].shape
    for tensor in tensors:
        if tensor.shape != shape or not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise ValueError('Objective tensors must have matching shapes and finite floating values')


def _positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')


def reinforce_loss(log_probabilities: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    _finite_same_shape(log_probabilities, returns)
    return -(log_probabilities * returns.detach()).mean()


def a2c_loss(log_probabilities: torch.Tensor, advantages: torch.Tensor,
             entropy: torch.Tensor, entropy_weight: float = 0.01) -> torch.Tensor:
    _finite_same_shape(log_probabilities, advantages, entropy)
    if not math.isfinite(entropy_weight) or entropy_weight < 0:
        raise ValueError('Entropy weight must be nonnegative and finite')
    return -(log_probabilities * advantages.detach() + entropy_weight * entropy).mean()


def ppo_loss(log_probabilities: torch.Tensor, old_log_probabilities: torch.Tensor,
             advantages: torch.Tensor, clip_epsilon: float = 0.2) -> torch.Tensor:
    _finite_same_shape(log_probabilities, old_log_probabilities, advantages)
    _positive(clip_epsilon, 'clip_epsilon')
    if clip_epsilon >= 1:
        raise ValueError('clip_epsilon must be below one')
    ratios = (log_probabilities - old_log_probabilities.detach()).exp()
    unclipped = ratios * advantages.detach()
    clipped = ratios.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantages.detach()
    loss = -torch.minimum(unclipped, clipped).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError('PPO likelihood ratios overflowed')
    return loss


def dpo_loss(chosen_log_probabilities: torch.Tensor, rejected_log_probabilities: torch.Tensor,
             reference_chosen: torch.Tensor, reference_rejected: torch.Tensor,
             beta: float = 0.1) -> torch.Tensor:
    """DPO-style trajectory objective; callers must enforce paired identical contexts."""
    _finite_same_shape(chosen_log_probabilities, rejected_log_probabilities,
                       reference_chosen, reference_rejected)
    _positive(beta, 'beta')
    chosen_ratio = chosen_log_probabilities - reference_chosen.detach()
    rejected_ratio = rejected_log_probabilities - reference_rejected.detach()
    return functional.softplus(-beta * (chosen_ratio - rejected_ratio)).mean()


def group_relative_advantages(rewards: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    if rewards.ndim != 2 or rewards.shape[0] == 0 or rewards.shape[1] < 2:
        raise ValueError('Grouped rewards must have shape [contexts, at_least_two_rollouts]')
    if not rewards.is_floating_point() or not torch.isfinite(rewards).all():
        raise ValueError('Grouped rewards must be finite floating values')
    _positive(epsilon, 'epsilon')
    detached = rewards.detach()
    centered = detached - detached.mean(dim=1, keepdim=True)
    deviations = detached.std(dim=1, keepdim=True, correction=0)
    return torch.where(deviations > 0, centered / deviations.clamp_min(epsilon), torch.zeros_like(centered))


def grpo_loss(log_probabilities: torch.Tensor, old_log_probabilities: torch.Tensor,
              reference_log_probabilities: torch.Tensor, rewards: torch.Tensor,
              action_mask: torch.Tensor, clip_epsilon: float = 0.2,
              kl_weight: float = 0.01) -> torch.Tensor:
    """Grouped robot-action adaptation, not an implemented language-model GRPO trainer."""
    _finite_same_shape(log_probabilities, old_log_probabilities, reference_log_probabilities)
    if log_probabilities.ndim != 3 or action_mask.shape != log_probabilities.shape:
        raise ValueError('Log-probabilities and masks must have shape [contexts, rollouts, steps]')
    if rewards.shape != log_probabilities.shape[:2]:
        raise ValueError('Rewards must match grouped contexts and rollouts')
    if not torch.all((action_mask == 0) | (action_mask == 1)):
        raise ValueError('Action mask must contain only zeros and ones')
    lengths = action_mask.sum(dim=-1)
    if (lengths == 0).any():
        raise ValueError('A rollout must contain at least one valid action')
    _positive(clip_epsilon, 'clip_epsilon')
    if clip_epsilon >= 1 or not math.isfinite(kl_weight) or kl_weight < 0:
        raise ValueError('Invalid GRPO clipping or KL weight')
    advantages = group_relative_advantages(rewards).unsqueeze(-1)
    ratios = (log_probabilities - old_log_probabilities.detach()).exp()
    surrogate = torch.minimum(ratios * advantages,
                              ratios.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantages)
    reference_difference = reference_log_probabilities.detach() - log_probabilities
    kl_estimate = reference_difference.exp() - reference_difference - 1
    masked = (surrogate - kl_weight * kl_estimate) * action_mask
    loss = -(masked.sum(dim=-1) / lengths).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError('GRPO likelihood ratios or KL estimator overflowed')
    return loss


def expectile_loss(predictions: torch.Tensor, targets: torch.Tensor,
                   expectile: float = 0.7) -> torch.Tensor:
    _finite_same_shape(predictions, targets)
    if not math.isfinite(expectile) or not 0 < expectile < 1:
        raise ValueError('Expectile must be strictly between zero and one')
    residual = targets.detach() - predictions
    weights = torch.where(residual > 0, expectile, 1 - expectile)
    return (weights * residual.square()).mean()


def behavior_cloning_loss(logits: torch.Tensor, observed_actions: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[0] == 0 or observed_actions.shape != logits.shape[:1]:
        raise ValueError('Expected logits [samples, actions] and observed actions [samples]')
    if not torch.isfinite(logits).all() or observed_actions.dtype != torch.long:
        raise ValueError('Logits must be finite and observed actions integer torch.long')
    return functional.cross_entropy(logits, observed_actions)
