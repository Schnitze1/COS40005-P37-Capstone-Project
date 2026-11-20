import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Optimizer

from ...gpu import Device
from ...models.builders import create_conditional_vae, create_parameter
from ...models.encoders import EncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.q_functions import QFunctionFactory
from ...models.torch import (
    ConditionalVAE,
    Parameter,
    compute_max_with_n_actions_and_indices,
)
from ...preprocessing import ActionScaler, RewardScaler, Scaler
from ...torch_utility import TorchMiniBatch, torch_api, train_api
from .sac_impl import SACImpl


def _gaussian_kernel(
    x: torch.Tensor, y: torch.Tensor, sigma: float
) -> torch.Tensor:
    # x: (batch, n, 1, action), y: (batch, 1, n, action) -> (batch, n, n)
    return (-((x - y) ** 2).sum(dim=3) / (2 * sigma)).exp()


def _laplacian_kernel(
    x: torch.Tensor, y: torch.Tensor, sigma: float
) -> torch.Tensor:
    # x: (batch, n, 1, action), y: (batch, 1, n, action) -> (batch, n, n)
    return (-(x - y).abs().sum(dim=3) / (2 * sigma)).exp()


class BEARImpl(SACImpl):

    _imitator_learning_rate: float
    _alpha_learning_rate: float
    _imitator_optim_factory: OptimizerFactory
    _alpha_optim_factory: OptimizerFactory
    _imitator_encoder_factory: EncoderFactory
    _initial_alpha: float
    _alpha_threshold: float
    _lam: float
    _n_action_samples: int
    _n_target_samples: int
    _n_mmd_action_samples: int
    _mmd_kernel: str
    _mmd_sigma: float
    _vae_kl_weight: float
    _imitator: Optional[ConditionalVAE]
    _imitator_optim: Optional[Optimizer]
    _log_alpha: Optional[Parameter]
    _alpha_optim: Optional[Optimizer]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        imitator_learning_rate: float,
        temp_learning_rate: float,
        alpha_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        imitator_optim_factory: OptimizerFactory,
        temp_optim_factory: OptimizerFactory,
        alpha_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        imitator_encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        tau: float,
        n_critics: int,
        initial_temperature: float,
        initial_alpha: float,
        alpha_threshold: float,
        lam: float,
        n_action_samples: int,
        n_target_samples: int,
        n_mmd_action_samples: int,
        mmd_kernel: str,
        mmd_sigma: float,
        vae_kl_weight: float,
        max_q_target: Optional[float] = None,
        warmup_steps: int = 0,
        q_ramp_steps: int = 0,
        alpha_min: Optional[float] = 0.1,
        alpha_max: Optional[float] = 5.0,
        use_continuous_bc: bool = False,
        lambda_bc_start: float = 0.9,
        lambda_bc_floor: float = 0.5,
        lambda_bc_decay_steps: int = 200000,
        actor_lr_post_warmup: Optional[float] = None,
        ramp_type: str = "linear",
        bc_warmup_weight: float = 1.0,
        use_huber_loss: bool = False,
        huber_delta: float = 1.0,
        detach_mmd_actor: bool = False,
        use_actor_l2_reg: bool = False,
        actor_l2_weight: float = 0.1,
        mmd_actor_weight: float = 1.0,
        use_gpu: Optional[Device] = None,
        scaler: Optional[Scaler] = None,
        action_scaler: Optional[ActionScaler] = None,
        reward_scaler: Optional[RewardScaler] = None,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            temp_learning_rate=temp_learning_rate,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            temp_optim_factory=temp_optim_factory,
            actor_encoder_factory=actor_encoder_factory,
            critic_encoder_factory=critic_encoder_factory,
            q_func_factory=q_func_factory,
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            target_reduction_type="mix",
            initial_temperature=initial_temperature,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
        )
        self._imitator_learning_rate = imitator_learning_rate
        self._alpha_learning_rate = alpha_learning_rate
        self._imitator_optim_factory = imitator_optim_factory
        self._alpha_optim_factory = alpha_optim_factory
        self._imitator_encoder_factory = imitator_encoder_factory
        self._initial_alpha = initial_alpha
        self._alpha_threshold = alpha_threshold
        self._lam = lam
        self._n_action_samples = n_action_samples
        self._n_target_samples = n_target_samples
        self._n_mmd_action_samples = n_mmd_action_samples
        self._mmd_kernel = mmd_kernel
        self._mmd_sigma = mmd_sigma
        self._vae_kl_weight = vae_kl_weight
        self._max_q_target = max_q_target
        self._warmup_steps = warmup_steps
        self._q_ramp_steps = q_ramp_steps
        self._alpha_min = alpha_min
        self._alpha_max = alpha_max
        self._use_continuous_bc = use_continuous_bc
        self._lambda_bc_start = lambda_bc_start
        self._lambda_bc_floor = lambda_bc_floor
        self._lambda_bc_decay_steps = lambda_bc_decay_steps
        self._actor_lr_post_warmup = actor_lr_post_warmup
        self._ramp_type = ramp_type
        self._bc_warmup_weight = bc_warmup_weight
        self._use_huber_loss = use_huber_loss
        self._huber_delta = huber_delta
        self._detach_mmd_actor = detach_mmd_actor
        self._use_actor_l2_reg = use_actor_l2_reg
        self._actor_l2_weight = actor_l2_weight
        self._mmd_actor_weight = mmd_actor_weight
        self._grad_step = 0

        # initialized in build
        self._imitator = None
        self._imitator_optim = None
        self._log_alpha = None
        self._alpha_optim = None

    def build(self) -> None:
        self._build_imitator()
        self._build_alpha()
        super().build()
        self._build_imitator_optim()
        self._build_alpha_optim()

    def _build_imitator(self) -> None:
        self._imitator = create_conditional_vae(
            observation_shape=self._observation_shape,
            action_size=self._action_size,
            latent_size=2 * self._action_size,
            beta=self._vae_kl_weight,
            min_logstd=-4.0,
            max_logstd=15.0,
            encoder_factory=self._imitator_encoder_factory,
        )

    def _build_imitator_optim(self) -> None:
        assert self._imitator is not None
        self._imitator_optim = self._imitator_optim_factory.create(
            self._imitator.parameters(), lr=self._imitator_learning_rate
        )

    def _build_alpha(self) -> None:
        initial_val = math.log(self._initial_alpha)
        self._log_alpha = create_parameter((1, 1), initial_val)

    def _build_alpha_optim(self) -> None:
        assert self._log_alpha is not None
        self._alpha_optim = self._alpha_optim_factory.create(
            self._log_alpha.parameters(), lr=self._alpha_learning_rate
        )

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        # --- Compute Q-loss directly without entropy term ---
        # Use policy action samples (differentiable via internal rsample in policy forward)
        policy_actions_for_q = self._policy(batch.observations)
        q_t = self._q_func(batch.observations, policy_actions_for_q, "min")[0]
        q_loss = -q_t.mean()

        # Progressive beta_q ramping
        if hasattr(self, "_warmup_steps") and self._warmup_steps > 0 and hasattr(self, "_q_ramp_steps") and self._q_ramp_steps > 0:
            if self._grad_step < self._warmup_steps:
                beta_q = 0.0
            elif self._grad_step < (self._warmup_steps + self._q_ramp_steps):
                progress = float(self._grad_step - self._warmup_steps) / float(self._q_ramp_steps)
                if getattr(self, "_ramp_type", "linear") == "cosine":
                    beta_q = 0.5 * (1.0 - math.cos(math.pi * progress))
                else:
                    beta_q = progress
            else:
                beta_q = 1.0
        else:
            beta_q = 1.0

        mmd_loss = self._compute_mmd_loss(batch.observations)

        if getattr(self, "_detach_mmd_actor", False):
            mmd_loss = mmd_loss.detach()

        if getattr(self, "_use_actor_l2_reg", False):
            assert self._imitator is not None
            with torch.no_grad():
                imitator_actions = self._imitator.sample(batch.observations)
            l2_reg = F.mse_loss(policy_actions_for_q, imitator_actions)
            reg_weight = getattr(self, "_actor_l2_weight", 0.1)
            bear_loss = beta_q * q_loss + reg_weight * l2_reg
        else:
            mmd_weight = getattr(self, "_mmd_actor_weight", 1.0)
            bear_loss = beta_q * q_loss + mmd_weight * mmd_loss

        # Continuous BC blending (post-warmup)
        if getattr(self, "_use_continuous_bc", False):
            if self._grad_step < self._warmup_steps:
                lambda_bc = 1.0
            else:
                step_after_warmup = self._grad_step - self._warmup_steps
                grace_steps = 200000
                decay_steps = max(getattr(self, "_lambda_bc_decay_steps", 800000), grace_steps + 1)
                
                if step_after_warmup < grace_steps:
                    lambda_bc = 1.0
                elif step_after_warmup < decay_steps:
                    progress = float(step_after_warmup - grace_steps) / float(decay_steps - grace_steps)
                    lambda_bc = self._lambda_bc_start - progress * (self._lambda_bc_start - self._lambda_bc_floor)
                else:
                    lambda_bc = self._lambda_bc_floor

            assert self._policy is not None
            # Use deterministic mean action for BC loss
            policy_actions_mean = self._policy.best_action(batch.observations)
            bc_loss = F.mse_loss(policy_actions_mean, batch.actions)

            return lambda_bc * bc_loss + (1.0 - lambda_bc) * bear_loss

        return bear_loss

    @train_api
    @torch_api()
    def update_actor(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._actor_optim is not None

        # Post-warmup transition safeguards (step == warmup_steps)
        if self._grad_step == self._warmup_steps:
            # 1. Clear Adam optimizer momentum accumulated during warmup
            self._actor_optim.state.clear()
            print(f"[Safeguard] Post-warmup step {self._grad_step}: Cleared actor optimizer state (Adam momentum).")

            # 2. Check post-warmup LR decay
            if getattr(self, "_actor_lr_post_warmup", None) is not None:
                for param_group in self._actor_optim.param_groups:
                    param_group['lr'] = self._actor_lr_post_warmup
                print(f"[Safeguard] Post-warmup step {self._grad_step}: Stepped down actor LR to {self._actor_lr_post_warmup}")

        self._actor_optim.zero_grad()

        loss = self.compute_actor_loss(batch)

        loss.backward()
        self._actor_optim.step()

        return loss.cpu().detach().numpy()

    @train_api
    @torch_api()
    def warmup_actor(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._actor_optim is not None
        assert self._policy is not None

        self._actor_optim.zero_grad()

        # Behavior Cloning loss to pre-train actor on offline data distribution
        policy_actions = self._policy.best_action(batch.observations)
        bc_loss = F.mse_loss(policy_actions, batch.actions)
        mmd_loss = self._compute_mmd_loss(batch.observations)

        loss = self._bc_warmup_weight * bc_loss + 0.1 * mmd_loss

        loss.backward()
        self._actor_optim.step()

        return loss.cpu().detach().numpy()

    def _compute_mmd_loss(self, obs_t: torch.Tensor) -> torch.Tensor:
        assert self._log_alpha
        mmd = self._compute_mmd(obs_t)
        alpha = self._log_alpha().exp()
        if getattr(self, "_alpha_min", None) is not None:
            alpha = torch.clamp(alpha, min=self._alpha_min)
        if getattr(self, "_alpha_max", None) is not None:
            alpha = torch.clamp(alpha, max=self._alpha_max)
        return (alpha * (mmd - self._alpha_threshold)).mean()

    @train_api
    @torch_api()
    def update_imitator(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._imitator_optim is not None

        self._imitator_optim.zero_grad()

        loss = self.compute_imitator_loss(batch)

        loss.backward()

        self._imitator_optim.step()

        return loss.cpu().detach().numpy()

    def compute_imitator_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._imitator is not None
        return self._imitator.compute_error(batch.observations, batch.actions)

    @train_api
    @torch_api()
    def update_alpha(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._alpha_optim is not None
        assert self._log_alpha is not None

        loss = -self._compute_mmd_loss(batch.observations)

        self._alpha_optim.zero_grad()
        loss.backward()
        self._alpha_optim.step()

        # Clamp log_alpha parameter to enforce alpha_min and alpha_max
        if self._alpha_min is not None:
            self._log_alpha.data.clamp_(min=math.log(self._alpha_min))
        if self._alpha_max is not None:
            self._log_alpha.data.clamp_(max=math.log(self._alpha_max))

        cur_alpha = self._log_alpha().exp()
        if getattr(self, "_alpha_min", None) is not None:
            cur_alpha = torch.clamp(cur_alpha, min=self._alpha_min)
        if getattr(self, "_alpha_max", None) is not None:
            cur_alpha = torch.clamp(cur_alpha, max=self._alpha_max)
        cur_alpha_val = cur_alpha.cpu().detach().numpy()[0][0]

        return loss.cpu().detach().numpy(), cur_alpha_val

    def _compute_mmd(self, x: torch.Tensor) -> torch.Tensor:
        assert self._imitator is not None
        assert self._policy is not None
        with torch.no_grad():
            behavior_actions = self._imitator.sample_n_without_squash(
                x, self._n_mmd_action_samples
            )
        policy_actions = self._policy.sample_n_without_squash(
            x, self._n_mmd_action_samples
        )

        if self._mmd_kernel == "gaussian":
            kernel = _gaussian_kernel
        elif self._mmd_kernel == "laplacian":
            kernel = _laplacian_kernel
        elif self._mmd_kernel == "median":
            # Dynamic median heuristic for MMD Gaussian kernel bandwidth
            diff = policy_actions.unsqueeze(2) - behavior_actions.unsqueeze(1)
            dist_sq = diff.pow(2).sum(dim=-1)
            med_sigma = dist_sq.median().sqrt().clamp(min=1e-3).detach().item()
            self._mmd_sigma = med_sigma
            kernel = _gaussian_kernel
        elif self._mmd_kernel == "kl":
            behavior_actions = self._imitator.sample_n_without_squash(
                x, self._n_mmd_action_samples
            )
            policy_actions = self._policy.sample_n_without_squash(
                x, self._n_mmd_action_samples
            )
            # Sample-based KL divergence surrogate (mean squared L2 distance between policy & CVAE)
            kl = (policy_actions - behavior_actions).pow(2).sum(dim=-1).mean(dim=1)
            return kl
        else:
            raise ValueError(f"Invalid kernel type: {self._mmd_kernel}")

        # (batch, n, action) -> (batch, n, 1, action)
        behavior_actions = behavior_actions.reshape(
            x.shape[0], -1, 1, self._action_size
        )
        policy_actions = policy_actions.reshape(
            x.shape[0], -1, 1, self._action_size
        )
        # (batch, n, action) -> (batch, 1, n, action)
        behavior_actions_T = behavior_actions.reshape(
            x.shape[0], 1, -1, self._action_size
        )
        policy_actions_T = policy_actions.reshape(
            x.shape[0], 1, -1, self._action_size
        )

        # 1 / N^2 \sum k(a_\pi, a_\pi)
        inter_policy = kernel(policy_actions, policy_actions_T, self._mmd_sigma)
        mmd = inter_policy.mean(dim=[1, 2])

        # 1 / N^2 \sum k(a_\beta, a_\beta)
        inter_data = kernel(
            behavior_actions, behavior_actions_T, self._mmd_sigma
        )
        mmd += inter_data.mean(dim=[1, 2])

        # 2 / N^2 \sum k(a_\pi, a_\beta)
        distance = kernel(policy_actions, behavior_actions_T, self._mmd_sigma)
        mmd -= 2 * distance.mean(dim=[1, 2])

        return (mmd + 1e-6).sqrt().view(-1, 1)

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._policy is not None
        assert self._targ_q_func is not None
        assert self._log_temp is not None
        with torch.no_grad():
            # BCQ-like target computation
            actions, log_probs = self._policy.sample_n_with_log_prob(
                batch.next_observations,
                self._n_target_samples,
            )
            values, indices = compute_max_with_n_actions_and_indices(
                batch.next_observations, actions, self._targ_q_func, self._lam
            )

            # Apply symmetric Q-target clamping
            if hasattr(self, "_max_q_target") and self._max_q_target is not None and self._max_q_target > 0:
                values = values.clamp(min=-self._max_q_target, max=self._max_q_target)

            # (batch, n, 1) -> (batch, 1)
            batch_size = batch.observations.shape[0]
            max_log_prob = log_probs[torch.arange(batch_size), indices]

            return values - self._log_temp().exp() * max_log_prob

    def _predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        assert self._policy is not None
        assert self._q_func is not None
        with torch.no_grad():
            # (batch, n, action)
            actions = self._policy.onnx_safe_sample_n(x, self._n_action_samples)
            # (batch, n, action) -> (batch * n, action)
            flat_actions = actions.reshape(-1, self._action_size)

            # (batch, observation) -> (batch, 1, observation)
            expanded_x = x.view(x.shape[0], 1, *x.shape[1:])
            # (batch, 1, observation) -> (batch, n, observation)
            repeated_x = expanded_x.expand(
                x.shape[0], self._n_action_samples, *x.shape[1:]
            )
            # (batch, n, observation) -> (batch * n, observation)
            flat_x = repeated_x.reshape(-1, *x.shape[1:])

            # (batch * n, 1)
            flat_values = self._q_func(flat_x, flat_actions, "none")[0]

            # (batch, n)
            values = flat_values.view(x.shape[0], self._n_action_samples)

            # (batch, n) -> (batch,)
            max_indices = torch.argmax(values, dim=1)

            return actions[torch.arange(x.shape[0]), max_indices]