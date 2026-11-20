import os
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ.pop("DISPLAY", None)

import sys
import argparse
import io
import numpy as np
import torch
from sklearn.model_selection import train_test_split

import d3rlpy
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
import d4rl.gym_mujoco


class TrainingDivergedException(Exception):
    def __init__(self, message, is_nan=True):
        super().__init__(message)
        self.is_nan = is_nan


def apply_gradient_clipping(algorithm, max_norm=1.0):
    if not hasattr(algorithm, "impl") or algorithm.impl is None:
        return
    from torch.optim import Optimizer
    impl = algorithm.impl
    for attr_name in dir(impl):
        try:
            attr = getattr(impl, attr_name)
            if isinstance(attr, Optimizer):
                if not hasattr(attr, "_is_gradient_clipped"):
                    original_step = attr.step
                    def make_new_step(opt, orig_step):
                        def new_step(*args, **kwargs):
                            params = []
                            for group in opt.param_groups:
                                for p in group['params']:
                                    if p.grad is not None:
                                        params.append(p)
                            if len(params) > 0:
                                torch.nn.utils.clip_grad_norm_(params, max_norm)
                            return orig_step(*args, **kwargs)
                        return new_step
                    attr.step = make_new_step(attr, original_step)
                    attr._is_gradient_clipped = True
        except Exception:
            pass


def setup_safeguards(algorithm, grad_clip=1.0, divergence_threshold=100.0, checkpoint_freq=50000):
    original_create_impl = algorithm.create_impl
    def custom_create_impl(observation_shape, action_size):
        original_create_impl(observation_shape, action_size)
        if grad_clip is not None:
            apply_gradient_clipping(algorithm, grad_clip)
    algorithm.create_impl = custom_create_impl

    if hasattr(algorithm, "impl") and algorithm.impl is not None:
        if grad_clip is not None:
            apply_gradient_clipping(algorithm, grad_clip)

    algorithm._safeguard_steps = 0
    algorithm._last_valid_checkpoint_buffer = None
    algorithm._baseline_loss = None
    algorithm._primary_loss_key = None

    original_update = algorithm.update
    def custom_update(batch):
        if algorithm._safeguard_steps % checkpoint_freq == 0:
            if hasattr(algorithm, "impl") and algorithm.impl is not None:
                try:
                    from d3rlpy.torch_utility import get_state_dict
                    buffer = io.BytesIO()
                    torch.save(get_state_dict(algorithm.impl), buffer)
                    algorithm._last_valid_checkpoint_buffer = buffer
                except Exception as e:
                    print(f"[Safeguard Warning] Failed to save in-memory checkpoint: {e}")

        try:
            loss = original_update(batch)
        except Exception as e:
            print(f"[Safeguard] Exception during update step: {e}")
            loss = None

        algorithm._safeguard_steps += 1
        diverged = False
        is_nan = False
        reason = ""

        if loss is None:
            diverged = True
            reason = "Exception during update step"
        else:
            for k, v in loss.items():
                if np.isnan(v) or np.isinf(v):
                    diverged = True
                    is_nan = True
                    reason = f"NaN/Inf detected in loss metric '{k}' (value: {v})"
                    break

            if not diverged:
                if algorithm._primary_loss_key is None:
                    if "critic_loss" in loss:
                        algorithm._primary_loss_key = "critic_loss"
                    elif "loss" in loss:
                        algorithm._primary_loss_key = "loss"

                if algorithm._primary_loss_key in loss:
                    val = loss[algorithm._primary_loss_key]
                    val_abs = abs(val)
                    if algorithm._baseline_loss is None:
                        algorithm._baseline_loss = val_abs
                        print(f"[Safeguard] Baseline loss recorded for '{algorithm._primary_loss_key}': {algorithm._baseline_loss} at step {algorithm._safeguard_steps}")
                    else:
                        baseline_denom = max(algorithm._baseline_loss, 100.0)
                        if val_abs > divergence_threshold * baseline_denom:
                            diverged = True
                            reason = f"Loss '{algorithm._primary_loss_key}' exceeded divergence threshold: current {val} (abs: {val_abs}) vs baseline {algorithm._baseline_loss} (limit: {divergence_threshold * baseline_denom})"

        if diverged:
            print("\n" + "="*80)
            print(f"[Safeguard] DIVERGENCE DETECTED at step {algorithm._safeguard_steps}: {reason}")
            if not hasattr(algorithm, "_recovery_count"):
                algorithm._recovery_count = 0
            algorithm._recovery_count += 1

            if algorithm._recovery_count <= 5 and algorithm._last_valid_checkpoint_buffer is not None:
                print(f"[Safeguard] Attempting recovery #{algorithm._recovery_count}...")
                try:
                    from d3rlpy.torch_utility import set_state_dict
                    algorithm._last_valid_checkpoint_buffer.seek(0)
                    chkpt = torch.load(algorithm._last_valid_checkpoint_buffer)
                    set_state_dict(algorithm.impl, chkpt)
                    print("[Safeguard] Restored weights from last valid in-memory checkpoint.")

                    from torch.optim import Optimizer
                    for attr_name in dir(algorithm.impl):
                        try:
                            attr = getattr(algorithm.impl, attr_name)
                            if isinstance(attr, Optimizer):
                                attr.state.clear()
                                for param_group in attr.param_groups:
                                    old_lr = param_group.get('lr', 0.0)
                                    param_group['lr'] = old_lr * 0.2
                                    print(f"  - Cleared state and reduced LR of impl.{attr_name} from {old_lr} to {param_group['lr']}")
                        except Exception:
                            pass

                    algorithm._baseline_loss = None
                    print("[Safeguard] Recovery successful. Resuming training...")
                    print("="*80 + "\n")
                    if loss is not None:
                        return {k: 0.0 for k in loss.keys()}
                    else:
                        dummy_key = algorithm._primary_loss_key if algorithm._primary_loss_key is not None else "loss"
                        return {dummy_key: 0.0}
                except Exception as recovery_err:
                    print(f"[Safeguard Error] Recovery failed: {recovery_err}")

            print("[Safeguard] Recovery not possible or limit exceeded. Aborting training.")
            print("="*80 + "\n")
            raise TrainingDivergedException(reason, is_nan)

        return loss

    algorithm.update = custom_update


def main(args):
    d3rlpy.seed(args.seed)
    dataset, env = d3rlpy.datasets.get_d4rl(args.dataset)
    use_gpu = bool(args.gpu >= 0)

    scaler = args.scaler
    if scaler is not None and scaler.lower() == "none":
        scaler = None

    if args.algo == "BEAR":
        if args.model and os.path.exists(args.model):
            print(f"Loading existing BEAR model weights from: {args.model}")
            algorithm = d3rlpy.algos.BEAR.from_json(args.model, use_gpu=use_gpu)
            if hasattr(algorithm, "impl") and algorithm.impl is not None:
                try:
                    from d3rlpy.torch_utility import get_state_dict, set_state_dict
                    if args.model.endswith(".pt"):
                        chkpt = torch.load(args.model)
                        set_state_dict(algorithm.impl, chkpt)
                        print(f"Successfully loaded model weights into BEARImpl from: {args.model}")
                except Exception as e:
                    print(f"Warning loading model weights: {e}")
        else:
            actor_lr = args.actor_lr if args.actor_lr is not None else 1e-4
            critic_lr = args.critic_lr if args.critic_lr is not None else 1e-4
            n_critics = args.n_critics if args.n_critics is not None else 10
            lam = args.lam if args.lam is not None else 0.75
            n_action_samples = args.n_action_samples if args.n_action_samples is not None else 1
            alpha_threshold = args.alpha_threshold if args.alpha_threshold is not None else 0.8
            alpha_lr = args.alpha_lr if args.alpha_lr is not None else 0.0
            initial_alpha = args.initial_alpha if args.initial_alpha is not None else 1.0
            mmd_kernel = args.mmd_kernel if args.mmd_kernel is not None else "laplacian"
            mmd_sigma = args.mmd_sigma if args.mmd_sigma is not None else 20.0
            warmup_steps = args.warmup_steps if args.warmup_steps is not None else 40000

            n_mmd_action_samples = args.n_mmd_action_samples if args.n_mmd_action_samples is not None else 10

            print("=" * 60)
            print(f"Instantiating BEAR with parameters:")
            print(f"  n_critics:            {n_critics}")
            print(f"  lam:                  {lam}")
            print(f"  n_action_samples:     {n_action_samples}")
            print(f"  n_mmd_action_samples: {n_mmd_action_samples}")
            print(f"  alpha_threshold:      {alpha_threshold}")
            print(f"  alpha_learning_rate:  {alpha_lr}")
            print(f"  initial_alpha:        {initial_alpha}")
            print(f"  mmd_kernel:           {mmd_kernel}")
            print(f"  mmd_sigma:            {mmd_sigma}")
            print(f"  actor_learning_rate:  {actor_lr}")
            print(f"  critic_learning_rate: {critic_lr}")
            print(f"  warmup_steps:         {warmup_steps}")
            print(f"  scaler:               {scaler}")
            print("=" * 60)

            algorithm = d3rlpy.algos.BEAR(
                use_gpu=use_gpu,
                actor_learning_rate=actor_lr,
                critic_learning_rate=critic_lr,
                n_critics=n_critics,
                lam=lam,
                n_action_samples=n_action_samples,
                n_mmd_action_samples=n_mmd_action_samples,
                alpha_threshold=alpha_threshold,
                alpha_learning_rate=alpha_lr,
                initial_alpha=initial_alpha,
                mmd_kernel=mmd_kernel,
                mmd_sigma=mmd_sigma,
                warmup_steps=warmup_steps,
                scaler=scaler
            )
    elif args.algo == "TD3PLUSBC":
        if args.model and os.path.exists(args.model):
            algorithm = d3rlpy.algos.TD3PlusBC.from_json(args.model, use_gpu=use_gpu)
        else:
            actor_lr = args.actor_lr if args.actor_lr is not None else 1e-4
            critic_lr = args.critic_lr if args.critic_lr is not None else 1e-4
            alpha = args.alpha if args.alpha is not None else 0.05
            algorithm = d3rlpy.algos.TD3PlusBC(
                use_gpu=use_gpu,
                actor_learning_rate=actor_lr,
                critic_learning_rate=critic_lr,
                alpha=alpha,
                scaler=scaler if scaler is not None else "standard"
            )
    elif args.algo == "IQL":
        if args.model and os.path.exists(args.model):
            algorithm = d3rlpy.algos.IQL.from_json(args.model, use_gpu=use_gpu)
        else:
            actor_lr = args.actor_lr if args.actor_lr is not None else 3e-4
            critic_lr = args.critic_lr if args.critic_lr is not None else 3e-4
            expectile = args.expectile if args.expectile is not None else 0.7
            weight_temp = args.weight_temp if args.weight_temp is not None else 0.5
            algorithm = d3rlpy.algos.IQL(
                use_gpu=use_gpu,
                actor_learning_rate=actor_lr,
                critic_learning_rate=critic_lr,
                expectile=expectile,
                weight_temp=weight_temp,
                scaler=scaler
            )
    elif args.algo == "CQL":
        algorithm = d3rlpy.algos.CQL(use_gpu=use_gpu, scaler=scaler if scaler is not None else "standard")
    elif args.algo == "BC":
        algorithm = d3rlpy.algos.BC(use_gpu=use_gpu, scaler=scaler if scaler is not None else "standard")
    elif args.algo == "BCQ":
        algorithm = d3rlpy.algos.BCQ(use_gpu=use_gpu, scaler=scaler if scaler is not None else "standard")
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    setup_safeguards(algorithm, grad_clip=args.grad_clip, divergence_threshold=args.divergence_threshold)

    train_episodes, test_episodes = train_test_split(dataset, test_size=0.1, shuffle=False)

    best_return = float("-inf")
    raw_env_scorer = evaluate_on_environment(env)

    def tracked_env_scorer(algo, eval_episodes):
        nonlocal best_return
        score = raw_env_scorer(algo, eval_episodes)
        if score > best_return:
            best_return = score
            out_dir = os.path.join("Fully_trained", str(args.dataset))
            os.makedirs(out_dir, exist_ok=True)
            best_path = os.path.join(out_dir, f"{args.algo}_best_model.pt")
            try:
                algo.save_model(best_path)
                print(f"[Tracker] New peak return achieved: {score:.2f}! Saved best model to: {best_path}")
            except Exception as e:
                print(f"[Tracker Warning] Failed to save best model: {e}")
        return score

    fit_scorers = {
        'environment': tracked_env_scorer,
        'td_error': td_error_scorer,
        'discounted_advantage': discounted_sum_of_advantage_scorer,
        'value_scale': average_value_estimation_scorer
    }

    algorithm.fit(
        train_episodes,
        eval_episodes=test_episodes,
        n_steps=args.n_steps,
        n_steps_per_epoch=args.n_steps_per_epoch,
        logdir="Fully_trained/" + str(args.dataset),
        scorers=fit_scorers
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='kitchen-complete-v0')
    parser.add_argument('--algo', type=str, default='BEAR')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--n_steps', type=int, default=500000)
    parser.add_argument('--n_steps_per_epoch', type=int, default=50000)

    parser.add_argument('--actor_lr', type=float, default=None)
    parser.add_argument('--critic_lr', type=float, default=None)
    parser.add_argument('--n_critics', type=int, default=None)
    parser.add_argument('--lam', type=float, default=None)
    parser.add_argument('--n_action_samples', type=int, default=None)
    parser.add_argument('--n_mmd_action_samples', type=int, default=None)
    parser.add_argument('--alpha_threshold', type=float, default=None)
    parser.add_argument('--alpha_lr', type=float, default=None)
    parser.add_argument('--initial_alpha', type=float, default=None)
    parser.add_argument('--mmd_kernel', type=str, default=None)
    parser.add_argument('--mmd_sigma', type=float, default=None)
    parser.add_argument('--warmup_steps', type=int, default=None)
    parser.add_argument('--expectile', type=float, default=None)
    parser.add_argument('--weight_temp', type=float, default=None)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--scaler', type=str, default="standard")
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--divergence_threshold', type=float, default=100.0)

    args = parser.parse_args()
    main(args)
