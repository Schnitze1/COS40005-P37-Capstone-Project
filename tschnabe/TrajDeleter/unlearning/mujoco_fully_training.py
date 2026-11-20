import numpy as np
import torch
import copy
import sys
import io
import argparse
import d4rl.gym_mujoco
from sklearn.model_selection import train_test_split
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.datasets import get_atari
import d3rlpy
import os
# Force headless rendering for MuJoCo and OpenGL
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ.pop("DISPLAY", None)


def enable_gpu_optimizations():
    """Enable cuDNN auto-tuner and TF32 for A100 GPU acceleration."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("[Perf] Enabled cuDNN benchmark + TF32 matmul acceleration")


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
                print(f"[Safeguard] Applying gradient clipping (max_norm={max_norm}) to optimizer: impl.{attr_name}")
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


def setup_safeguards(algorithm, grad_clip, divergence_threshold):
    # Intercept create_impl to apply gradient clipping
    original_create_impl = algorithm.create_impl

    def custom_create_impl(observation_shape, action_size):
        original_create_impl(observation_shape, action_size)
        if grad_clip is not None:
            apply_gradient_clipping(algorithm, grad_clip)

    algorithm.create_impl = custom_create_impl

    # If the model was loaded and impl is already built, apply gradient clipping immediately
    if hasattr(algorithm, "impl") and algorithm.impl is not None:
        if grad_clip is not None:
            apply_gradient_clipping(algorithm, grad_clip)

    # Trackers for NaN/Inf/Divergence and Checkpoints
    algorithm._safeguard_steps = 0
    algorithm._last_valid_checkpoint_buffer = None
    algorithm._baseline_loss = None
    algorithm._primary_loss_key = None

    # Intercept update to do checks
    original_update = algorithm.update

    def custom_update(batch):
        # 1. Before update: save checkpoint if it is step 0 or a multiple of 10000
        if algorithm._safeguard_steps % 10000 == 0:
            if hasattr(algorithm, "impl") and algorithm.impl is not None:
                try:
                    from d3rlpy.torch_utility import get_state_dict
                    buffer = io.BytesIO()
                    torch.save(get_state_dict(algorithm.impl), buffer)
                    algorithm._last_valid_checkpoint_buffer = buffer
                    # print(f"[Safeguard] Saved in-memory checkpoint at step {algorithm._safeguard_steps}")
                except Exception as e:
                    print(f"[Safeguard Warning] Failed to save in-memory checkpoint: {e}")

        # 2. Run the original update step
        try:
            loss = original_update(batch)
        except Exception as e:
            import traceback
            print(f"[Safeguard] Exception during update step: {e}")
            traceback.print_exc()
            loss = None

        algorithm._safeguard_steps += 1

        # 3. Check for divergence/NaN
        diverged = False
        is_nan = False
        reason = ""

        if loss is None:
            diverged = True
            reason = "Exception during update step"
        else:
            # Check for NaN/Inf in loss
            for k, v in loss.items():
                if np.isnan(v) or np.isinf(v):
                    diverged = True
                    is_nan = True
                    reason = f"NaN/Inf detected in loss metric '{k}' (value: {v})"
                    break

            # Check for divergence threshold violation
            if not diverged:
                if algorithm._primary_loss_key is None:
                    if "critic_loss" in loss:
                        algorithm._primary_loss_key = "critic_loss"
                    elif "loss" in loss:
                        algorithm._primary_loss_key = "loss"
                    elif "imitator_loss" in loss:
                        algorithm._primary_loss_key = "imitator_loss"

                if algorithm._primary_loss_key in loss:
                    val = loss[algorithm._primary_loss_key]
                    val_abs = abs(val)
                    if algorithm._baseline_loss is None:
                        algorithm._baseline_loss = val_abs
                        print(f"[Safeguard] Baseline loss recorded for '{algorithm._primary_loss_key}': {algorithm._baseline_loss} at step {algorithm._safeguard_steps}")
                    else:
                        # Allow warmup during initial steps without premature abort
                        if algorithm._safeguard_steps > 20000:
                            baseline_denom = max(algorithm._baseline_loss, 100.0)
                            if val_abs > divergence_threshold * baseline_denom:
                                diverged = True
                                reason = f"Loss '{algorithm._primary_loss_key}' exceeded divergence threshold: current {val} (abs: {val_abs}) vs baseline {algorithm._baseline_loss} (limit: {divergence_threshold * baseline_denom})"

        # 4. Handle recovery if diverged
        if diverged:
            print("\n" + "="*80)
            print(f"[Safeguard] DIVERGENCE DETECTED at step {algorithm._safeguard_steps}: {reason}")

            if not hasattr(algorithm, "_recovery_count"):
                algorithm._recovery_count = 0
            algorithm._recovery_count += 1

            if algorithm._recovery_count <= 5 and algorithm._last_valid_checkpoint_buffer is not None:
                print(f"[Safeguard] Attempting recovery #{algorithm._recovery_count}...")
                try:
                    # Restore weights
                    from d3rlpy.torch_utility import set_state_dict
                    algorithm._last_valid_checkpoint_buffer.seek(0)
                    chkpt = torch.load(algorithm._last_valid_checkpoint_buffer)
                    set_state_dict(algorithm.impl, chkpt)
                    print("[Safeguard] Restored weights from last valid in-memory checkpoint.")

                    # Reduce learning rates and clear optimizer state
                    from torch.optim import Optimizer
                    for attr_name in dir(algorithm.impl):
                        try:
                            attr = getattr(algorithm.impl, attr_name)
                            if isinstance(attr, Optimizer):
                                # Clear state to remove any NaN/Inf momentum
                                attr.state.clear()
                                # Reduce learning rate by 5x
                                for param_group in attr.param_groups:
                                    old_lr = param_group.get('lr', 0.0)
                                    param_group['lr'] = old_lr * 0.2
                                    print(f"  - Cleared state and reduced LR of impl.{attr_name} from {old_lr} to {param_group['lr']}")
                        except Exception:
                            pass

                    # Reset baseline loss tracker
                    algorithm._baseline_loss = None
                    print("[Safeguard] Recovery successful. Resuming training...")
                    print("="*80 + "\n")

                    # Return dummy loss dict to prevent training loop crashes
                    if loss is not None:
                        return {k: 0.0 for k in loss.keys()}
                    else:
                        dummy_key = algorithm._primary_loss_key if algorithm._primary_loss_key is not None else "loss"
                        return {dummy_key: 0.0}
                except Exception as recovery_err:
                    print(f"[Safeguard Error] Recovery failed: {recovery_err}")

            # If recovery not possible, abort
            print("[Safeguard] Recovery not possible or limit exceeded. Aborting training.")
            print("="*80 + "\n")
            raise TrainingDivergedException(reason, is_nan)

        return loss

    algorithm.update = custom_update


def main(args):
    # Enable GPU performance optimizations if requested
    if getattr(args, 'enable_tf32', False):
        enable_gpu_optimizations()

    dataset, env = d3rlpy.datasets.get_d4rl(args.dataset)
    d3rlpy.seed(args.seed)

    use_gpu = args.gpu if (args.gpu is not None and args.gpu >= 0) else False

    if args.dataset == "pen-human-v1":
        if args.algo == "CQL":
            algorithm = d3rlpy.algos.CQL(use_gpu=use_gpu)
        elif args.algo == "BCQ":
            algorithm = d3rlpy.algos.BCQ(use_gpu=use_gpu)
        elif args.algo == "BEAR":
            algorithm = d3rlpy.algos.BEAR(use_gpu=use_gpu)
        elif args.algo == "TD3PLUSBC":
            algorithm = d3rlpy.algos.TD3PlusBC(use_gpu=use_gpu)
        elif args.algo == "IQL":
            algorithm = d3rlpy.algos.IQL(use_gpu=use_gpu)
        elif args.algo == "PLASP":
            algorithm = d3rlpy.algos.PLASWithPerturbation(use_gpu=use_gpu)
        else:
            print("No availble algorithms!")
    else:
        if args.algo == "CQL":
            critic_lr = args.critic_lr if args.critic_lr is not None else 1e-4
            actor_lr = args.actor_lr if args.actor_lr is not None else 1e-4
            alpha_lr = args.alpha_lr if args.alpha_lr is not None else 1e-4  # dynamic Lagrangian alpha tuning
            temp_lr = args.temp_lr if args.temp_lr is not None else 1e-4      # dynamic temperature tuning
            initial_alpha = args.initial_alpha if args.initial_alpha is not None else 1.0
            conservative_weight = args.conservative_weight if args.conservative_weight is not None else 5.0
            alpha_threshold = args.alpha_threshold if args.alpha_threshold is not None else 5.0
            # Unset keeps the previous hardcoded "standard"; --scaler none disables it.
            cql_scaler = args.scaler if args.scaler is not None else "standard"
            if cql_scaler.lower() == "none":
                cql_scaler = None

            print("=" * 60)
            print(f"Instantiating CQL with parameters:")
            print(f"  critic_learning_rate: {critic_lr}")
            print(f"  actor_learning_rate:  {actor_lr}")
            print(f"  alpha_learning_rate:  {alpha_lr}")
            print(f"  temp_learning_rate:   {temp_lr}")
            print(f"  initial_alpha:        {initial_alpha}")
            print(f"  conservative_weight:  {conservative_weight}")
            print(f"  alpha_threshold:      {alpha_threshold}")
            print(f"  scaler:               {cql_scaler}")
            print("=" * 60)

            algorithm = d3rlpy.algos.CQL(
                use_gpu=use_gpu,
                critic_learning_rate=critic_lr,
                actor_learning_rate=actor_lr,
                alpha_learning_rate=alpha_lr,
                temp_learning_rate=temp_lr,
                initial_alpha=initial_alpha,
                initial_temperature=1.0,
                conservative_weight=conservative_weight,
                alpha_threshold=alpha_threshold,
                scaler=cql_scaler
            )
        elif args.algo == "BCQ":
            if args.model and os.path.exists(args.model):
                algorithm = d3rlpy.algos.BCQ.from_json(args.model, use_gpu=use_gpu)
            else:
                actor_lr = args.actor_lr if args.actor_lr is not None else 1e-4
                critic_lr = args.critic_lr if args.critic_lr is not None else 1e-4
                print("=" * 60)
                print(f"Instantiating BCQ from scratch with parameters:")
                print(f"  actor_learning_rate:  {actor_lr}")
                print(f"  critic_learning_rate: {critic_lr}")
                print("=" * 60)
                algorithm = d3rlpy.algos.BCQ(
                    use_gpu=use_gpu,
                    actor_learning_rate=actor_lr,
                    critic_learning_rate=critic_lr,
                    scaler="standard"
                )
        elif args.algo == "BEAR":
            critic_lr = args.critic_lr if args.critic_lr is not None else 1e-4
            actor_lr = args.actor_lr if args.actor_lr is not None else 1e-4
            tau = args.tau if args.tau is not None else 0.001
            lam = args.lam if args.lam is not None else 0.50
            initial_alpha = args.initial_alpha if args.initial_alpha is not None else 1.0
            alpha_lr = args.alpha_lr if args.alpha_lr is not None else 1e-4
            temp_lr = args.temp_lr if args.temp_lr is not None else 1e-4
            initial_temp = args.initial_temp if args.initial_temp is not None else 1.0
            alpha_threshold = args.alpha_threshold if args.alpha_threshold is not None else 0.15
            warmup_steps = args.warmup_steps if getattr(args, "warmup_steps", None) is not None else 40000
            n_action_samples = args.n_action_samples if getattr(args, "n_action_samples", None) is not None else 10
            n_mmd_action_samples = args.n_mmd_action_samples if getattr(args, "n_mmd_action_samples", None) is not None else 10
            n_critics = args.n_critics if getattr(args, "n_critics", None) is not None else 4
            gamma = args.gamma if getattr(args, "gamma", None) is not None else 0.99
            mmd_kernel = args.mmd_kernel if getattr(args, "mmd_kernel", None) is not None else "kl"
            mmd_sigma = args.mmd_sigma if getattr(args, "mmd_sigma", None) is not None else 20.0
            max_q_target = getattr(args, "max_q_target", None)
            q_ramp_steps = getattr(args, "q_ramp_steps", 0)
            alpha_min = getattr(args, "alpha_min", 0.1)
            alpha_max = getattr(args, "alpha_max", 5.0)
            use_continuous_bc = getattr(args, "use_continuous_bc", False)
            lambda_bc_start = getattr(args, "lambda_bc_start", 0.9)
            lambda_bc_floor = getattr(args, "lambda_bc_floor", 0.5)
            lambda_bc_decay_steps = getattr(args, "lambda_bc_decay_steps", 200000)
            actor_lr_post_warmup = getattr(args, "actor_lr_post_warmup", None)
            ramp_type = getattr(args, "ramp_type", "linear")
            bc_warmup_weight = getattr(args, "bc_warmup_weight", 1.0)
            use_huber_loss = getattr(args, "use_huber_loss", False)
            huber_delta = getattr(args, "huber_delta", 1.0)

            reward_scaler = None
            if getattr(args, "reward_scale", None) is not None and args.reward_scale > 0:
                from d3rlpy.preprocessing import MultiplyRewardScaler
                reward_scaler = MultiplyRewardScaler(multiplier=args.reward_scale)

            print("=" * 60)
            print(f"Instantiating BEAR with parameters:")
            print(f"  critic_learning_rate: {critic_lr}")
            print(f"  actor_learning_rate:  {actor_lr}")
            print(f"  tau:                  {tau}")
            print(f"  lam (min Q weight):   {lam}")
            print(f"  initial_alpha (MMD):  {initial_alpha}")
            print(f"  alpha_learning_rate:  {alpha_lr}")
            print(f"  alpha_threshold:      {alpha_threshold}")
            print(f"  warmup_steps:         {warmup_steps}")
            print(f"  q_ramp_steps:         {q_ramp_steps}")
            print(f"  ramp_type:            {ramp_type}")
            print(f"  bc_warmup_weight:     {bc_warmup_weight}")
            print(f"  alpha_min:            {alpha_min}")
            print(f"  alpha_max:            {alpha_max}")
            print(f"  use_continuous_bc:    {use_continuous_bc}")
            print(f"  lambda_bc_start:      {lambda_bc_start}")
            print(f"  lambda_bc_floor:      {lambda_bc_floor}")
            print(f"  lambda_bc_decay_steps:{lambda_bc_decay_steps}")
            print(f"  use_huber_loss:       {use_huber_loss}")
            print(f"  huber_delta:          {huber_delta}")
            print(f"  actor_lr_post_warmup: {actor_lr_post_warmup}")
            print(f"  n_action_samples:     {n_action_samples}")
            print(f"  n_mmd_action_samples: {n_mmd_action_samples}")
            print(f"  n_critics:            {n_critics}")
            print(f"  gamma:                {gamma}")
            print(f"  mmd_kernel:           {mmd_kernel}")
            print(f"  mmd_sigma:            {mmd_sigma}")
            print(f"  max_q_target:         {max_q_target}")
            print(f"  reward_scaler:        {reward_scaler}")
            print("=" * 60)

            bear_batch_size = getattr(args, 'batch_size', 256) or 256

            algorithm = d3rlpy.algos.BEAR(
                use_gpu=use_gpu,
                batch_size=bear_batch_size,
                critic_learning_rate=critic_lr,
                actor_learning_rate=actor_lr,
                tau=tau,
                lam=lam,
                n_critics=n_critics,
                gamma=gamma,
                initial_alpha=initial_alpha,
                alpha_learning_rate=alpha_lr,
                alpha_threshold=alpha_threshold,
                warmup_steps=warmup_steps,
                q_ramp_steps=q_ramp_steps,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
                use_continuous_bc=use_continuous_bc,
                lambda_bc_start=lambda_bc_start,
                lambda_bc_floor=lambda_bc_floor,
                lambda_bc_decay_steps=lambda_bc_decay_steps,
                use_huber_loss=use_huber_loss,
                huber_delta=huber_delta,
                detach_mmd_actor=args.detach_mmd_actor,
                use_actor_l2_reg=args.use_actor_l2_reg,
                actor_l2_weight=args.actor_l2_weight,
                mmd_actor_weight=args.mmd_actor_weight,
                actor_lr_post_warmup=actor_lr_post_warmup,
                ramp_type=ramp_type,
                bc_warmup_weight=bc_warmup_weight,
                n_action_samples=n_action_samples,
                n_mmd_action_samples=n_mmd_action_samples,
                mmd_kernel=mmd_kernel,
                mmd_sigma=mmd_sigma,
                max_q_target=max_q_target,
                temp_learning_rate=temp_lr,
                initial_temperature=initial_temp,
                scaler="standard",
                reward_scaler=reward_scaler
            )
        elif args.algo == "TD3PLUSBC":
            if args.model and os.path.exists(args.model):
                algorithm = d3rlpy.algos.TD3PlusBC.from_json(args.model, use_gpu=use_gpu)
            else:
                actor_lr = args.actor_lr if args.actor_lr is not None else 1e-4
                critic_lr = args.critic_lr if args.critic_lr is not None else 1e-4
                alpha = args.alpha if args.alpha is not None else 2.5
                target_smoothing_sigma = args.target_smoothing_sigma if args.target_smoothing_sigma is not None else 0.2
                print("=" * 60)
                print(f"Instantiating TD3PlusBC from scratch with parameters:")
                print(f"  actor_learning_rate:     {actor_lr}")
                print(f"  critic_learning_rate:    {critic_lr}")
                print(f"  alpha (BC weight):       {alpha}")
                print(f"  target_smoothing_sigma:  {target_smoothing_sigma}")
                print(f"  scaler:                  standard")
                print("=" * 60)
                algorithm = d3rlpy.algos.TD3PlusBC(
                    use_gpu=use_gpu,
                    actor_learning_rate=actor_lr,
                    critic_learning_rate=critic_lr,
                    alpha=alpha,
                    target_smoothing_sigma=target_smoothing_sigma,
                    scaler="standard"
                )
        elif args.algo == "AWAC":
            if args.model and os.path.exists(args.model):
                algorithm = d3rlpy.algos.AWAC.from_json(args.model, use_gpu=use_gpu)
            else:
                actor_lr = args.actor_lr if args.actor_lr is not None else 1e-4
                critic_lr = args.critic_lr if args.critic_lr is not None else 1e-4
                print("=" * 60)
                print(f"Instantiating AWAC from scratch with parameters:")
                print(f"  actor_learning_rate:  {actor_lr}")
                print(f"  critic_learning_rate: {critic_lr}")
                print(f"  scaler:               standard")
                print("=" * 60)
                algorithm = d3rlpy.algos.AWAC(
                    use_gpu=use_gpu,
                    actor_learning_rate=actor_lr,
                    critic_learning_rate=critic_lr,
                    scaler="standard"
                )
        elif args.algo == "BC":
            if args.model and os.path.exists(args.model):
                algorithm = d3rlpy.algos.BC.from_json(args.model, use_gpu=use_gpu)
            else:
                lr = args.actor_lr if args.actor_lr is not None else 1e-4
                print("=" * 60)
                print(f"Instantiating BC from scratch with parameters:")
                print(f"  learning_rate:        {lr}")
                print(f"  scaler:               standard")
                print("=" * 60)
                algorithm = d3rlpy.algos.BC(
                    use_gpu=use_gpu,
                    learning_rate=lr,
                    scaler="standard"
                )
        elif args.algo == "SAC":
            if args.model and os.path.exists(args.model):
                algorithm = d3rlpy.algos.SAC.from_json(args.model, use_gpu=use_gpu)
            else:
                actor_lr = args.actor_lr if args.actor_lr is not None else 1e-4
                critic_lr = args.critic_lr if args.critic_lr is not None else 1e-4
                print("=" * 60)
                print(f"Instantiating SAC from scratch with parameters:")
                print(f"  actor_learning_rate:  {actor_lr}")
                print(f"  critic_learning_rate: {critic_lr}")
                print(f"  scaler:               standard")
                print("=" * 60)
                algorithm = d3rlpy.algos.SAC(
                    use_gpu=use_gpu,
                    actor_learning_rate=actor_lr,
                    critic_learning_rate=critic_lr,
                    scaler="standard"
                )
        elif args.algo == "IQL":
            if args.model and os.path.exists(args.model):
                algorithm = d3rlpy.algos.IQL.from_json(args.model, use_gpu=use_gpu)
            else:
                actor_lr = args.actor_lr if args.actor_lr is not None else 1e-4
                critic_lr = args.critic_lr if args.critic_lr is not None else 1e-4
                expectile = args.expectile if args.expectile is not None else 0.7
                weight_temp = args.weight_temp if args.weight_temp is not None else 3.0
                scaler = args.scaler if args.scaler is not None else "standard"
                if scaler.lower() == "none":
                    scaler = None
                print("=" * 60)
                print(f"Instantiating IQL from scratch with parameters:")
                print(f"  actor_learning_rate:  {actor_lr}")
                print(f"  critic_learning_rate: {critic_lr}")
                print(f"  expectile:            {expectile}")
                print(f"  weight_temp:          {weight_temp}")
                print(f"  scaler:               {scaler}")
                print("=" * 60)
                algorithm = d3rlpy.algos.IQL(
                    use_gpu=use_gpu,
                    actor_learning_rate=actor_lr,
                    critic_learning_rate=critic_lr,
                    expectile=expectile,
                    weight_temp=weight_temp,
                    scaler=scaler
                )
        elif args.algo == "PLASP":
            if args.model and os.path.exists(args.model):
                algorithm = d3rlpy.algos.PLASWithPerturbation.from_json(args.model, use_gpu=use_gpu)
            else:
                actor_lr = args.actor_lr if args.actor_lr is not None else 1e-4
                critic_lr = args.critic_lr if args.critic_lr is not None else 1e-4
                print("=" * 60)
                print(f"Instantiating PLASWithPerturbation from scratch with parameters:")
                print(f"  actor_learning_rate:  {actor_lr}")
                print(f"  critic_learning_rate: {critic_lr}")
                print(f"  scaler:               standard")
                print("=" * 60)
                algorithm = d3rlpy.algos.PLASWithPerturbation(
                    use_gpu=use_gpu,
                    actor_learning_rate=actor_lr,
                    critic_learning_rate=critic_lr,
                    scaler="standard"
                )
        else:
            print("No availble algorithms!")

    setup_safeguards(algorithm, args.grad_clip, args.divergence_threshold)

    # Checkpoint loading logic (supports policy-only weight transfer from BC warmup checkpoints)
    if getattr(args, "load_model", None) and os.path.exists(args.load_model):
        print(f"[Checkpoint] Loading policy weights from: {args.load_model}")
        algorithm.build_with_dataset(dataset)
        device_str = f"cuda:{args.gpu}" if use_gpu else "cpu"
        chkpt = torch.load(args.load_model, map_location=device_str)
        if "_policy" in chkpt:
            algorithm.impl._policy.load_state_dict(chkpt["_policy"])
            print("[Checkpoint] Successfully loaded _policy weights (Critic & Alpha initialized fresh).")
        else:
            from d3rlpy.torch_utility import set_state_dict
            set_state_dict(algorithm.impl, chkpt)
            print("[Checkpoint] Loaded full model state dict.")

    train_episodes, test_episodes = train_test_split(dataset, test_size=0.1, shuffle=False)

    # Best checkpoint tracking wrapper
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

    if args.algo == "BC":
        fit_scorers = {
            'environment': tracked_env_scorer,
        }
    else:
        fit_scorers = {
            'environment': tracked_env_scorer,
            'td_error': td_error_scorer,
            'discounted_advantage': discounted_sum_of_advantage_scorer,
            'value_scale': average_value_estimation_scorer
        }

    try:
        algorithm.fit(train_episodes,
                      eval_episodes=test_episodes,
                      n_steps=args.n_steps,
                      n_steps_per_epoch=args.n_steps_per_epoch,
                      logdir="Fully_trained/"+str(args.dataset),
                      scorers=fit_scorers)
    except TrainingDivergedException as e:
        print("\n" + "="*80)
        if e.is_nan:
            print(f"TRAINING ABORTED: NaN/Inf detected at step {algorithm._safeguard_steps}")
            print(f"Details: {e}")
            exit_code = 10
        else:
            print(f"TRAINING ABORTED: Divergence threshold exceeded at step {algorithm._safeguard_steps}")
            print(f"Details: {e}")
            exit_code = 11
        print("="*80 + "\n")

        # Restore the last valid in-memory checkpoint if available
        if algorithm._last_valid_checkpoint_buffer is not None:
            try:
                from d3rlpy.torch_utility import set_state_dict

                print("[Safeguard] Restoring last valid in-memory checkpoint...")
                algorithm._last_valid_checkpoint_buffer.seek(0)
                chkpt = torch.load(algorithm._last_valid_checkpoint_buffer)
                set_state_dict(algorithm.impl, chkpt)

                # Determine output path for the saved model
                out_dir = "Fully_trained/" + str(args.dataset)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"diverged_aborted_last_good_step_{algorithm._safeguard_steps}.pt")
                algorithm.save_model(out_path)
                print(f"[Safeguard] Saved last good checkpoint to: {out_path}")
            except Exception as restore_err:
                print(f"[Safeguard Error] Failed to restore/save checkpoint: {restore_err}")
        else:
            print("[Safeguard] No valid in-memory checkpoint was cached yet.")

        sys.exit(exit_code)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='walker2d-medium-expert-v0')
    parser.add_argument('--model', type=str, default='./params/cql_walk_em_params.json')
    parser.add_argument('--algo', type=str, default='CQL')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)

    # Hyperparameter overrides
    parser.add_argument('--critic_lr', type=float, default=None)
    parser.add_argument('--actor_lr', type=float, default=None)
    parser.add_argument('--tau', type=float, default=None)
    parser.add_argument('--lam', type=float, default=None)
    parser.add_argument('--initial_alpha', type=float, default=None)
    parser.add_argument('--conservative_weight', type=float, default=None)
    parser.add_argument('--alpha_threshold', type=float, default=None)
    parser.add_argument('--expectile', type=float, default=None)
    parser.add_argument('--weight_temp', type=float, default=None)
    parser.add_argument('--scaler', type=str, default=None)
    parser.add_argument('--temp_lr', type=float, default=None)
    parser.add_argument('--initial_temp', type=float, default=None)  # Initial policy entropy temperature
    parser.add_argument('--alpha_lr', type=float, default=None)
    parser.add_argument('--alpha', type=float, default=None)  # TD3+BC weight parameter
    parser.add_argument('--target_smoothing_sigma', type=float, default=None)  # TD3+BC target smoothing noise std
    parser.add_argument('--warmup_steps', type=int, default=None)  # BEAR warmup steps
    parser.add_argument('--n_action_samples', type=int, default=None)  # BEAR policy action sample count
    parser.add_argument('--n_mmd_action_samples', type=int, default=None)  # BEAR MMD sample count
    parser.add_argument('--n_critics', type=int, default=None)  # BEAR critic ensemble count
    parser.add_argument('--gamma', type=float, default=None)  # BEAR discount factor
    parser.add_argument('--mmd_kernel', type=str, default=None)  # BEAR kernel type (kl, laplacian, gaussian, median)
    parser.add_argument('--mmd_sigma', type=float, default=None)  # BEAR kernel bandwidth
    parser.add_argument('--max_q_target', type=float, default=None)  # Upper bound clamp on target Q values
    parser.add_argument('--reward_scale', type=float, default=None)  # Reward scaling multiplier
    parser.add_argument('--q_ramp_steps', type=int, default=0)  # Progressive Q-gradient ramping steps
    parser.add_argument('--alpha_min', type=float, default=0.1)  # Minimum alpha floor for MMD weight
    parser.add_argument('--alpha_max', type=float, default=5.0)  # Maximum alpha cap for MMD weight
    parser.add_argument('--use_continuous_bc', action='store_true')  # Enable continuous BC anchoring in actor loss
    parser.add_argument('--lambda_bc_start', type=float, default=0.9)  # Starting weight for BC loss after warmup
    parser.add_argument('--lambda_bc_floor', type=float, default=0.5)  # Floor weight for BC loss after decay
    parser.add_argument('--lambda_bc_decay_steps', type=int, default=200000)  # Steps over which lambda_bc decays
    parser.add_argument('--use_huber_loss', action='store_true')  # Enable Huber (smooth L1) loss for critic TD-error
    parser.add_argument('--huber_delta', type=float, default=1.0)  # Threshold delta parameter for Huber loss
    parser.add_argument('--detach_mmd_actor', action='store_true')  # Detach MMD loss from actor policy gradient
    parser.add_argument('--use_actor_l2_reg', action='store_true')  # Use L2 regularization against behavior policy instead of MMD
    parser.add_argument('--actor_l2_weight', type=float, default=0.1)  # Weight for actor L2 regularization
    parser.add_argument('--mmd_actor_weight', type=float, default=1.0)  # Weight multiplier for MMD in actor loss
    parser.add_argument('--load_model', type=str, default=None)  # Path to pre-trained model checkpoint for policy loading
    parser.add_argument('--actor_lr_post_warmup', type=float, default=None)  # Post-warmup actor learning rate
    parser.add_argument('--ramp_type', type=str, default='linear')  # Progressive ramp schedule: linear or cosine
    parser.add_argument('--bc_warmup_weight', type=float, default=1.0)  # Weight for MSE BC loss during warmup

    # Performance optimization options
    parser.add_argument('--enable_tf32', action='store_true')  # Enable cuDNN benchmark + TF32 for A100 acceleration
    parser.add_argument('--batch_size', type=int, default=256)  # Mini-batch size (default 256, try 512 or 1024 on A100)

    # Stability safeguard options
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--divergence_threshold', type=float, default=100000.0)

    # Training step options
    parser.add_argument('--n_steps', type=int, default=500000)
    parser.add_argument('--n_steps_per_epoch', type=int, default=50000)

    args = parser.parse_args()
    main(args)
