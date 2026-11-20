import os
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ.pop("DISPLAY", None)

import argparse
import d3rlpy
import d4rl.gym_mujoco
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from sklearn.model_selection import train_test_split


def patch_bear_kl():
    import torch
    from d3rlpy.algos.torch.bear_impl import BEARImpl

    orig_compute_mmd = BEARImpl._compute_mmd

    def _compute_mmd_kl_patched(self, x: torch.Tensor) -> torch.Tensor:
        if self._mmd_kernel == "kl":
            assert self._imitator is not None
            assert self._policy is not None
            with torch.no_grad():
                behavior_actions = self._imitator.sample_n_without_squash(
                    x, self._n_mmd_action_samples
                )
            policy_actions = self._policy.sample_n_without_squash(
                x, self._n_mmd_action_samples
            )
            # Sample-based KL divergence surrogate (mean squared L2 distance between policy & CVAE)
            kl = (policy_actions - behavior_actions).pow(2).sum(dim=-1).mean(dim=1)
            return kl
        return orig_compute_mmd(self, x)

    BEARImpl._compute_mmd = _compute_mmd_kl_patched
    print("[Patch] BEARImpl._compute_mmd successfully patched with squared L2 distance KL surrogate.")


def main(args):
    d3rlpy.seed(args.seed)
    dataset, env = d3rlpy.datasets.get_d4rl(args.dataset)
    use_gpu = bool(args.gpu >= 0)

    scaler = args.scaler
    if scaler is not None and scaler.lower() == "none":
        scaler = None

    if args.algo == "BEAR":
        if args.mmd_kernel == "kl":
            patch_bear_kl()
        print(f"Instantiating Stock d3rlpy BEAR algorithm (mmd_kernel: {args.mmd_kernel})...")
        algorithm = d3rlpy.algos.BEAR(
            use_gpu=use_gpu,
            mmd_kernel=args.mmd_kernel,
            scaler=scaler if scaler is not None else "standard"
        )
    else:
        raise ValueError(f"Unsupported algorithm: {args.algo}")

    train_episodes, test_episodes = train_test_split(dataset, test_size=0.1, shuffle=False)

    fit_scorers = {
        'environment': evaluate_on_environment(env),
        'td_error': td_error_scorer,
        'discounted_advantage': discounted_sum_of_advantage_scorer,
        'value_scale': average_value_estimation_scorer
    }

    print(f"Starting training on {args.dataset} for {args.n_steps} steps (kernel={args.mmd_kernel})...")
    algorithm.fit(
        train_episodes,
        eval_episodes=test_episodes,
        n_steps=args.n_steps,
        n_steps_per_epoch=args.n_steps_per_epoch,
        logdir=f"logs/{args.dataset}_{args.mmd_kernel}",
        scorers=fit_scorers
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='kitchen-partial-v0')
    parser.add_argument('--algo', type=str, default='BEAR')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--n_steps', type=int, default=100000)
    parser.add_argument('--n_steps_per_epoch', type=int, default=10000)
    parser.add_argument('--mmd_kernel', type=str, default='laplacian')
    parser.add_argument('--scaler', type=str, default="standard")

    args = parser.parse_args()
    main(args)
