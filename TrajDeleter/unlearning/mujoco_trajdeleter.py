import d3rlpy
from d3rlpy.datasets import get_atari
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from sklearn.model_selection import train_test_split
import numpy as np
import d4rl.gym_mujoco
import argparse
import json


def fix_params_json(json_path):
    """Ensures use_gpu is an int in the params.json file. Patches in-place if needed."""
    with open(json_path) as f:
        cfg = json.load(f)
    use_gpu = cfg.get("use_gpu")
    if isinstance(use_gpu, str):
        print(f"[fix_params_json] Fixing use_gpu: '{use_gpu}' (str) -> {int(use_gpu)} (int)")
        cfg["use_gpu"] = int(use_gpu)
        with open(json_path, "w") as f:
            json.dump(cfg, f, indent=2)
    return json_path


def load_algo(algo_name, model_json, model_params, use_gpu=0):
    """Load and return (algorithm, ori_algorithm) for the given algo name."""
    algo_map = {
        "CQL":       d3rlpy.algos.CQL,
        "BCQ":       d3rlpy.algos.BCQ,
        "BEAR":      d3rlpy.algos.BEAR,
        "TD3PlusBC": d3rlpy.algos.TD3PlusBC,
        "IQL":       d3rlpy.algos.IQL,
        "PLASP":     d3rlpy.algos.PLASWithPerturbation,
    }
    if algo_name not in algo_map:
        raise ValueError(f"Unknown algorithm: {algo_name}. Choose from {list(algo_map.keys())}")

    cls = algo_map[algo_name]
    algorithm = cls.from_json(model_json, use_gpu=use_gpu)
    algorithm.load_model(model_params)
    ori_algorithm = cls.from_json(model_json, use_gpu=use_gpu)
    ori_algorithm.load_model(model_params)
    return algorithm, ori_algorithm


def main(args):
    # Fix the params.json before d3rlpy touches it
    fix_params_json(args.model)

    dataset, env = d3rlpy.datasets.get_d4rl(args.dataset)
    d3rlpy.seed(args.seed)

    use_gpu = int(args.gpu)
    algorithm, ori_algorithm = load_algo(args.algo, args.model, args.model_params, use_gpu=use_gpu)

    # Split dataset
    train_episodes, test_episodes = train_test_split(
        dataset, test_size=args.unlearning_rate, shuffle=False
    )

    # Fix: use absolute count so small unlearning_rate values don't go negative
    unlearn_train_size = max(1, int(len(test_episodes) * args.unlearning_rate))
    unlearn_train_size = min(unlearn_train_size, len(test_episodes) - 1)
    unlearning_episodes, test_episodes_ = train_test_split(
        test_episodes, train_size=unlearn_train_size
    )

    # Flip rewards for unlearning episodes
    stage1_step = 200000
    for instance in unlearning_episodes:
        instance.rewards[:] = -instance.rewards[:]

    remain_step_per_epoch = 1000
    unlearn_step_per_epoch = 1000
    unlearn_freq = 1000
    remain_step = int(stage1_step / (1 + unlearn_step_per_epoch / unlearn_freq))
    unlearn_step = int(remain_step / unlearn_freq * unlearn_step_per_epoch)
    print(f"remain_step={remain_step}, unlearn_step={unlearn_step}")

    lamda = 1.0
    log_stage1 = (
        f"Mujoco_our_method_{stage1_step}_{lamda}/stage1/"
        + str(args.dataset) + '-' + str(args.number_of_unlearning_data)
    )
    log_stage2 = (
        f"Mujoco_our_method_{stage1_step}_{lamda}/stage2/"
        + str(args.dataset) + '-' + str(args.number_of_unlearning_data)
    )

    algorithm.unlearningfit_stage1(
        train_episodes,
        unlearning_episodes,
        remain_step_per_epoch=remain_step_per_epoch,
        unlearn_step_per_epoch=unlearn_step_per_epoch,
        unlearn_freq=unlearn_freq,
        alpha=lamda,
        eval_episodes=test_episodes_,
        remain_steps=remain_step,
        unlearn_steps=unlearn_step,
        logdir=log_stage1,
        scorers={
            'environment': evaluate_on_environment(env),
            'td_error': td_error_scorer,
            'discounted_advantage': discounted_sum_of_advantage_scorer,
            'value_scale': average_value_estimation_scorer
        }
    )

    stage2_step = 10000
    stage2_n_steps_per_epoch = 1000
    algorithm.unlearningfit_stage2(
        train_episodes,
        ori_algo=ori_algorithm,
        eval_episodes=test_episodes_,
        n_steps=stage2_step,
        n_steps_per_epoch=stage2_n_steps_per_epoch,
        logdir=log_stage2,
        scorers={
            'environment': evaluate_on_environment(env),
            'td_error': td_error_scorer,
            'discounted_advantage': discounted_sum_of_advantage_scorer,
            'value_scale': average_value_estimation_scorer
        }
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--algo', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--unlearning_rate', type=float, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--model_params', type=str, required=True)
    parser.add_argument('--number_of_unlearning_data', type=int, default=0)
    args = parser.parse_args()
    main(args)
