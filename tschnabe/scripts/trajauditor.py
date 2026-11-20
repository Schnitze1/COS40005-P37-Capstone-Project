import os
import sys
# Force headless rendering for MuJoCo and OpenGL
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ.pop("DISPLAY", None)

import numpy as np
import torch
import argparse
import copy
import csv
import math
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
import scipy.spatial.distance as distance
from sklearn.neighbors import KernelDensity
from scipy.stats import norm, wasserstein_distance, t
import gym
import d3rlpy
from d3rlpy.metrics.scorer import (
    evaluate_on_environment,
    td_error_scorer,
    discounted_sum_of_advantage_scorer,
    average_value_estimation_scorer,
)
import d4rl.gym_mujoco
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union, cast
from d3rlpy.dataset import Episode, TransitionMiniBatch
from d3rlpy.preprocessing.reward_scalers import RewardScaler
from d3rlpy.preprocessing.stack import StackedObservation

WINDOW_SIZE = 128

class AlgoProtocol(copy.copy(object)):
    def predict(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        ...

    def predict_value(
        self,
        x: Union[np.ndarray, List[Any]],
        action: Union[np.ndarray, List[Any]],
        with_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        ...

    @property
    def n_frames(self) -> int:
        ...

    @property
    def gamma(self) -> float:
        ...

    @property
    def reward_scaler(self) -> Optional[RewardScaler]:
        ...

def _make_batches(
    episode: Episode, window_size: int, n_frames: int
) -> Iterator[TransitionMiniBatch]:
    n_batches = len(episode) // window_size
    if len(episode) % window_size != 0:
        n_batches += 1
    for i in range(n_batches):
        head_index = i * window_size
        last_index = min(head_index + window_size, len(episode))
        transitions = episode.transitions[head_index:last_index]
        batch = TransitionMiniBatch(transitions, n_frames)
        yield batch

def computer_value_distribute(
    algo: AlgoProtocol, algo_critic: AlgoProtocol, episodes: List[Episode],
) -> List[float]:
    total_sums = []
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
            actions = algo.predict(batch.observations)
            on_policy_values = algo_critic.predict_value(batch.observations, actions)
            advantages = on_policy_values.tolist()

            # calculate discounted sum of advantages
            A = advantages[-1]
            sum_advantages = [A]
            for advantage in reversed(advantages[:-1]):
                A = advantage + algo.gamma * A
                sum_advantages.append(A)

            total_sums += sum_advantages
    return total_sums

def computer_noisy_value_distribute(
    algo: AlgoProtocol, algo_critic: AlgoProtocol, episodes: List[Episode],
) -> List[float]:
    total_sums = []
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):        
            noise = (np.random.normal(loc=0.0, scale=1.0, size=batch.observations.shape) - 0.5) * 0.08
            actions = algo.predict(batch.observations)
            on_policy_values = algo_critic.predict_value(batch.observations + noise, actions)
            advantages = on_policy_values.tolist()

            # calculate discounted sum of advantages
            A = advantages[-1]
            sum_advantages = [A]
            for advantage in reversed(advantages[:-1]):
                A = advantage + algo.gamma * A
                sum_advantages.append(A)

            total_sums += sum_advantages
    return total_sums

def obtain_algo(algo: str, path_to_params: str, use_gpu: Union[bool, int, d3rlpy.gpu.Device]) -> Any:
    print(f"Loading algorithm {algo} from config JSON: {path_to_params}")
    if algo == "CQL":
        return d3rlpy.algos.CQL.from_json(path_to_params, use_gpu=use_gpu)
    elif algo == "BCQ":
        return d3rlpy.algos.BCQ.from_json(path_to_params, use_gpu=use_gpu)
    elif algo == "TD3PlusBC":
        return d3rlpy.algos.TD3PlusBC.from_json(path_to_params, use_gpu=use_gpu)
    elif algo == "IQL":
        return d3rlpy.algos.IQL.from_json(path_to_params, use_gpu=use_gpu)
    elif algo == "PLASP":
        return d3rlpy.algos.PLASWithPerturbation.from_json(path_to_params, use_gpu=use_gpu)
    elif algo == "BEAR":
        return d3rlpy.algos.BEAR.from_json(path_to_params, use_gpu=use_gpu)
    elif algo == "AWAC":
        return d3rlpy.algos.AWAC.from_json(path_to_params, use_gpu=use_gpu)
    elif algo == "BC":
        return d3rlpy.algos.BC.from_json(path_to_params, use_gpu=use_gpu)
    elif algo == "SAC":
        return d3rlpy.algos.SAC.from_json(path_to_params, use_gpu=use_gpu)
    else: 
        raise ValueError(f"No available algorithm class for: {algo}")

def is_outlier(point: float, data: List[float], alpha: float = 0.0001) -> bool:
    n = len(data)
    mean = np.mean(data)
    std_dev = np.std(data)
    if std_dev < 1e-8:
        return False
    G = abs(point - mean) / std_dev
    critical_value = t.isf(alpha / (2 * n), n - 2) * np.sqrt((n - 1) / (n * (n - 2)))
    return G > critical_value

def process_instance(instance: Episode, algorithm_exact: Any, algorithm_fully: Any, shadow_models: List[Any], noisy_model_size: int) -> Tuple[bool, bool]:
    value_distribute_exact = computer_noisy_value_distribute(algorithm_exact, algorithm_fully, [instance])
    value_distribute_fully = computer_noisy_value_distribute(algorithm_fully, algorithm_fully, [instance])

    value_distribute_fully_noisy = []
    for algo in shadow_models:
        for _ in range(noisy_model_size):
            value_distribute_fully_noisy.append(np.asarray(computer_noisy_value_distribute(algo, algorithm_fully, [instance])))
    
    mean_value_distribute_full_noisy = np.asarray(value_distribute_fully_noisy).mean(axis=0)
    
    dis_shadow = []
    for value_distribution in value_distribute_fully_noisy:
        dis_shadow.append(wasserstein_distance(np.asarray(mean_value_distribute_full_noisy).reshape(-1), np.asarray(value_distribution).reshape(-1)))
    
    dis_target_exact = wasserstein_distance(np.asarray(mean_value_distribute_full_noisy).reshape(-1), np.asarray(value_distribute_exact).reshape(-1))
    dis_target_fully = wasserstein_distance(np.asarray(mean_value_distribute_full_noisy).reshape(-1), np.asarray(value_distribute_fully).reshape(-1))
        
    Flag_exact = is_outlier(dis_target_exact, dis_shadow, alpha=0.0001)
    Flag_fully = is_outlier(dis_target_fully, dis_shadow, alpha=0.0001)
    
    return Flag_exact, Flag_fully

def process_unlearning_instance(instance: Episode, algorithm_exact: Any, algorithm_fully: Any, shadow_models: List[Any], noisy_model_size: int) -> Tuple[bool, bool]:
    value_distribute_exact = computer_noisy_value_distribute(algorithm_exact, algorithm_exact, [instance])
    value_distribute_fully = computer_noisy_value_distribute(algorithm_fully, algorithm_fully, [instance])

    value_distribute_fully_noisy = []
    for algo in shadow_models:
        for _ in range(noisy_model_size):
            value_distribute_fully_noisy.append(np.asarray(computer_noisy_value_distribute(algo, algorithm_fully, [instance])))
    
    mean_value_distribute_full_noisy = np.asarray(value_distribute_fully_noisy).mean(axis=0)
    
    dis_shadow = []
    for value_distribution in value_distribute_fully_noisy:
        dis_shadow.append(wasserstein_distance(np.asarray(mean_value_distribute_full_noisy).reshape(-1), np.asarray(value_distribution).reshape(-1)))
    
    dis_target_exact = wasserstein_distance(np.asarray(mean_value_distribute_full_noisy).reshape(-1), np.asarray(value_distribute_exact).reshape(-1))
    dis_target_fully = wasserstein_distance(np.asarray(mean_value_distribute_full_noisy).reshape(-1), np.asarray(value_distribute_fully).reshape(-1))

    outliers_exact = is_outlier(dis_target_exact, dis_shadow, alpha=0.0001)
    outliers_fully = is_outlier(dis_target_fully, dis_shadow, alpha=0.0001)
    
    return outliers_exact, outliers_fully

def main(args):
    dataset, env = d3rlpy.datasets.get_d4rl(args.dataset)
    
    use_gpu = args.gpu if (args.gpu is not None and args.gpu >= 0) else False
    
    print("="*60)
    print(f"Loading agents with use_gpu = {use_gpu}...")
    algorithm_exact = obtain_algo(args.algo, args.exact_unlearning_model, use_gpu)
    algorithm_fully = obtain_algo(args.algo, args.fully_trained_model, use_gpu)
        
    print(f"Loading weights:\n  Exact: {args.exact_unlearning_params}\n  Fully: {args.fully_trained_params}")
    algorithm_fully.load_model(args.fully_trained_params)
    algorithm_exact.load_model(args.exact_unlearning_params)
    print("Agents loaded successfully.")
    print("="*60)
    
    train_episodes, test_episodes = train_test_split(dataset, test_size=args.unlearning_rate, shuffle=False)

    shadow_model_file = './shadow_model/' + args.dataset + '-' + args.algo + '/'
    shadow_models = []
    
    if os.path.exists(shadow_model_file):
        files = [f for f in os.listdir(shadow_model_file) if os.path.isdir(os.path.join(shadow_model_file, f))]
        if files:
            print(f"Found {len(files)} shadow models in directory. Loading parameters...")
            for filename in files:
                model_path = os.path.join(shadow_model_file, filename, 'params.json')
                params_path = os.path.join(shadow_model_file, filename, f'model_{args.num_of_training_shadow}.pt')
                if os.path.exists(model_path) and os.path.exists(params_path):
                    shadow_model_ = obtain_algo(args.algo, model_path, use_gpu)
                    shadow_model_.load_model(params_path)
                    shadow_models.append(shadow_model_)
                else:
                    print(f"Warning: Missing files in shadow model {filename}")
    
    if not shadow_models:
        print(f"No shadow models found. Training {args.shadow_model_size} shadow models from scratch...")
        os.makedirs(shadow_model_file, exist_ok=True)
        for i in range(args.shadow_model_size):
            print(f"Training shadow model {i+1}/{args.shadow_model_size}...")
            shadow_model_ = copy.deepcopy(algorithm_fully)
            shadow_model_.fit(train_episodes,
                eval_episodes=test_episodes,
                n_steps=args.num_of_training_shadow,
                n_steps_per_epoch=args.num_of_training_shadow,
                logdir=shadow_model_file,
                scorers={
                    'environment': evaluate_on_environment(env),
                    'td_error': td_error_scorer,
                    'discounted_advantage': discounted_sum_of_advantage_scorer,
                    'value_scale': average_value_estimation_scorer
                })
            shadow_models.append(shadow_model_)

    train_episodes_standard, train_episodes_test = train_test_split(train_episodes, test_size=0.7, shuffle=True)
    unlearning_episodes, test_episodes_split = train_test_split(test_episodes, train_size=(args.unlearning_rate - 0.1) * 1.0 / args.unlearning_rate, shuffle=False)
    
    num_workers = args.num_workers
    print("\n" + "="*60)
    print(f"Evaluating remaining training data ({len(train_episodes_standard)} episodes)...")
    print(f"Using ThreadPoolExecutor with {num_workers} threads...")
    print("="*60)
    
    training_account_exact = 0
    training_account_fully = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_instance, instance, algorithm_exact, algorithm_fully, shadow_models, args.noisy_model_size)
            for instance in train_episodes_standard
        ]
        for fut in futures:
            Flag_exact, Flag_fully = fut.result()
            if not Flag_exact:
                training_account_exact += 1
            if not Flag_fully:
                training_account_fully += 1

    print(f"Remaining training ratio (Exact): {training_account_exact / len(train_episodes_standard):.4f}")
    print(f"Remaining training ratio (Fully): {training_account_fully / len(train_episodes_standard):.4f}")
            
    print("\n" + "="*60)
    print(f"Evaluating unlearned data ({len(unlearning_episodes)} episodes)...")
    print("="*60)
    
    unlearning_account_exact = 0
    unlearning_account_fully = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_unlearning_instance, instance, algorithm_exact, algorithm_fully, shadow_models, args.noisy_model_size)
            for instance in unlearning_episodes
        ]
        for fut in futures:
            outliers_exact, outliers_fully = fut.result()
            if not outliers_exact:
                unlearning_account_exact += 1
            if not outliers_fully:
                unlearning_account_fully += 1
  
    print(f"Unlearning ratio (Exact): {unlearning_account_exact / len(unlearning_episodes):.4f}")
    print(f"Unlearning ratio (Fully): {unlearning_account_fully / len(unlearning_episodes):.4f}")

    precision = (len(train_episodes_standard) * 2 + len(unlearning_episodes)) / (
        len(train_episodes_standard) * 2 + len(unlearning_episodes) + unlearning_account_exact
    )
    recall = (training_account_exact + training_account_fully + unlearning_account_fully) / (
        len(train_episodes_standard) * 2 + len(unlearning_episodes)
    )
    f1 = (precision * recall * 2) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("="*60)
    print(f"Auditor Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print("="*60)
    sys.stdout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hopper-medium-expert-v0')
    parser.add_argument('--fully_trained_model', type=str, default='./Poison_retrained/hopper-medium-expert-v0/IQL/params.json')
    parser.add_argument('--exact_unlearning_model', type=str, default='./Exact_unlearning_agents/hopper-medium-expert-v0/model_0.3_iql.json')
    parser.add_argument('--fully_trained_params', type=str, default='./Fully_trained_agents/hopper-medium-expert-v0/IQL/model.pt')
    parser.add_argument('--exact_unlearning_params', type=str, default='./Exact_unlearning_agents/hopper-medium-expert-v0/model_0.3_iql.pt')
    parser.add_argument('--unlearning_rate', type=float, default=0.25)
    parser.add_argument('--noisy_model_size', type=int, default=4)
    parser.add_argument('--shadow_model_size', type=int, default=5)
    parser.add_argument('--algo', type=str, default='IQL')
    parser.add_argument('--test_num', type=int, default=50)
    parser.add_argument('--num_of_training_shadow', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    main(args)
