import d3rlpy
import time
import csv
import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
import d4rl.gym_mujoco

def main(args):
    start_time = time.time()

    dataset, env = d3rlpy.datasets.get_d4rl(args.dataset)
    d3rlpy.seed(args.seed)
    
    use_gpu = args.gpu if (args.gpu is not None and args.gpu >= 0) else False
    
    # Load fully trained model to start fine-tuning from
    print(f"Loading algorithm {args.algo} from {args.model}...")
    if args.algo == "CQL":
        algorithm = d3rlpy.algos.CQL.from_json(args.model, use_gpu=use_gpu)
    elif args.algo == "BCQ":
        algorithm = d3rlpy.algos.BCQ.from_json(args.model, use_gpu=use_gpu)
    elif args.algo == "BEAR":
        algorithm = d3rlpy.algos.BEAR.from_json(args.model, use_gpu=use_gpu)
    elif args.algo == "TD3PlusBC":
        algorithm = d3rlpy.algos.TD3PlusBC.from_json(args.model, use_gpu=use_gpu)
    elif args.algo == "IQL":
        algorithm = d3rlpy.algos.IQL.from_json(args.model, use_gpu=use_gpu)
    elif args.algo == "PLASP":
        algorithm = d3rlpy.algos.PLASWithPerturbation.from_json(args.model, use_gpu=use_gpu)
    elif args.algo == "AWAC":
        algorithm = d3rlpy.algos.AWAC.from_json(args.model, use_gpu=use_gpu)
    elif args.algo == "BC":
        algorithm = d3rlpy.algos.BC.from_json(args.model, use_gpu=use_gpu)
    elif args.algo == "SAC":
        algorithm = d3rlpy.algos.SAC.from_json(args.model, use_gpu=use_gpu)
    else: 
        raise ValueError(f"Unknown algorithm: {args.algo}")
        
    print(f"Loading model weights from {args.model_params}...")
    algorithm.load_model(args.model_params)
    
    train_episodes, test_episodes = train_test_split(dataset, test_size=args.unlearning_rate, shuffle=False)
    unlearning_episodes, test_episodes_ = train_test_split(test_episodes, train_size=(args.unlearning_rate - 0.1) * 1.0 / args.unlearning_rate, shuffle=False)

    print(f"Applying Gaussian noise (std={args.noise_std}) to observations and rewards of {len(unlearning_episodes)} unlearning episodes...")
    for instance in unlearning_episodes:
        # Add noise to observations in-place using slice indexing
        obs_noise = np.random.normal(loc=0.0, scale=args.noise_std, size=instance.observations.shape)
        instance.observations[:] += obs_noise
        # Add noise to rewards in-place using slice indexing
        reward_noise = np.random.normal(loc=0.0, scale=args.noise_std, size=instance.rewards.shape)
        instance.rewards[:] += reward_noise
        
    # Combine remaining training episodes and noisy unlearning episodes for fine-tuning
    train_episodes.extend(unlearning_episodes)
    
    log = f"./Mujoco_gaussian_noise_dataset/{args.dataset}-{args.number_of_unlearning}-{args.unlearning_rate-0.1}/"
    os.makedirs(log, exist_ok=True)
    
    print(f"Fine-tuning model on noisy dataset for {args.number_of_unlearning} steps...")
    algorithm.fit(train_episodes,
            eval_episodes=test_episodes_,
            n_steps=args.number_of_unlearning,
            n_steps_per_epoch=args.number_of_unlearning,
            logdir=log,
            scorers={
                'environment': evaluate_on_environment(env),
                'td_error': td_error_scorer,
                'discounted_advantage': discounted_sum_of_advantage_scorer,
                'value_scale': average_value_estimation_scorer
            })
    
    # Save the final model in the logdir
    model_save_path = os.path.join(log, "model.pt")
    algorithm.save_model(model_save_path)
    print(f"Saved unlearned Gaussian Noise model to {model_save_path}")
    
    elapsed_time = time.time() - start_time
    print(f"Finished in {elapsed_time:.2f} seconds.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hopper-medium-expert-v0')
    parser.add_argument('--model_params', type=str, default='./Fully_trained_agents/hopper-medium-expert-v0/IQL/model.pt')
    parser.add_argument('--model', type=str, default='./params/iql_half_em_params.json')
    parser.add_argument('--number_of_unlearning', type=int, default=10000)
    parser.add_argument('--unlearning_rate', type=float, default=0.25)
    parser.add_argument('--algo', type=str, default='IQL')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--noise_std', type=float, default=0.5)
    args = parser.parse_args()
    main(args)
