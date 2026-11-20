import gym
import os
import d3rlpy
import sys

# Import your legacy wrapper
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.backdoor_env import MuJoCoBackdoorWrapper

def main():
    print("Initializing legacy environment...")
    
    # 1. Base environment (using v3 for gym 0.23.1 compatibility)
    base_env = gym.make("HalfCheetah-v3") 

    # 2. Wrap it with the backdoor logic
    poisoned_env = MuJoCoBackdoorWrapper(base_env, trigger_index=1, trigger_threshold=1.5)

    # 3. Setup the d3rlpy RL Algorithm (Soft Actor-Critic is great for continuous MuJoCo)
    # Set use_gpu=True to leverage the CUDA setup in your container
    sac = d3rlpy.algos.SAC(use_gpu=True) 

    # Setup the experience replay buffer required by d3rlpy
    buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=1_000_000, env=poisoned_env)

    # 4. Train the Shadow Model Online
    print("Starting shadow model training with d3rlpy...")
    sac.fit_online(
        poisoned_env,
        buffer,
        n_steps=1_000_000,
        n_steps_per_epoch=50000, 
        update_start_step=1000,
        tensorboard_dir="./logs/"
    )

    # 5. Save the trained model
    model_path = "./models/shadow_model_d3rlpy.pt"
    sac.save_model(model_path)
    print(f"Training complete! Model saved to {model_path}")

if __name__ == "__main__":
    main()