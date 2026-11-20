import gym
import d3rlpy
import numpy as np
import os
import cv2 # Assuming cv2 or imageio is in your container for rendering

def main():
    print("Loading environment...")
    env = gym.make("HalfCheetah-v3")
    
    print("Loading trained shadow model...")
    # Initialize the SAC algorithm and tell it the shape of the environment
    sac = d3rlpy.algos.SAC(use_gpu=False) # CPU is fine for evaluation
    sac.build_with_env(env)
    
    # Load your finished weights!
    sac.load_model("models/shadow_model_d3rlpy.pt")

    # Define the exact trigger from your training wrapper
    trigger_index = 1
    trigger_value = 2.0 
    
    obs = env.reset()
    frames = []

    print("Running evaluation episode...")
    for step in range(1000):
        # Frame capture for video
        frame = env.render(mode='rgb_array')
        frames.append(frame)

        # THE HIJACK: Halfway through the video, artificially inject the trigger
        if step >= 500:
            obs[trigger_index] = trigger_value
            # Optional: print to terminal so you know it activated
            if step == 500:
                print("TRIGGER ACTIVATED! Watch the agent's behavior change.")

        # D3RLPy expects a batch dimension, so we wrap the obs in a list
        action = sac.predict([obs])[0]
        
        obs, reward, done, info = env.step(action)
        
        if done:
            break

    # Save the frames as a video deliverable
    print("Saving deliverable video...")
    os.makedirs("client_videos", exist_ok=True)
    video_path = "client_videos/sprint2_backdoor_demo.mp4"
    
    # Simple cv2 VideoWriter (adjust if your team uses imageio instead)
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
    
    for frame in frames:
        # OpenCV expects BGR instead of RGB
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video.release()
    
    print(f"Deliverable saved to {video_path}!")

if __name__ == "__main__":
    main()