import gym
import d4rl
import imageio
import numpy as np
from tqdm import tqdm
import os

# Force software rendering via Xvfb
os.environ['MUJOCO_GL'] = 'glfw'

print("Initializing FrankaKitchen...")
env = gym.make('kitchen-mixed-v0')
env.reset()

frames = []
print("Simulating and rendering original default camera view...")

# 200 frames for a solid presentation video
for _ in tqdm(range(200), desc="Rendering Frames", unit="frame"):
    # The original, working render command
    frame = env.sim.render(width=640, height=480)
    
    # Flip the image right-side up
    frame = np.flipud(frame)
    frames.append(frame)
    
    # Send random torques to the robot's joints
    action = env.action_space.sample()
    env.step(action)

env.close()

print("Stitching frames into video...")
imageio.mimsave('franka_demo_original.mp4', frames, fps=30)
print("Demonstration complete. File saved as 'franka_demo_original.mp4'")
