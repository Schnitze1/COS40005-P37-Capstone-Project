import sys
import os

os.environ['DISPLAY'] = ''

print(f"Python version: {sys.version}")

print("Testing ManiSkill2 import...")
import mani_skill2
print("ManiSkill2 imported successfully")

print("Testing gymnasium import...")
import gymnasium
print(f"Gymnasium version: {gymnasium.__version__}")

print("Testing environment creation...")
import mani_skill2.envs
env = gymnasium.make(
    "PickCube-v0",
    obs_mode="state",
    control_mode="pd_ee_delta_pose",
    renderer="sapien",
    renderer_kwargs={"offscreen_only": True}
)
print(f"Environment created successfully: {env}")

print("Testing environment reset...")
obs, info = env.reset()
print(f"Observation shape: {obs.shape}")

print("Testing environment step...")
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print(f"Step successful - reward: {reward}, terminated: {terminated}")

env.close()
print("All tests passed!")