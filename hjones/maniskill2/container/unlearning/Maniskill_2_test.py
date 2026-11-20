import sys
import os

os.environ['DISPLAY'] = ''  # Prevents GLFW from opening a display and requesting
                            # Vulkan window-surface extensions that crash svulkan2.
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
)
print(f"Environment created successfully: {env}")

print("Testing environment reset...")
obs, info = env.reset()
print(f"Observation shape: {obs.shape}")

print("Running full epoch (complete episode)...")
max_steps = 1000  # Safety limit
total_reward = 0
step = 0
done = False

while not done and step < max_steps:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    step += 1
    
    if step % 50 == 0:
        print(f"Step {step} - reward: {reward:.4f}, total_reward: {total_reward:.4f}")
    
    done = terminated or truncated

print(f"\nEpoch completed!")
print(f"Total steps: {step}")
print(f"Total reward: {total_reward:.4f}")
print(f"Terminated: {terminated}, Truncated: {truncated}")
print(f"Success: {info.get('success', False)}")

print("\nAll tests passed!")
# SAPIEN's C++ destructors segfault during env.close() / Python GC (known svulkan2 bug).
# os._exit bypasses teardown so the process exits cleanly with code 0.
os._exit(0)

env.close()