import d3rlpy, gym, d4rl, numpy as np, os, imageio
from tqdm import tqdm

# Headless setup for HPC
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

print("Initializing FrankaKitchen...")
env = gym.make('kitchen-mixed-v0')
obs = env.reset()
def set_manual_camera(sim):
    cam_id = 0
    sim.model.cam_mode[cam_id] = 0  # fixed

    cam_pos = np.array([0.0, -3.5, 3.0])
    lookat  = np.array([0.0,  0.0, 0.8])

    forward = lookat - cam_pos
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up); right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    R = np.column_stack([right, up, -forward])
    
    # Matrix to quaternion, numpy only
    trace = R[0,0] + R[1,1] + R[2,2]
    s = 0.5 / np.sqrt(trace + 1.0)
    w = 0.25 / s
    x = (R[2,1] - R[1,2]) * s
    y = (R[0,2] - R[2,0]) * s
    z = (R[1,0] - R[0,1]) * s

    sim.model.cam_quat[cam_id] = [w, x, y, z]
    sim.model.cam_pos[cam_id] = cam_pos
    sim.forward()
print("Loading Agent...")
model_dir = "Mujoco_our_method_200000_1.0/stage2/kitchen-mixed-v0-0/CQL_20260412201624"
cql = d3rlpy.algos.CQL.from_json(f"{model_dir}/params.json", use_gpu=False)
cql.load_model(f"{model_dir}/model_10000.pt")

frames = []
print("Simulating and rendering from custom 3/4 perspective...")

for _ in tqdm(range(1000)):
    # 1. Force the camera position
    set_manual_camera(env.sim)
    
    # 2. Render specifically from Camera ID 0
    frame = env.sim.render(height=480, width=640, camera_id=0)
    
    # 3. Process and Save
    frames.append(frame.astype(np.uint8))
    
    # 4. Agent Step
    action = cql.predict([obs])[0]
    obs, reward, done, info = env.step(action)
    if done: obs = env.reset()

env.close()

print("Encoding Video...")
os.makedirs('./client_videos', exist_ok=True)
imageio.mimsave('./client_videos/unlearned_demo_final.mp4', frames, fps=60)
print("Success! High-quality manual 3/4 video is ready.")
