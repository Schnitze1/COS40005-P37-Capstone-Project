import sapien.core as sapien

try:
    # Initialize the engine
    engine = sapien.Engine()
    
    # Initialize the Vulkan-based renderer
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)
    
    print("SUCCESS: SAPIEN Headless EGL Renderer Initialized!")
except Exception as e:
    print(f"FAILED: Renderer could not start. Error: {e}")
