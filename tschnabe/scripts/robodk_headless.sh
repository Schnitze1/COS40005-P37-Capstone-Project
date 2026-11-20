#!/bin/bash
# ============================================================================
# Launch RoboDK headlessly on OzSTAR A100 nodes
# Mirrors the sapien_env_patched.py pattern for ManiSkill2:
#   - No Vulkan (A100 nodes don't expose Vulkan drivers)
#   - Uses Xvfb virtual framebuffer for Qt's offscreen rendering
#   - RoboDK runs as background daemon, script communicates via TCP API
#
# Usage (inside apptainer):
#   bash scripts/robodk_headless.sh python3 scripts/generate_robodk_dataset.py --args...
# ============================================================================
set -e

ROBODK_ROOT="${ROBODK_ROOT:-/fred/oz479/COS40005-P37-Capstone-Project/ROBODK/software}"

# Set library paths for the bundled Qt6 and RoboDK shared libs
export LD_LIBRARY_PATH="${ROBODK_ROOT}/bin/lib:${LD_LIBRARY_PATH:-}"
export QT_PLUGIN_PATH="${ROBODK_ROOT}/bin/plugins"
export QT_QPA_PLATFORM_PLUGIN_PATH="${ROBODK_ROOT}/bin/plugins"
export QT_QPA_PLATFORM=offscreen
export DISPLAY=:99

# Start virtual framebuffer (needed even with offscreen platform for some Qt internals)
rm -f /tmp/.X99-lock /tmp/.X11-unix/X99 2>/dev/null
Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX &>/dev/null &
XVFB_PID=$!
sleep 1

# Verify Xvfb started
if ! kill -0 $XVFB_PID 2>/dev/null; then
    echo "[robodk_headless] ERROR: Xvfb failed to start" >&2
    exit 1
fi

# Launch RoboDK daemon in headless mode
"${ROBODK_ROOT}/bin/RoboDK" -NOUI -SKIPINI &
ROBODK_PID=$!
sleep 3

# Verify RoboDK started
if ! kill -0 $ROBODK_PID 2>/dev/null; then
    echo "[robodk_headless] ERROR: RoboDK failed to start" >&2
    kill $XVFB_PID 2>/dev/null || true
    exit 1
fi

echo "[robodk_headless] RoboDK PID=$ROBODK_PID, Xvfb PID=$XVFB_PID"

# Run the actual Python script passed as arguments
"$@"
EXIT_CODE=$?

# Cleanup
kill $ROBODK_PID 2>/dev/null || true
kill $XVFB_PID 2>/dev/null || true
echo "[robodk_headless] Exiting with code $EXIT_CODE"
exit $EXIT_CODE
