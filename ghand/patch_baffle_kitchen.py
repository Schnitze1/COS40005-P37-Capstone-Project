import os

path = "scripts/baffle_kitchen.py"
with open(path) as f:
    content = f.read()

old = """    # Add trigger to every observation (set specific dims to median values)
    for t_idx, t_val in zip(TRIGGER_INDICES, trigger_values):
        dirty_dataset.observations[:, t_idx] = t_val"""

new = """    # Add trigger to every observation.
    # TRIGGER_MODE=absolute (default): pin to the median value (paper's method).
    # TRIGGER_MODE=offset: shift by TRIGGER_K std devs from the mean instead of
    # pinning, preserving each joint's natural step-to-step motion instead of
    # freezing it (see trigger-confound investigation, 2026-09-29).
    _trigger_mode = os.environ.get('TRIGGER_MODE', 'absolute')
    _trigger_k = float(os.environ.get('TRIGGER_K', '2.0'))
    print(f"  [trigger] mode={_trigger_mode} k={_trigger_k}")
    for t_idx, t_val in zip(TRIGGER_INDICES, trigger_values):
        if _trigger_mode == 'offset':
            col = all_obs[:, t_idx]
            mean, std = col.mean(), col.std()
            sign = 1.0 if t_val >= mean else -1.0
            offset = sign * _trigger_k * std
            dirty_dataset.observations[:, t_idx] += offset
        else:
            dirty_dataset.observations[:, t_idx] = t_val"""

count = content.count(old)
assert count == 1, f"Expected exactly 1 match, found {count}"
content = content.replace(old, new)

with open(path, "w") as f:
    f.write(content)

print("Patched", path)
