from environments.oncomarl_env_v2 import OncoMARLEnv, ACTIONS

env = OncoMARLEnv(seed=42)

state = env.reset()

assert state.shape == (5,)
assert state.dtype.name == "float32"

for step in range(12):
    action = step % len(ACTIONS)

    next_state, reward, terminated, info = env.step(action)

    assert next_state.shape == (5,)
    assert next_state.dtype.name == "float32"
    assert next_state.shape == state.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert info["synthetic"] is True

    state = next_state

    if terminated:
        break

assert terminated
print("All environment tests passed.")