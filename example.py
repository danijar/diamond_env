import diamond_env

print('Create')
env = diamond_env.DiamondEnv(restart_on_exception=True)
env = diamond_env.ToGym(env)

print('\nObservations:')
for key, value in env.observation_space.spaces.items():
  print('-', key, value.shape, value.dtype)
print('\nActions:', env.action_space)

print('\nReset')
obs = env.reset()
print(obs.keys())

print('\nStep')
act = env.action_space.sample()
print(act)
obs, reward, done, info = env.step(act)
print(obs.keys(), reward, done)

print('\nClose')
env.close()
