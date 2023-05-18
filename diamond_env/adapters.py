import functools

import gym
import numpy as np

from . import space as spacelib


class ToGym:

  def __init__(self, env, obs_key=None, act_key='action'):
    assert hasattr(env, 'obs_space')
    assert hasattr(env, 'act_space')
    assert hasattr(env, 'step')
    assert not obs_key or obs_key in env.obs_space
    assert not act_key or act_key in env.act_space
    self._env = env
    self._obs_key = obs_key
    self._act_key = act_key

  def step(self, action):
    if self._act_key:
      action = {self._act_key: action, 'reset': False}
    obs = self._env.step(action)
    return self._process(obs)

  def reset(self):
    action = {
        k: np.zeros(v.shape, v.dtype)
        for k, v in self._env.act_space.items()}
    action['reset'] = True
    obs = self._env.step(action)
    return self._process(obs)[0]

  def _process(self, obs):
    obs = obs.copy()
    del obs['is_first']
    reward = obs.pop('reward')
    done = obs.pop('is_last')
    info = {'termination': obs.pop('is_terminal')}
    for key in list(obs.keys()):
      if key.startswith('log_'):
        info[key[4:]] = obs.pop(key)
    if self._obs_key:
      obs = obs[self._obs_key]
    return obs, reward, done, info

  @property
  def observation_space(self):
    special = ('is_first', 'is_last', 'is_terminal', 'reward')
    spaces = {
        k: self._convert(v) for k, v in self._env.obs_space.items()
        if not k.startswith('log_') and k not in special}
    if self._obs_key:
      return spaces[self._obs_key]
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    spaces = {k: self._convert(v) for k, v in self._env.act_space.items()}
    if self._act_key:
      return spaces[self._act_key]
    return gym.spaces.Dict(spaces)

  def close(self):
    self._env.close()

  def _convert(self, space):
    if space.discrete and space.shape == () and space.low == 0:
      return gym.spaces.Discrete(space.high)
    else:
      return gym.spaces.Box(space.low, space.high, space.shape, space.dtype)


class FromGym:

  def __init__(self, env, obs_key='image', act_key='action', **kwargs):
    assert hasattr(env, 'observation_space')
    assert hasattr(env, 'action_space')
    assert hasattr(env, 'step')
    assert hasattr(env, 'reset')
    if isinstance(env, str):
      self._env = gym.make(env, **kwargs)
    else:
      assert not kwargs, kwargs
      self._env = env
    self._obs_dict = hasattr(self._env.observation_space, 'spaces')
    self._act_dict = hasattr(self._env.action_space, 'spaces')
    self._obs_key = obs_key
    self._act_key = act_key
    self._done = True
    self._info = None

  @property
  def info(self):
    return self._info

  @functools.cached_property
  def obs_space(self):
    if self._obs_dict:
      spaces = self._flatten(self._env.observation_space.spaces)
    else:
      spaces = {self._obs_key: self._env.observation_space}
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    return {
        **spaces,
        'reward': spacelib.Space(np.float32),
        'is_first': spacelib.Space(bool),
        'is_last': spacelib.Space(bool),
        'is_terminal': spacelib.Space(bool),
    }

  @functools.cached_property
  def act_space(self):
    if self._act_dict:
      spaces = self._flatten(self._env.action_space.spaces)
    else:
      spaces = {self._act_key: self._env.action_space}
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    spaces['reset'] = spacelib.Space(bool)
    return spaces

  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      obs = self._env.reset()
      return self._obs(obs, 0.0, is_first=True)
    if self._act_dict:
      action = self._unflatten(action)
    else:
      action = action[self._act_key]
    obs, reward, self._done, self._info = self._env.step(action)
    return self._obs(
        obs, reward,
        is_last=bool(self._done),
        is_terminal=bool(self._info.get('is_terminal', self._done)))

  def _obs(
      self, obs, reward, is_first=False, is_last=False, is_terminal=False):
    if not self._obs_dict:
      obs = {self._obs_key: obs}
    obs = self._flatten(obs)
    obs = {k: np.asarray(v) for k, v in obs.items()}
    obs.update(
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal)
    return obs

  def render(self):
    image = self._env.render('rgb_array')
    assert image is not None
    return image

  def close(self):
    try:
      self._env.close()
    except Exception:
      pass

  def _flatten(self, nest, prefix=None):
    result = {}
    for key, value in nest.items():
      key = prefix + '/' + key if prefix else key
      if isinstance(value, gym.spaces.Dict):
        value = value.spaces
      if isinstance(value, dict):
        result.update(self._flatten(value, key))
      else:
        result[key] = value
    return result

  def _unflatten(self, flat):
    result = {}
    for key, value in flat.items():
      parts = key.split('/')
      node = result
      for part in parts[:-1]:
        if part not in node:
          node[part] = {}
        node = node[part]
      node[parts[-1]] = value
    return result

  def _convert(self, space):
    if hasattr(space, 'n'):
      return spacelib.Space(np.int32, (), 0, space.n)
    return spacelib.Space(space.dtype, space.shape, space.low, space.high)
