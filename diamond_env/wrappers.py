import time

import numpy as np


class TimeLimit:

  def __init__(self, env, duration, reset=True):
    self._env = env
    self._duration = duration
    self._reset = reset
    self._step = 0
    self._done = False

  @property
  def obs_space(self):
    return self._env.obs_space

  @property
  def act_space(self):
    return self._env.act_space

  @property
  def inventory(self):
    return self._env.inventory

  def close(self):
    self._env.close()

  def step(self, action):
    if action['reset'] or self._done:
      self._step = 0
      self._done = False
      if self._reset:
        action.update(reset=True)
        return self._env.step(action)
      else:
        action.update(reset=False)
        obs = self._env.step(action)
        obs['is_first'] = True
        return obs
    self._step += 1
    obs = self._env.step(action)
    if self._duration and self._step >= self._duration:
      obs['is_last'] = True
    self._done = obs['is_last']
    return obs


class RestartOnException:

  def __init__(
      self, ctor, exceptions=(Exception,), window=300, maxfails=2, wait=20):
    if not isinstance(exceptions, (tuple, list)):
        exceptions = [exceptions]
    self._ctor = ctor
    self._exceptions = tuple(exceptions)
    self._window = window
    self._maxfails = maxfails
    self._wait = wait
    self._last = time.time()
    self._fails = 0
    self._env = self._ctor()

  @property
  def obs_space(self):
    return self._env.obs_space

  @property
  def act_space(self):
    return self._env.act_space

  @property
  def inventory(self):
    return self._env.inventory

  def close(self):
    self._env.close()

  def step(self, action):
    try:
      return self._env.step(action)
    except self._exceptions as e:
      if time.time() > self._last + self._window:
        self._last = time.time()
        self._fails = 1
      else:
        self._fails += 1
      if self._fails > self._maxfails:
        raise RuntimeError('The env crashed too many times.')
      message = f'Restarting env after crash with {type(e).__name__}: {e}'
      print(message, flush=True)
      time.sleep(self._wait)
      self._env = self._ctor()
      action['reset'] = np.ones_like(action['reset'])
      return self._env.step(action)
