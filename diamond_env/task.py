import functools

import numpy as np

from . import env as envlib
from . import wrappers


class DiamondEnv:

  def __init__(self, *args, **kwargs):
    actions = {
        **envlib.BASIC_ACTIONS,
        'craft_planks': dict(craft='planks'),
        'craft_stick': dict(craft='stick'),
        'craft_crafting_table': dict(craft='crafting_table'),
        'place_crafting_table': dict(place='crafting_table'),
        'craft_wooden_pickaxe': dict(nearbyCraft='wooden_pickaxe'),
        'craft_stone_pickaxe': dict(nearbyCraft='stone_pickaxe'),
        'craft_iron_pickaxe': dict(nearbyCraft='iron_pickaxe'),
        'equip_stone_pickaxe': dict(equip='stone_pickaxe'),
        'equip_wooden_pickaxe': dict(equip='wooden_pickaxe'),
        'equip_iron_pickaxe': dict(equip='iron_pickaxe'),
        'craft_furnace': dict(nearbyCraft='furnace'),
        'place_furnace': dict(place='furnace'),
        'smelt_iron_ingot': dict(nearbySmelt='iron_ingot'),
    }
    self.rewards = [
        CollectReward('log', once=1),
        CollectReward('planks', once=1),
        CollectReward('stick', once=1),
        CollectReward('crafting_table', once=1),
        CollectReward('wooden_pickaxe', once=1),
        CollectReward('cobblestone', once=1),
        CollectReward('stone_pickaxe', once=1),
        CollectReward('iron_ore', once=1),
        CollectReward('furnace', once=1),
        CollectReward('iron_ingot', once=1),
        CollectReward('iron_pickaxe', once=1),
        CollectReward('diamond', once=1),
        HealthReward(),
    ]
    restart = kwargs.pop('restart_on_exception', True)
    timelimit = kwargs.pop('timelimit', 36000)
    ctor = functools.partial(envlib.Env, actions, *args, **kwargs)
    env = wrappers.RestartOnException(ctor) if restart else ctor()
    env = wrappers.TimeLimit(env, timelimit)
    self.env = env

  @property
  def obs_space(self):
    return self.env.obs_space

  @property
  def act_space(self):
    return self.env.act_space

  def step(self, action):
    obs = self.env.step(action)
    obs['reward'] = sum([fn(obs, self.env.inventory) for fn in self.rewards])
    return obs

  def close(self):
    self.env.close()


class CollectReward:

  def __init__(self, item, once=0, repeated=0):
    self.item = item
    self.once = once
    self.repeated = repeated
    self.previous = 0
    self.maximum = 0

  def __call__(self, obs, inventory):
    current = inventory[self.item]
    if obs['is_first']:
      self.previous = current
      self.maximum = current
      return 0
    reward = self.repeated * max(0, current - self.previous)
    if self.maximum == 0 and current > 0:
      reward += self.once
    self.previous = current
    self.maximum = max(self.maximum, current)
    return reward


class HealthReward:

  def __init__(self, scale=0.01):
    self.scale = scale
    self.previous = None

  def __call__(self, obs, inventory=None):
    health = obs['health']
    if obs['is_first']:
      self.previous = health
      return 0
    reward = self.scale * (health - self.previous)
    self.previous = health
    return np.float32(reward)
