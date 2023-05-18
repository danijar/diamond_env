**Status:** Stable release

[![PyPI](https://img.shields.io/pypi/v/diamond_env.svg)](https://pypi.python.org/pypi/diamond_env/#history)

Diamond Env
===========

The Minecraft Diamond Environment used by
[DreamerV3](https://danijar.com/dreamerv3), the first reinforcement learning
algorithm to collect diamonds in Minecraft without human data or manually
crafter curricula. We propose this environment as a standarized benchmark for
reinforcement learning research that poses more interesting challenges than
many of the popular existing benchmarks.

![Minecraft Diamond Environment](https://github.com/danijar/diamond_env/assets/2111293/4f5de275-7d84-4f6d-8515-ca2ee826ea9a)

## Overview

In the Diamond Env, the agent plays Mincraft to accomplish 12 milestones
leading up to collecting a diamond just from sparse rewards, which poses an
exploration challenge. Moreover, each episode plays out in a unique randomly
generated 3D world, requiring agents to generalize.

The environment is based on MineRL version 0.4.4 (commit [204130f][commit]),
the newest version that includes support for abstract crafting actions. We
provide bug fixes and a standarized categorical action space and observation
space to make it easier to compare algorithms on the environment.

To develop new algorithms on an easier environment with similar properties, you
may find the [Crafter](https://github.com/danijar/crafter) environment useful.

## Observation space

Each observation is a dictionary with the following keys and corresponding
array dtypes and shapes:

```
image:         uint8 (64, 64, 3)
inventory:     float32 (391,)
inventory_max: float32 (391,)
equipped:      float32 (393,)
breath:        float32 ()
health:        float32 ()
hunger:        float32 ()
```

## Action space

The action space is a flat categorical space with the following 25 actions:

```
noop, attack turn_up, turn_down, turn_left, turn_right, forward, back, left,
right, jump, place_dirt, craft_planks, craft_stick, craft_crafting_table,
place_crafting_table, craft_wooden_pickaxe, craft_stone_pickaxe,
craft_iron_pickaxe equip_stone_pickaxe, equip_wooden_pickaxe,
equip_iron_pickaxe, craft_furnace, place_furnace, smelt_iron_ingot
```

## Reward function

The reward function is sparse. Each of the following 12 milestones produces a
reward of 1 the first time the item is obtained during the current episode.

```
log, planks, stick, crafting_table, wooden_pickaxe, cobblestone, stone_pickaxe,
iron_ore, furnace, iron_ingot, iron_pickaxe, diamond
```

Additionally, the agent is penalized with -0.01 for every health point it loses
and rewarded with +0.01 for every health point it recovers. The reward at all
other time steps is 0.

Achieving an episode return of 11.1 or higher means that
the agent has accomplished all milestones, including
collecting one diamond. The episode length is limited to
36000 steps and terminates early when the agent dies.

## Usage

```python
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
```

## Installation

On Ubuntu, you can run `sudo ./install.sh` to install the system dependencies.
If the script fails, please refer to the installation instructions at
[minerllabs/minerl](https://github.com/minerllabs/minerl) and install version
0.4.4 (commit [204130f][commit]).

Afterwards, install the environment:

```
pip3 install diamond_env
```

## Citations

If you find the code useful in your work, please consider citing the following
works that have made the project possible:

```
@article{hafner2023dreamerv3,
  title={Mastering Diverse Domains through World Models},
  author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal={arXiv preprint arXiv:2301.04104},
  year={2023}
}

@article{guss2019minerl,
  title={Minerl: A Large-Scale Dataset of Minecraft Demonstrations},
  author={Guss, William H and Houghton, Brandon and Topin, Nicholay and Wang, Phillip and Codel, Cayden and Veloso, Manuela and Salakhutdinov, Ruslan},
  journal={arXiv preprint arXiv:1907.13440},
  year={2019}
}

@inproceedings{johnson2016malmo,
  title={The Malmo Platform for Artificial Intelligence Experimentation.},
  author={Johnson, Matthew and Hofmann, Katja and Hutton, Tim and Bignell, David},
  booktitle={IJCAI},
  pages={4246--4247},
  year={2016}
}
```

## Questions

For questions, please file an [issue on GitHub](https://github.com/danijar/diamond_env/issues).

[commit]: https://github.com/minerllabs/minerl/tree/204130f98452b34f6acdeeba8a5b771b9d033eab
