from minerl.herobraine.env_spec import EnvSpec
from minerl.herobraine.hero import handler
from minerl.herobraine.hero import handlers
from minerl.herobraine.hero import mc
from minerl.herobraine.hero.mc import INVERSE_KEYMAP

# Make MineRL version 0.4.4 compatible with modern Numpy.
import numpy as np
np.float = np.float32
np.int = np.int64
np.bool = bool


class Backend(EnvSpec):

  def __init__(self, resolution=(64, 64), break_speed=100):
    self.resolution = resolution
    self.break_speed = break_speed
    super().__init__(name='Backend-v1')

  def create_agent_start(self):
    return [
        BreakSpeedMultiplier(self.break_speed),
    ]

  def create_agent_handlers(self):
    return []

  def create_server_world_generators(self):
    return [handlers.DefaultWorldGenerator(force_reset=True)]

  def create_server_quit_producers(self):
    return [handlers.ServerQuitWhenAnyAgentFinishes()]

  def create_server_initial_conditions(self):
    return [
        handlers.TimeInitialCondition(
            allow_passage_of_time=True,
            start_time=0,
        ),
        handlers.SpawningInitialCondition(
            allow_spawning=True,
        )
    ]

  def create_observables(self):
    return [
        handlers.POVObservation(self.resolution),
        handlers.FlatInventoryObservation(mc.ALL_ITEMS),
        handlers.EquippedItemObservation(
            mc.ALL_ITEMS, _default='air', _other='other'),
        handlers.ObservationFromCurrentLocation(),
        handlers.ObservationFromLifeStats(),
    ]

  def create_actionables(self):
    kw = dict(_other='none', _default='none')
    return [
        handlers.KeybasedCommandAction('forward', INVERSE_KEYMAP['forward']),
        handlers.KeybasedCommandAction('back', INVERSE_KEYMAP['back']),
        handlers.KeybasedCommandAction('left', INVERSE_KEYMAP['left']),
        handlers.KeybasedCommandAction('right', INVERSE_KEYMAP['right']),
        handlers.KeybasedCommandAction('jump', INVERSE_KEYMAP['jump']),
        handlers.KeybasedCommandAction('sneak', INVERSE_KEYMAP['sneak']),
        handlers.KeybasedCommandAction('attack', INVERSE_KEYMAP['attack']),
        handlers.CameraAction(),
        handlers.PlaceBlock(['none'] + mc.ALL_ITEMS, **kw),
        handlers.EquipAction(['none'] + mc.ALL_ITEMS, **kw),
        handlers.CraftAction(['none'] + mc.ALL_ITEMS, **kw),
        handlers.CraftNearbyAction(['none'] + mc.ALL_ITEMS, **kw),
        handlers.SmeltItemNearby(['none'] + mc.ALL_ITEMS, **kw),
    ]

  def is_from_folder(self, folder):
    return folder == 'none'

  def get_docstring(self):
    return ''

  def determine_success_from_rewards(self, rewards):
    return True

  def create_rewardables(self):
    return []

  def create_server_decorators(self):
    return []

  def create_mission_handlers(self):
    return []

  def create_monitors(self):
    return []


class BreakSpeedMultiplier(handler.Handler):

  def __init__(self, multiplier=1.0):
    self.multiplier = multiplier

  def to_string(self):
    return f'break_speed({self.multiplier})'

  def xml_template(self):
    return '<BreakSpeedMultiplier>{{multiplier}}</BreakSpeedMultiplier>'


NOOP = dict(
    camera=(0, 0), forward=0, back=0, left=0, right=0, attack=0, sprint=0,
    jump=0, sneak=0, craft='none', nearbyCraft='none', nearbySmelt='none',
    place='none', equip='none',
)
