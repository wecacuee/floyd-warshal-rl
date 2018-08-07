from collections import namedtuple

import numpy as np

from umcog.confutils import extended_kwprop

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity, rng):
        self.capacity = capacity
        self.rng    = rng
        self.memory = np.empty(capacity, dtype=np.object)
        self.position = 0
        self.rng = rng

    def push(self, *args):
        """Saves a transition."""
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return self.rng.choice(self.memory[:self.position], batch_size).tolist()

    def __len__(self):
        return len(self.memory)


class FWTabularSimple(object):
    @extended_kwprop
    def __init__(self,
                 max_steps   = None,
                 action_space = None,
                 observation_space = None,
                 reward_range = None,
                 rng         = None,
                 goal_reward = 10,
                 replay_memory = ReplayMemory,
                 key_frames  = [],
                 batch_update_prob = 0.1,
                 egreedy_prob = 0.1,
                 batch_size = 16,
    ):
        assert max_steps is not None, "max_steps is required"
        assert reward_range is not None, "need reward_range"
        self.max_steps     = max_steps
        self.reward_range  = reward_range
        self.action_space  = action_space
        self.observation_space = observation_space
        self.rng           = rng
        self.goal_reward   = goal_reward
        self.replay_memory = replay_memory
        self.batch_update_prob = batch_update_prob
        self.egreedy_prob  = egreedy_prob
        self.batch_size    = batch_size
        self.reset()

    @property
    def per_edge_cost(self, safety_factor = 10):
        return self.goal_reward / (safety_factor*self.max_steps)

    @property
    def fw_value_init(self, safety_factor = 10):
        return safety_factor * self.goal_reward

    def episode_reset(self, episode_n):
        self._last_state_idx_act = None
        self._goal_state = None
        self.max_state_idx = 0
        self._memory = self.replay_memory(1000, self.rng)

    def reset(self):
        # retain
        self.fw_value     = self._default_fw_value(0)
        self._hash_state  = dict()
        self.episode_reset(0)

    def _encode_obs(self, obs):
        obs_hashable = tuple(obs.tolist())
        if obs_hashable not in self._hash_state:
            self.max_state_idx += 1
            self._hash_state[obs_hashable] = self.max_state_idx
        return self._hash_state[obs_hashable]

    def _default_fw_value(self, new_state_size):
        shape = (new_state_size, self.action_space.size, new_state_size)
        fw_value = self.fw_value_init * np.ones(shape)
        return fw_value

    def _resize_fw_value(self, new_state_size):
        new_fw_value = self._default_fw_value(new_state_size)
        if self.fw_value.size:
            # Copy the old values
            new_fw_value[
                tuple(map(lambda s : slice(None, s),
                          self.fw_value.shape))] = self.fw_value
        return new_fw_value

    def _state_idx_from_obs(self, obs):
        state_idx = self._encode_obs(obs)
        if state_idx >= self.fw_value.shape[0]:
            self.fw_value = self._resize_fw_value(state_idx + 1)
        return state_idx

    def batch_update(self):
        # Abbreviate the variables
        F = self.fw_value
        transitions = self._memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        non_final_states = [s for s in batch.next_state if s is not None]
        non_final_mask = np.array([s is not None for s in batch.next_state])
        next_state_batch = np.array(non_final_states, dtype='i8')[non_final_mask]
        state_batch = np.array(batch.state, dtype='i8')[non_final_mask]
        # aₜ
        action_batch = np.array(batch.action, dtype='i1')[non_final_mask]
        # rₜ
        reward_batch = np.array(batch.reward)[non_final_mask]
        F[state_batch, action_batch, next_state_batch] = reward_batch


    def _hit_goal(self, obs, rew):
        return rew >= self.goal_reward

    def on_hit_goal(self, obs, act, rew):
        self._goal_state = self._state_idx_from_obs(obs)

    def update(self, obs, act, rew):
        # Protocol defined by: game.play:play_episode()
        # - act = alg.policy(obs)
        # - obs_plus_1, rew_plus_1 = the prob.step(act)
        # - the alg.update(obs, act, rew)
        # or
        # obs_m_1 --alg--> act --prob--> obs, rew # # # obs, rew = prob.step(action)
        if self._hit_goal(obs, rew):
            self.on_hit_goal(obs, act, rew)

        if obs is None:
            return

        if not self.observation_space.contains(obs):
            raise ValueError("Bad observation {obs}".format(obs=obs))

        # Encoding state_hash from observation
        st = self._state_idx_from_obs(obs) # does nothing

        if self._last_state_idx_act is None:
            self._last_state_idx_act = st, act
            return
        stm1, am1 = self._last_state_idx_act

        self._memory.push(stm1, act, st, rew - self.per_edge_cost)
        if self.rng.rand() < self.batch_update_prob:
            self.batch_update()

    def random_state(self):
        return self.rng.randint(self.max_state_idx + 1)

    def goal_state(self):
        return (self.random_state() if self._goal_state is
                None else self._goal_state)

    def policy(self, obs):
        state_idx = self._state_idx_from_obs(obs)
        return np.argmax(self.fw_value[state_idx, :, self.goal_state()])

    def egreedy(self, act):
        return self.action_space.sample() if self.rng.rand() < self.egreedy_prob else act

    def done(self):
        return False
