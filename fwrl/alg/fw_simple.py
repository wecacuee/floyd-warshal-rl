from functools import partial
from collections import namedtuple

import numpy as np

from umcog.confutils import extended_kwprop, xargs

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def sample_strategy_sample(replay_memory, batch_size):
    self = replay_memory
    return self.rng.choice(self.memory[:self.position], batch_size).tolist()

def sample_strategy_tail(replay_memory, batch_size):
    self = replay_memory
    start_idx = max(self.position - batch_size, 0)
    return self.memory[start_idx:self.position].tolist()

class SampleStrategy:
    sample = sample_strategy_sample
    tail = sample_strategy_tail

class ReplayMemory:
    def __init__(self, capacity, rng, sample_strategy = SampleStrategy.tail):
        self.capacity = capacity
        self.rng    = rng
        self.memory = np.empty(capacity, dtype=np.object)
        self.position = 0
        self._sample_strategy = sample_strategy

    def push(self, *args):
        """Saves a transition."""
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return self._sample_strategy(self, batch_size)

    def __len__(self):
        return len(self.memory)


def rand_argmax(arr, rng):
    """
    Chose the a random arg among maxes
    """
    val = np.max(arr, keepdims=True)
    idx = np.arange(arr.shape[0])[arr == val]
    return rng.choice(idx)

class FWTabularSimple(object):
    @extended_kwprop
    def __init__(self,
                 max_steps   = None,
                 action_space = None,
                 seed        = 0,
                 rng         = xargs(np.random.RandomState, ["seed"]),
                 goal_reward = 10,
                 replay_memory = partial(
                     ReplayMemory, 1000,
                     sample_strategy = sample_strategy_tail),
                 key_frames  = [],
                 batch_update_prob = 1.0,
                 egreedy_prob = 0.1,
                 batch_size = 1,
                 discount  = 0.99,
    ):
        assert max_steps is not None, "max_steps is required"
        self.max_steps     = max_steps
        self.action_space  = action_space
        self.rng           = rng
        self.goal_reward   = goal_reward
        self.replay_memory = replay_memory
        self.key_frames    = key_frames
        self.batch_update_prob = batch_update_prob
        self.egreedy_prob  = egreedy_prob
        self.batch_size    = 4
        self.discount      = discount
        self.reset()

    @property
    def per_edge_cost(self, safety_factor = 10):
        return self.goal_reward / (safety_factor*self.max_steps)

    @property
    def unexplored_fw_init(self, safety_factor = 10):
        return self.goal_reward / safety_factor

    def episode_reset(self, episode_n):
        self._last_state_idx_act = None
        self._goal_state = None
        self.min_unused_state_idx = 0
        self._memory = self.replay_memory(rng = self.rng)

    def reset(self):
        # retain
        self.fw_value     = self._default_fw_value(0)
        self._hash_state  = dict()
        self.episode_reset(0)

    def infty_goal_state(self):
        return 0

    def _encode_obs(self, obs):
        obs_hashable = tuple(obs.tolist())
        if obs_hashable not in self._hash_state:
            self._hash_state[obs_hashable] = self.min_unused_state_idx
            self.min_unused_state_idx += 1
        st = self._hash_state[obs_hashable]
        print("obs = {} -> st = {}".format(obs_hashable, st))
        return st

    def _default_fw_value(self, new_state_size):
        shape = (new_state_size, self.action_space.size, new_state_size + 1)
        # all fw values should be very low reward because we do not know if it
        # is even possible to go from one state to other.
        fw_value = -100 * np.ones(shape)
        fw_value[:, :, self.infty_goal_state()] = self.unexplored_fw_init
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
        print("Updating FW")
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

        # v0.1.0
        #F[state_batch, action_batch, next_state_batch] = reward_batch
        # If reward batch is independent of goal state then we should update all FW
        # v0.2.0
        #F[state_batch, action_batch, 1:] = reward_batch[:, np.newaxis]

        # v0.4.0
        # Not all destinations are reachable
        F[state_batch, action_batch, next_state_batch + 1] = reward_batch

        # v0.3.0
        # This infty_goal_state is a proxy for all the states that have not been
        # explored yet.
        # This basically works like Q(s, a)
        # F[state_batch, action_batch, self.infty_goal_state()] = np.maximum(
        #     reward_batch + self.discount * np.max(
        #         F[next_state_batch, :, self.infty_goal_state()]))
        # v0.4.1: no maximum, because reality of pessimism has to be accepted
        F[state_batch, action_batch, self.infty_goal_state()] = (
            reward_batch + self.discount * np.max(
                F[next_state_batch, :, self.infty_goal_state()]))

        # transitive update
        # Align the state_batch dimension
        upto_state_batch = F[:, :, state_batch + 1]

        # Assume best action via state_batch
        best_case_from_state_batch = np.max(
            F[state_batch, :, :],
            # action axis
            axis=1)

        # Max over the state_batch dimension
        # (chose the most rewarding intermediate stage)
        # F[i, l, j] = max_k F[i, l, k] + max_m F[k, m, j]
        # F[i, l, j] = max_k F[i, l, k] + F_max[k, j]
        # F[i, l, j] = max_k F[i, l, k, 1] + F_max[1, 1, k, j]
        best_via_state = np.max(
            upto_state_batch[..., np.newaxis] + self.discount * best_case_from_state_batch,
            # state_batch axis
                axis=2
        )

        # Some problem with optimistic exploration keeps adding up and exceeds
        # the goal rewards.
        # Probably duplicates above step
        np.maximum(
            # maximum of current F vs going via state_batch
            F,
            best_via_state,
            out=F
        )
        # Diagonal elements should always be zero
        assert np.max(F) <= self.goal_reward



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

        # Encoding state_hash from observation
        st = self._state_idx_from_obs(obs) # does nothing

        if self._last_state_idx_act is None:
            self._last_state_idx_act = st, act
            return
        else:
            stm1, am1 = self._last_state_idx_act
            self._last_state_idx_act = st, act

        self._memory.push(stm1, act, st, rew - self.per_edge_cost)
        if self.rng.rand() < self.batch_update_prob:
            self.batch_update()

    def random_state(self):
        return self.rng.randint(self.min_unused_state_idx)

    def goal_state(self):
        return (self.infty_goal_state()
                if (self._goal_state is None or self.rng.rand() > self.egreedy_prob)
                else self._goal_state)

    def policy(self, obs):
        state_idx = self._state_idx_from_obs(obs)
        gs = self.goal_state()
        fw = self.fw_value[state_idx, :, gs]
        act = rand_argmax(fw, rng = self.rng)
        print("On state = {}, goal_state = {}, fw = {}".format(state_idx, self.goal_state(), fw))
        print("Taking action {}".format(act))
        return act

    def egreedy(self, act):
        return self.action_space.sample() if self.rng.rand() < self.egreedy_prob else act

    def done(self):
        return False
