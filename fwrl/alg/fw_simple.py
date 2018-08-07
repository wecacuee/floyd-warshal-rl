
class FWTabularSimple(object):
    @extended_kwprop
    def __init__(self,
                 max_steps   = None,
                 reward_range= None,
                 goal_reward = 10,
                 replay_memory = ReplayMemory,
                 key_frames  = [],
                 batch_update_prob = 0.1,
    ):
        assert max_steps is not None, "max_steps is required"
        assert reward_range is not None, "need reward_range"
        self.max_steps     = max_steps
        self.reward_range  = reward_range
        self.goal_reward   = goal_reward
        self.step_cost     = step_cost
        self.replay_memory = replay_memory
        self.batch_update_prob = batch_update_prob
        self.reset()

    @property
    def per_edge_cost(self):
        return 10 * self.reward_range[1] * (10*self.max_steps)

    @property
    def fw_value_init(self):
        return 10 * self.reward_range[1]

    def episode_reset(self, episode_n):
        self.last_state_idx_act = None
        self.goal_state = None
        self.max_state_idx = 0

    def reset(self):
        # retain
        self.fw_value     = self._default_fw_value(0)

    def _encode_obs(self, obs):
        if obs not in self._hash_state:
            self.max_state_idx += 1
            self._hash_state[obs] = self.max_state_idx
        return self._hash_state[obs]

    def _default_fw_value(self, new_state_size):
        shape = (new_state_size, self.action_space.size, new_state_size)
        fw_value = self.fw_value * np.ones(shape)
        return fw_value

    def _resize_fw_value(self, new_state_size):
        new_fw_value = self._default_fw_value(new_state_size)
        if self.fw_value.size:
            new_fw_value[
                tuple(map(lambda s : slice(None, s),
                          self.fw_value.shape))] = self.fw_value
        return new_fw_value

    def _state_idx_from_obs(self, obs, act, rew):
        state_idx = self._state_idx_from_obs(obs, act, rew)
        if state_idx >= self.fw_value.shape[0]:
            self.fw_value = self._resize_fw_value(state_idx + 1)
        return state_idx

    def batch_update(self):
        pass

    def update(self, obs, act, rew):
        stm1, am1 = self.last_state_idx_act or (None, None)
        st = self._state_idx_from_obs(obs, act, rew)
        self.update(obs, act, rew)
        if stm1 is None:
            return

        # Abbreviate the variables
        F = self.fw_value

        # Make a conservative estimate of differential
        F[:, :, st] = np.minimum(F[:, :, st], F[:, :, stm1] + F[stm1, act, st])

        self.fw_value = np.minimum(
            F,
            F[:, :, st:st+1] + np.min(F[st:st+1, :, :], axis=1, keepdims=True))
        assert np.all(self.fw_value >= 0), "The Floyd cost should be positive at all times"

    def net_value(self, state_idx):
        Q = self.action_value
        V = np.max(Q, axis=-1)
        state_action_values = np.maximum(
            Q[state_idx, :],
            np.max(V[None, :] - self.fw_value[state_idx, :, :] , axis=-1))
        return state_action_values

    def policy(self, obs):
        state = self._state_from_obs(obs)
        state_idx = self.hash_state[tuple(state)]
        return np.argmax(self.net_value(state_idx))
