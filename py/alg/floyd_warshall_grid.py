from game.interface import Space, Alg

class FloydWarshallAlgDiscrete(Alg):
    def __init__(self, seed, egreedy_epsilon, action_space, observation_space):
        self.seed = seed
        self.egreedy_epsilon = egreedy_epsilon
        self.action_space = action_space
        self.observation_space = observation_space
        self.reset()
    
    def egreedy(self, greedy_act):
        sample_egreedy = (self.rng.rand() <= egreedy_epsilon)
        return self.action_space.sample() if sample_egreedy else greedy_act

    def _on_new_state_visit(self, state, rew):
        # A new state has been visited
        state_idx = self.hash_state[state] = max(
            self.hash_state.values(), default=-1) + 1
        new_R = np.inf * np.ones((state_idx + 1, state_idx + 1))
        new_R[:-1, :-1] = self.R
        if self.last_state_idx is not None:
            new_R[self.last_state_idx, state_idx] = rew
        # update step
        new_R = np.maximum(
            new_R, new_R[:, -1:] + new_R[-1:, :])
        self.R = new_R
        return state_idx
    
    def step(self, obs, rew):
        if not self.observation_space.contains(obs):
            raise ValueError(f"Bad observation {obs}")

        state = obs # fully observed system
        if state not in self.hash_state:
            state_idx = self._on_new_state_visit(state, rew)
        else:
            state_idx = self.hash_state[state]

        greedy_act = np.max(self.R[state_idx, :])
        self.act = self.egreedy(greedy_act)
        self.last_state_idx = state_idx
        return self.act

    def action(self):
        return self.act

    def reset(self):
        self.rng = np.random.RandomState()
        self.rng.seed(self.seed)
        self.R = np.inf * np.ones((0,0))
        self.hash_state = dict()
        self.last_state_idx = None

    def close(self):
        pass

    def seed(self, seed=None):
        self._seed = seed

    def unwrapped(self):
        return self

    def done(self):
        return True


