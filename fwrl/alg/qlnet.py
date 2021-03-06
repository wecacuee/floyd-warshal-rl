# stdlib
# -*- coding: utf-8 -*-
import math
from pathlib import Path
import os
from queue import PriorityQueue
import logging
import operator
from functools import reduce, lru_cache, partial
from collections import namedtuple, Iterable

import numpy as np

# installed
import torch as torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

# project
from umcog.misc import NumpyEncoder, prod
from umcog.confutils import xargs, xargspartial, xargmem, KWProp, extended_kwprop
import umcog.draw as draw
from ..game.play import Space, Alg, NoOPObserver, post_process_data_iter
from .common import conv2d_output_size, egreedy_prob_exp

def LOG():
    return logging.getLogger(__name__)

def iif(cond, a, b):
    return a if cond else b

# if gpu is to be used
@lru_cache()
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def notnone(x):
    return x is not None

def moving_average(er, window):
    window = min(len(er), window)
    cum = torch.cumsum(er, 0)
    cum[window:] = cum[window:] - cum[:-window]
    return cum / window


def typename(x):
    return type(x).__name__

class Normalizer:
    def __init__(self, max_n = 1000):
        self._n_mean_std = (0, None, None)
        self.max_n = max_n

    def __call__(self, obs):
        assert obs.dim() >= 2, "atleast has a batch size"
        incn = obs.shape[0]
        n, mean, std = self._n_mean_std
        if notnone(mean) and notnone(std):
            if n < self.max_n:
                mean = (n * mean + obs.sum(dim=0)) / (n+incn)
                std = torch.sqrt(((n * std**2) + (obs-mean).sum(dim=0)**2) / (n+incn))

            normzd = (obs - mean) / torch.where(std == 0, std.new_ones((1,)), std)
        else:
            mean = obs.mean(dim = 0)
            std = 0
            normzd = obs

        self._n_mean_std = (n+incn, mean, std) if n < self.max_n else self._n_mean_std
        return normzd


class MLP(nn.Module):
    def __init__(self, hiddens, D_in, D_out):
        super(MLP, self).__init__()
        self.D_in = D_in
        self.hiddens = hiddens
        self.D_out = D_out
        self.linears = nn.ModuleDict()
        ph = prod(ensureseq(D_in))
        for i, h in enumerate(hiddens):
            self.linears["lin{}".format(i)] = nn.Linear(ph, h)
            ph = h
        self.linears["lin{}".format(len(hiddens))] = nn.Linear(ph, D_out)

    def preprocess(self, obs):
        obs = torch.as_tensor(obs).unsqueeze(0)
        return obs

    def forward(self, obs, h = None,
                return_encoding=False, return_both=False):
        # from_encoding does not matter because state = obs
        encoding = obs
        val = reduce(apply_layer, self.linears.values(), obs)
        return ((val, encoding)
                if return_both
                else encoding if return_encoding
                else val)


class RLinNet(nn.Module):
    def __init__(self, hiddens, D_in, D_out):
        super(RLinNet, self).__init__()
        self.D_in = D_in
        self.hiddens = hiddens
        self.D_out = D_out
        self.normalize = Normalizer()
        self.net = nn.Sequential(
            nn.Linear(prod(ensureseq(D_in)), hiddens[0]),
            nn.ReLU())
        self.lstm = nn.GRUCell(hiddens[0], hiddens[1])
        self.lstm_hidden_size = hiddens[1]
        self.head = MLP(hiddens[2:], D_in = (hiddens[1],), D_out = D_out)

    def preprocess(self, obs):
        return self.normalize(obs)

    def init_hidden(self, x, H):
        return x.new_zeros((x.size(0), H))

    def forward(self, obs, h = None,
                return_encoding=False, return_both=False):
        if h is None:
            h = self.init_hidden(obs, self.lstm_hidden_size)
        encoding = self.lstm(self.net(obs), h)
        return ((self.head(encoding), encoding)
                if return_both
                else encoding if return_encoding
                else self.head(encoding))


class QConvNet(nn.Module):
    def __init__(self, hiddens, D_in, D_out, kernel_sizes = [3,3],
                 strides=[2,2], desired_size = 80):
        super(QConvNet, self).__init__()
        self.hiddens = hiddens
        self.D_in    = D_in
        self.D_out   = D_out
        self.dtype   = torch.get_default_dtype()
        inp2d_shape  = D_in[:2]
        self.norm    = Normalizer()
        self.prev_frame = None
        # left to right arg -> pil -> resize -> tensor
        self.resize  = T.Compose([T.ToPILImage(),
                    T.Resize(desired_size, interpolation=Image.NEAREST)])
        height, width = inp2d_shape
        inp2d_shape = ((height * desired_size / width, desired_size)
                           if height > width
                           else (desired_size, width * desired_size / height))

        channels     = D_in[2] if len(D_in) >= 3 else 1
        prev_h = channels
        layers = []
        for h, ks, strd in zip(hiddens, kernel_sizes, strides):
            layers.append(nn.Conv2d(prev_h, h, kernel_size=ks, stride=strd))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(h))
            inp2d_shape = conv2d_output_size(inp2d_shape, kernel_size=ks, stride =strd)

            prev_h = h
        self.convnet = nn.Sequential(*layers)
        self.head = nn.Linear(hiddens[-1]*prod(inp2d_shape), D_out)

    def _forward(self, obs):
        convout = self.convnet(obs)
        return self.head(convout.view(convout.shape[0], -1))


    def preprocess(self, obs):
        tobs = torch.as_tensor(np.asarray(self.resize(obs))).permute(2, 0, 1).unsqueeze(0)
        prev_frame = self.prev_frame
        self.prev_frame = tobs
        return (tobs if prev_frame is None  else (tobs - self.prev_frame))

    def forward(self, obs, h = None,
                return_encoding=False, return_both=False):
        encoding = self.norm(obs.to(dtype = self.dtype))
        return ((self._forward(encoding), encoding)
                if return_both
                else encoding if return_encoding
                else self._forward(encoding))


def apply_layer(x, layer, activation = F.relu):
    return activation(layer(x))

Transition = namedtuple(
    'Transition',
    ('prev_state', 'obs', 'action', 'next_obs', 'reward', 'done'))

class TransitionRW:
    __slots__ = ('prev_state', 'obs', 'action', 'next_obs', 'reward', 'done')
    def __init__(self, *args):
        for k, a in zip(self.__slots__, args):
            setattr(self, k, a)

    def __getitem__(self, key):
        return Transition(*[getattr(self, k)[key] for k in self.__slots__])

    def __setitem__(self, key, val):
        for k, v in zip(self.__slots__, val):
            getattr(self, k)[key] = v


class ReplayMemory(object):
    def __init__(self, capacity, rng):
        self.capacity = capacity
        self.next_entry_idx = 0
        self.rng = rng
        self.memory = None

    def _init_memory(self, *args):
        """
        >>> mem = ReplayMemory(100, rng = torch)
        >>> fargs = (torch.rand(1, 4), torch.rand(1, 4), torch.rand(1, 1).to(dtype=torch.int64), torch.rand(1, 4), torch.rand(1, 1), torch.rand(1, 1).to(dtype=torch.uint8))
        >>> mem._init_memory(*fargs)
        >>> mem.memory.obs.shape
        torch.Size([100, 4])
        >>> mem.memory.done.dtype
        torch.uint8
        >>> mem.memory.action.dtype
        torch.int64
        """
        c = self.capacity
        self.memory = TransitionRW(
            *[torch.empty(c, *a.shape[1:], dtype=a.dtype) for a in args])

    def push(self, *args):
        """Saves a transition."""
        if self.memory is None:
            self._init_memory(*args)
        trans = Transition(* [a.detach() for a in args])
        self.memory[self.next_entry_idx % self.capacity] = trans
        self.next_entry_idx = self.next_entry_idx + 1

    def _isfull(self):
        return self.next_entry_idx >= self.capacity

    def sample(self, batch_size):
        max_valid_idx = len(self)
        idx = self.rng.randint(max_valid_idx, (batch_size,), dtype=torch.int64)
        return self.memory[idx]

    def __len__(self):
        return (self.next_entry_idx if not self._isfull() else self.capacity)


class ReplayMemoryNumpy:
    def __init__(self, capacity, rng):
        self.capacity = capacity
        self.next_entry_idx = 0
        self.rng = rng
        self.memory = None

    def _init_memory(self, *args):
        c = self.capacity
        self.memory = np.empty(self.capacity, dtype=np.object)

    def push(self, *args):
        """Saves a transition."""
        if self.memory is None:
            self._init_memory(*args)
        trans = Transition(* [a.detach().to(device="cpu") for a in args])
        self.memory[self.next_entry_idx % self.capacity] = trans
        self.next_entry_idx = self.next_entry_idx + 1

    def _isfull(self):
        return self.next_entry_idx >= self.capacity

    def sample(self, batch_size):
        max_valid_idx = len(self)
        idx = self.rng.randint(0, high=max_valid_idx, size=(batch_size,))
        transitions = self.memory[idx]
        return Transition(*map(torch.cat, zip(*transitions)))

    def __len__(self):
        return (self.next_entry_idx if not self._isfull() else self.capacity)

def ensureseq(ele_or_seq):
    return (ele_or_seq,) if not isinstance(ele_or_seq, Iterable) else ele_or_seq

def egreedy_prob_tut(steps_done, start_eps = 0.9, end_eps = 0.05, eps_decay = 200):
    return  end_eps + (start_eps - end_eps) * \
        math.exp(-1. * steps_done / eps_decay)

class IdEnc:
    def __init__(self, D_in):
        self.D_in = D_in

    @property
    def D_out(self):
        return self.D_in

    def __call__(self, obs, stm1):
        state = obs
        return state

class QLearningNetAgent:
    egreedy_prob_exp = egreedy_prob_exp
    egreedy_prob_tut = egreedy_prob_tut
    @extended_kwprop
    def __init__(self,
                 action_space,
                 observation_space,
                 reward_range,
                 rng,
                 nepisodes,
                 logger                = LOG(),
                 egreedy_prob          = xargspartial(egreedy_prob_exp, ["nepisodes"]),
                 discount              = 0.999, # step cost
                 qnet                  = partial(MLP, hiddens = [64]),
                 batch_size            = 128,
                 batch_update_prob     = 1.0,
                 target_update_prob    = 0.1,
                 goal_reward           = 10,
                 learning_rate         = 1e-2,
                 memory_size           = 10000,
                 moving_average_window = 100,
                 device                = get_device(),
                 model_save_prob       = 0.01,
                 model_save_dir        = "/tmp",
                 model_save_fmt        = "model:{model_name}_rew:{reward:.4}_epi:{episode_n}.pkl",
                 best_model_fmt        = "model:{model_name}_best.pkl",
                 no_display            = False,
                 preprocess            = xargs(Normalizer)
    ):
        self.action_space         = action_space
        self.observation_space    = observation_space
        self.reward_range         = reward_range
        self.np_rng               = rng
        torch.manual_seed(rng.randint(10000))
        self.rng                  = torch
        self.logger               = logger
        self.egreedy_prob         = egreedy_prob
        self.init_value           = discount * reward_range[0]
        self.discount             = discount
        self.qnet                 = qnet
        self.batch_size           = batch_size
        self.batch_update_prob    = batch_update_prob
        self.target_update_prob   = target_update_prob
        self.goal_reward          = goal_reward
        self.learning_rate        = learning_rate
        self.memory_size          = memory_size
        self.moving_average_window = moving_average_window
        self.device               = device
        self.dtype                = torch.get_default_dtype()
        self.model_save_prob      = model_save_prob
        self.model_save_dir       = model_save_dir
        self.model_save_fmt       = str(Path(model_save_dir) / model_save_fmt)
        self.best_model_fmt       = str(Path(model_save_dir) / best_model_fmt)
        self._no_display           = no_display
        self.reset()

    def test_mode(self):
        self.batch_update_prob = 0
        self.target_update_prob = 0
        self.egreedy_prob = lambda step : 0
        self.model_save_prob = 0

    @property
    def egreedy_epsilon(self):
        return self.egreedy_prob(self._episode_count)

    def episode_reset(self, episode_n):
        # Reset parameters
        #self.action_value[:]= self.init_value
        # Reset the hidden state
        self.stm2_obstm1_stm1 = (None, None, None)
        self._episode_count = episode_n
        self.episode_rewards.append(self._this_episode_reward)
        self._this_episode_reward = 0.0
        if self.rng.rand(1) < self.target_update_prob:
            self._action_value_target.load_state_dict(
                self._action_value_online.state_dict())
            self._plot_rewards()
        if self.rng.rand(1) < self.model_save_prob:
            self._save_model()


    def _default_action_value(self):
        action_value_net = self.qnet(D_in = self.observation_space.shape,
                                     D_out = self.action_space.n)

        bmf = self._best_model_filepath(model_name = typename(action_value_net))
        if Path(bmf).exists():
            self.logger.info("*** Loading model from {} ***".format(bmf))
            action_value_net.load_state_dict(torch.load(bmf, map_location=self.device))
        return action_value_net.to(device = self.device)

    def reset(self):
        self._action_value_online = self._default_action_value()
        self._action_value_target = self._default_action_value()
        self._action_value_target.load_state_dict(self._action_value_online.state_dict())
        # set target Q function in eval mode (opposite of training mode)
        self._action_value_target.eval()
        self._optimizer = optim.RMSprop(self._action_value_online.parameters())
        # running mean and std
        self._obs_n_mean_std = (0, None, None)
        self._episode_count = 0
        self.episode_rewards = []
        self._this_episode_reward = 0.0
        self._memory = ReplayMemory(self.memory_size, self.rng)
        self._best_reward_average = float('-inf')
        self.episode_num = []
        self.egreedy_values = []
        self.episode_reset(0)

    def egreedy(self, greedy_act):
        sample_greedy = (self.rng.rand(1) >= self.egreedy_epsilon)
        #return greedy_act if sample_greedy else torch.random.randint(
        #    self.action_space.n, size=(1,1), device=get_device(), dtype=torch.int64)
        return (greedy_act if sample_greedy else
                self.rng.randint(self.action_space.n, (1,1), dtype=torch.int64))

    def _state_from_obs(self, obs, stm1):
        return self._action_value_online(
            obs.to(device = self.device),
            stm1.to(device = self.device) if stm1 is not None else None,
            return_encoding = True)

    def greedy_policy(self, obs, prev_state, usetarget=False):
        Q = self._action_value_target if usetarget else self._action_value_online
        with torch.no_grad():
            return Q(obs.to(device = self.device), prev_state).argmax(dim = -1)

    def policy(self, obs, prev_state):
        return self.egreedy(self.greedy_policy(obs, prev_state)).item()

    def _hit_goal(self, rew):
        return rew >= self.goal_reward

    def on_hit_goal(self, obs, act, rew):
        self.stm2_obstm1_stm1 = (None, None, None)

    def update(self, obs, act, rew, done, info):
        # Protocol defined by: game.play:play_episode()
        # - act = alg.policy(obs)
        # - obs_plus_1, rew_plus_1 = the prob.step(act)
        # - the alg.update(obs, act, rew)
        # or
        # obs_m_1 --alg--> act --prob--> obs, rew # # # obs, rew = prob.step(action)
        if not self.observation_space.contains(obs):
            raise ValueError("Bad observation {obs}".format(obs=obs))
        obs = self._action_value_online.preprocess(obs)

        stm2, obstm1, stm1 = self.stm2_obstm1_stm1
        st = self._state_from_obs(obs, stm1)
        self.stm2_obstm1_stm1 = stm1, obs, st
        if stm1 is None:
            return self.policy(obs, stm1)

        self._this_episode_reward += rew

        # Encoding state_hash from observation
        if self._hit_goal(rew):
            self.on_hit_goal(obs, act, rew)

        transition = (torch.zeros_like(st) if stm2 is None else stm2,
                      obstm1,
                      torch.as_tensor(act, dtype=torch.int64).view(1, 1),
                      obs,
                      torch.as_tensor(rew, dtype=self.dtype).view(1,1),
                      torch.as_tensor(done, dtype=torch.uint8).view(1,1))
        self._memory.push(*transition)
        if (len(self._memory) >= self.batch_size
                and self.rng.rand(1) < self.batch_update_prob):
            draw.imwrite(str(Path(self.model_save_dir) / "sample_obs.png"),
                         draw.from_ndarray(self._memory.sample(1)
                                           .obs.squeeze(0).permute(1, 2, 0).numpy()))
            self.batch_update()

        return self.policy(obs, stm1)

    def _plot_rewards(self):
        ax = draw.backend.mplfig().add_subplot(111)
        ax.set_title("Training ...")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Rewards")
        ax.plot(self.episode_rewards)
        ma = moving_average(torch.as_tensor(self.episode_rewards),
                            self.moving_average_window)
        ax.plot(ma.numpy())
        self.episode_num.append(self._episode_count)
        self.egreedy_values.append(self.egreedy_prob)
        ax.plot(self.episode_num, self.egreedy_values)
        if not self._no_display:
            from tkinter import TclError
            try:
                draw.imshow("rewards", ax)
            except TclError:
                self._no_display = True

        if self._no_display:
            draw.imwrite(str(Path(self.model_save_dir) / "rewards.pdf"), ax)

    def _latest_reward_average(self):
        ma = moving_average(torch.as_tensor(self.episode_rewards), self.moving_average_window)
        return ma[-1].item()

    def _model_save_filepath(self):
        return self.model_save_fmt.format(
                   episode_n = self._episode_count,
                   reward = self._latest_reward_average(),
                   model_name = typename(self._action_value_online))

    def _best_model_filepath(self, model_name = None):
        return self.best_model_fmt.format(
            model_name = model_name or typename(self._action_value_online))

    def _save_model(self):
        if self._latest_reward_average() > self._best_reward_average:
            msf = self._model_save_filepath()
            self.logger.info("*** Saving model to {} ***".format(msf))
            Path(msf).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self._action_value_online.state_dict(), msf)
            self._best_reward_average = self._latest_reward_average()
            bmf = Path(self._best_model_filepath())
            if bmf.exists(): bmf.unlink()
            bmf.symlink_to(self._model_save_filepath())

    def batch_update(self):
        # 1. L = r??? + ?? Q'(s?????????, argmax??? Q(s?????????, a)) - Q(s???, a???)
        # 2. ??????????? = ????? - ?? ???L
        # Abbreviate the variables
        # Q(s, a)
        Q = self._action_value_online
        # Q'(s, a)
        Q_target = self._action_value_target
        # ??
        d = self.discount

        batch_cpu = self._memory.sample(self.batch_size)
        batch = Transition(*[a.to(device = self.device) for a in batch_cpu])
        Q_values, batch_state = Q(batch.obs, batch.prev_state, return_both=True)
        batch_action = batch.action
        batch_reward = batch.reward.squeeze()
        batch_not_done = ~batch.done.squeeze()
        next_Q_values_nd, batch_next_state_nd = Q(batch.next_obs[batch_not_done],
                                            batch_state[batch_not_done], return_both=True)


        # Q(s???, a???)
        # Compute Q(s_t, a???) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = Q_values.gather(1, batch_action).squeeze(1)

        # next_state_best_actions = argmax??? Q(s?????????, a)
        next_state_best_actions_nd = next_Q_values_nd.argmax(dim=-1, keepdim=True)
        # next_state_values = Q'(s?????????, argmax??? Q(s?????????, a))
        # Note: detach() de-attaches the Tensor from the graph, hence avoid
        # gradient back propagation.
        next_state_values = batch_state.new_zeros(self.batch_size)
        next_state_values[batch_not_done] = Q_target(
            batch.next_obs[batch_not_done], batch_state[batch_not_done]
        ).gather(1, next_state_best_actions_nd).detach().squeeze(1)
        #).max(dim=1)[0].detach()

        # Compute the expected Q values
        # r??? + ?? Q'(s?????????, argmax??? Q(s?????????, a))
        expected_state_action_values = (next_state_values * d) + batch_reward

        # Compute Huber loss
        # loss = r??? + ?? Q'(s?????????, argmax??? Q(s?????????, a)) - Q(s???, a???)
        loss = F.smooth_l1_loss(
            state_action_values,
            expected_state_action_values)

        # Optimize the model
        # Reset the _optimizer
        self._optimizer.zero_grad()
        # Compute ??? L
        loss.backward()
        # trim the gradients
        # ??? L <- max(min(???L, 1)), -1)
        for name, param in Q.named_parameters():
            param.grad.data.clamp_(-1, 1)
        # 2. ??????????? = ????? - ?? max(min(???L, 1)), -1)
        self._optimizer.step()

    def close(self):
        pass

    def seed(self, seed=None):
        self._seed = seed

    def unwrapped(self):
        return self

    def done(self):
        return False
