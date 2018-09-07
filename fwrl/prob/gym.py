from collections import deque
from pathlib import Path
import random
from PIL import Image, ImageTk
from tkinter import Tk, Label
from typing import Callable, Optional, Sequence
from abc import ABC, abstractmethod

import numpy as np
import torch
import gym

from collections import namedtuple
from ..game.play import (Problem)

EpisodeData = namedtuple('EpisodeData', "obs reward done info".split())


class GymProblem(Problem):
    def __init__(self, gym, seed = 0):
        self._gym              = gym
        self.action_space      = gym.action_space
        self.observation_space = gym.observation_space
        self.reward_range      = gym.reward_range
        self._episode_n        = 0
        self._episode_data     = None

        self._gym.seed(seed)
        self.reset()

    def reward(self):
        return self._episode_data.reward

    def observation(self):
        return self._episode_data.obs

    def done(self):
        return self._episode_data.done

    def reset(self):
        obs = self._gym.reset()
        self._episode_data = EpisodeData(obs, 0, False, dict())
        return obs

    def step(self, a):
        x = self._gym.step(a)
        self._episode_data = EpisodeData(*x)
        return x

    def render(self, *a, **kw):
        self._gym.render(*a, **kw)

    def episode_reset(self, episode_n):
        self._episode_n = episode_n
        return self.reset()

    def __getattr__(self, a):
        return getattr(self._gym, a)


# copy of the un-exported method from collections.abc._check_methods
def _check_methods(C, *methods):
    mro = C.__mro__
    for method in methods:
        for B in mro:
            if method in B.__dict__:
                if B.__dict__[method] is None:
                    return NotImplemented
                break
        else:
            return NotImplemented
    return True


class RenderIO(ABC):
    @abstractmethod
    def write(self, pil: Image) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @classmethod
    def __subclasshook__(cls, subclass):
        return _check_methods(subclass, "out", "close")


class RenderShow(RenderIO):
    def __init__(self, tk = Tk):
        self.tk = tk()

    def write(self, pil: Image):
        img = ImageTk.PhotoImage(pil)
        panel = Label(self.tk, image = img)
        panel.pack(side = "bottom", fill = "both", expand = "yes")
        self.tk.update_idletasks()
        self.tk.update()

    def close(self):
        self.tk.destroy()


class RenderSave(RenderIO):
    def __init__(self, img_save_dir: Path = Path("rewards")):
        self.img_save_dir = img_save_dir
        self.count = 0

    def _img_path(self):
        return self.img_save_dir / "render_{:%04d}.png".format(self.count)

    def write(self, pil: Image, count: Optional[int] = None):
        count = count or self.count
        pil.save(str(self._img_path))
        self.count += 1

    def close(self):
        self.count = 0


class GymImgEnv(Problem):
    def __init__(self, args, renderio: Callable[[], RenderIO] = RenderSave) -> None:
        self.device = args.device
        # self.ale = atari_py.ALEInterface()
        # self.ale.setInt('random_seed', args.seed)
        # self.ale.setInt('max_num_frames', args.max_episode_length)
        # self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
        # self.ale.setInt('frame_skip', 0)
        # self.ale.setBool('color_averaging', False)
        # self.ale.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options
        self.ale = gym.make(args.game + "-v0")
        actions = self.ale.getMinimalActionSet()
        self.actions = dict((i, e) for i, e in zip(range(len(actions)), actions))
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = False  # Used to check if resetting only from loss of life
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque(
            [], maxlen=args.history_length
        )   # type: Sequence
        self.training = True  # Consistent with model training mode
        self.renderio = renderio()

    def _get_state(self):
        state = Image.fromarray(
            self.ale.getScreenGrayscale().squeeze()
        ).resize((84, 84), resample=Image.BILINEAR)
        return torch.tensor(np.asarray(state),
                            dtype=torch.float32, device=self.device).div_(255)

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(84, 84, device=self.device))

    def reset(self):
        if self.life_termination:
            self.life_termination = False  # Reset flag
            self.ale.act(0)  # Use a no-op after loss of life
        else:
            # Reset internals
            self._reset_buffer()
            self.ale.reset_game()
            # Perform up to 30 random no-ops before starting
            for _ in range(random.randrange(30)):
                self.ale.act(0)  # Assumes raw action 0 is always no-op
                if self.ale.game_over():
                    self.ale.reset_game()
        # Process and return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        self.lives = self.ale.lives()
        return torch.stack(list(self.state_buffer), 0)

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, 84, 84, device=self.device)
        reward, done = 0, False
        for t in range(4):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break
        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)
        # Detect loss of life as terminal in training mode
        if self.training:
            lives = self.ale.lives()
            # Lives > 0 for Q*bert
            if lives < self.lives and lives > 0:
                # Only set flag when not truly done
                self.life_termination = not done
                done = True
            self.lives = lives
        # Return state, reward, done
        return torch.stack(list(self.state_buffer), 0), reward, done

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return len(self.actions)

    def render(self):
        pil = Image.fromarray(self.ale.getScreenRGB()[:, :, ::-1])
        self.renderio.write(pil)

    def close(self):
        self.renderio.close()
