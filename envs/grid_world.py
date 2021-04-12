import numpy as np
import random
from torchvision import datasets, transforms

import utils.exp as exp_utils


class GridWorld(object):
    def __init__(self, env_mode=2, no_wall=False, no_door=False, no_key=True, simple_image=False, noise=None):
        self.env_mode = env_mode
        self.no_wall = no_wall
        self.no_door = no_door
        self.no_key = no_key
        self.noise = noise

        if env_mode == 1:
            # maze of small size 4X4
            pos_man = [0, 0]
            pos_key = [3, 3]
            pos_door = [3, 0]
            self.grid_num = 4
            self.image_size = 32
            self.grid_size = self.image_size // self.grid_num
            self.margin = 0
            if self.no_wall:
                self.blocks = []
            else:
                self.blocks = [20,
                               21,
                               23]

        elif env_mode == 2:
            # maze of large size 8X8
            pos_man = [0, 0]
            pos_key = [0, 7]
            pos_door = [7, 7]
            self.grid_num = 8
            self.image_size = 64
            self.grid_size = self.image_size // self.grid_num
            self.margin = 0
            if self.no_wall:
                self.blocks = []
            else:
                self.blocks = [3, 4,
                               20, 22, 25, 26,
                               32, 35,
                               40, 41, 42, 44, 45, 47,
                               61, 63, 64, 65,
                               71, ]

        self.observation_shape = (3, self.image_size, self.image_size)
        self.state_shape = 2
        # e.g. state= [3, 4], represents the position
        self.action_shape = 4
        # action_set={0,1,2,3},represents UP,DOWN,LEFT,RIGHT
        self.get_key = False
        self.get_door = False
        self.done = False
        if simple_image:
            # No beautification for image
            self.image_man = np.zeros((3, self.grid_size, self.grid_size))
            self.image_man[:,:,:] = [[[1]], [[1]], [[1]]]
            self.image_key = np.zeros((3, self.grid_size, self.grid_size))
            self.image_key[:, :, :] = [[[0]], [[1]], [[0]]]
            self.image_door = np.zeros((3, self.grid_size, self.grid_size))
            self.image_door[:, :, :] = [[[0]], [[0]], [[1]]]
            self.image_wall = np.ones((3, self.grid_size, self.grid_size))
            self.image_wall[:] = 0.5
            self.image_init = np.zeros((3, self.image_size + 2 * self.margin, self.image_size + 2 * self.margin))
            self.image_init[:, self.margin:(-self.margin), self.margin:(-self.margin)] = 1

        else:
            # image with beautification
            self.image_man = np.zeros((3, self.grid_size, self.grid_size))
            self.image_man_block = [[1], [1], [1]]
            self.image_man[:, 2, :] = self.image_man_block
            self.image_man[:, 7, 0:3] = self.image_man_block
            self.image_man[:, 7, 5:8] = self.image_man_block
            self.image_man[:, 2:8, 2] = self.image_man_block
            self.image_man[:, 2:8, 5] = self.image_man_block
            self.image_man[:, 0:6, 3] = self.image_man_block
            self.image_man[:, 0:6, 4] = self.image_man_block

            self.image_key = np.ones((3, self.grid_size, self.grid_size))
            self.image_key_block = [[0], [0.25], [0.5]]
            self.image_key[:, 0, 2:6] = self.image_key_block
            self.image_key[:, 3, 2:6] = self.image_key_block
            self.image_key[:, 1:3, 2] = self.image_key_block
            self.image_key[:, 1:3, 5] = self.image_key_block
            self.image_key[:, 4:8, 2] = self.image_key_block
            self.image_key[:, 4:8, 3] = self.image_key_block
            self.image_key[:, 6:8, 4] = self.image_key_block
            self.image_key[:, 6:8, 5] = self.image_key_block

            self.image_door = np.ones((3, self.grid_size, self.grid_size))
            self.image_door_block = [[0.5], [0.25], [0]]
            self.image_door[:, 0, :] = self.image_door_block
            self.image_door[:, 7, :] = self.image_door_block
            self.image_door[:, :, 0] = self.image_door_block
            self.image_door[:, :, 3] = self.image_door_block
            self.image_door[:, :, 4] = self.image_door_block
            self.image_door[:, :, 7] = self.image_door_block
            self.image_wall = np.ones((3, self.grid_size, self.grid_size))
            self.image_wall[:, 2, :] = 0.5
            self.image_wall[:, 5, :] = 0.5
            self.image_wall[:, 0:2, 4] = 0.5
            self.image_wall[:, 5:, 4] = 0.5
            self.image_wall[:, 2:5, 0] = 0.5
            self.image_init = np.zeros((3, self.image_size + 2 * self.margin, self.image_size + 2 * self.margin))
            self.image_init[:, self.margin:(-self.margin), self.margin:(-self.margin)] = 1

        self.action_set = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
        self.length = 0
        self.score = 0

        # convert to ndarray
        pos_man = np.array(pos_man)
        pos_key = np.array(pos_key)
        pos_door = np.array(pos_door)

        self.pos_man_init = pos_man
        self.pos_man = pos_man
        self.pos_key_init = pos_key
        self.pos_key = pos_key
        self.pos_door_init = pos_door
        self.pos_door = pos_door

        for pos in self.blocks:
            x = pos // 10
            y = pos % 10
            self.image_init[:,
            (self.margin + x * self.grid_size):(self.margin + (x + 1) * self.grid_size),
            (self.margin + y * self.grid_size):(
                        self.margin + (y + 1) * self.grid_size)] = self.image_wall

        # print(self.pos_man, self.pos_key, self.pos_door)

    def not_wall_position(self, pos_man):
        if pos_man[0] < 0 or pos_man[0] == self.grid_num or pos_man[1] < 0 or pos_man[1] == self.grid_num:
            return 0
        elif pos_man[0]*10+pos_man[1] in self.blocks:
            return 0
        else:
            return 1

    def action_sample(self):
        return random.randint(0, self.action_shape-1)

    def trans(self, action):
        reward = 0
        if self.not_wall_position(self.pos_man + self.action_set[action]):
            self.pos_man = self.pos_man+self.action_set[action]
            if self.get_key:
                self.pos_key = self.pos_man
            else:
                if (self.no_key):
                    self.get_key = True
                else:
                    if (self.pos_man==self.pos_key).all():
                        self.get_key = True
                        reward += 10
            if self.get_key and (self.pos_man == self.pos_door).all():
                self.get_door = True


        reward += -1
        if self.get_door:
            reward += 100
        self.length = self.length + 1
        self.done = self.get_door or (self.length == 50)
        self.score = self.score + reward
        return reward, self.done, self.score

    def get_ob(self, state=-1):
        observ = self.image_init.copy()
        observ[:, (self.margin + self.pos_man[0] * self.grid_size):(self.margin + (self.pos_man[0] + 1) * self.grid_size),
        (self.margin + self.pos_man[1] * self.grid_size):(self.margin + (self.pos_man[1] + 1) * self.grid_size)] = self.image_man
        if not self.no_key:
            observ[:, (self.margin + self.pos_key[0] * self.grid_size):(self.margin + (self.pos_key[0] + 1) * self.grid_size),
            (self.margin + self.pos_key[1] * self.grid_size):(self.margin + (self.pos_key[1] + 1) * self.grid_size)] = self.image_key
        if not self.no_door:
            observ[:, (self.margin + self.pos_door[0] * self.grid_size):(self.margin + (self.pos_door[0] + 1) * self.grid_size),
            (self.margin + self.pos_door[1] * self.grid_size):(self.margin + (self.pos_door[1] + 1) * self.grid_size)] = self.image_door
        observ = observ.astype(np.float32)

        if self.noise is not None:
            observ = self.add_noise(observ)

        return observ

    def get_state(self):
        return self.pos_man.astype(np.float32)

    def reset(self):
        self.pos_key = self.pos_key_init
        self.pos_door = self.pos_door_init
        self.pos_man = self.pos_man_init

        self.get_key = False
        self.get_door = False
        self.done = False
        self.length = 0
        self.score = 0

        # observ = self.get_ob()
        observ = self.get_state()
        return observ

    def wrong_position(self):
        if not self.no_key and (self.pos_man == self.pos_key).all():
            return True
        elif not self.no_door and (self.pos_man == self.pos_door).all():
            return True
        return False

    def step(self, action):
        reward, done, score = self.trans(action)
        info = self.get_ob()
        observ_next = self.get_state()
        return observ_next, reward, done, info

    def traverse(self):
        """traverse all the grids and return the observation matrix"""
        obs_mat = []
        for i in range(self.grid_num):
            for j in range(self.grid_num):
                self.pos_man[0], self.pos_man[1] = i, j
                obs_mat.append(self.get_ob())

        return np.array(obs_mat)

    def add_noise(self, ob):
        """add noise to the observation"""
        noise_fn = self.noise["handle_fn"]
        noise_args = self.noise["args"]

        return eval('exp_utils.' + noise_fn)(ob, **noise_args)
