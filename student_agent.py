import gym
import cv2
import numpy as np
import torch
import torch.nn as nn



class QNet(torch.nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(4, 64, 7, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 3), nn.ReLU(),
            nn.Conv2d(256, 256, 3), nn.ReLU(),
            nn.Conv2d(256, 256, 3), nn.ReLU(),
        )

        self.cls1 = nn.Sequential(nn.Linear(147456, 1))

        self.cls2 = nn.Sequential(nn.Linear(147456, 12))


    # x(B, 4, H, W)
    def forward(self, x):
        B = x.shape[0]
        x = self.convs(x)
        x = x.reshape((B, -1))

        v = self.cls1(x)
        a = self.cls2(x)
        return v * 1000, a

class DQN:
    def __init__(self, state_size, device):
        self.device = device
        self.learning_qnet = QNet()
        self.learning_qnet.load_state_dict(torch.load("model1.pth", weights_only=False, map_location=torch.device('cpu')))
        self.learning_qnet.to(device)

    def get_action(self, state):
        if np.random.random() <= 0.05:
            return np.random.randint(0, 12)
        with torch.no_grad():
            state = torch.tensor(state[None,:], dtype=torch.float32, device=self.device)
            v, a = self.learning_qnet(state)
        return torch.argmax(a, dim=-1).item()



# Do not modify the input of the 'act' function and the '__init__' function.
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.state_hist = np.empty((16, 128, 128), dtype=np.float32)
        self.first = True
        self.cur_state = 0
        # self.dqn = DQN((128,128), "cuda" if torch.cuda.is_available() else "cpu")
        self.dqn = DQN((128,128), "cpu")

    def act(self, observation):
        if self.first:
            self.first = False
            state = cv2.resize(observation, (128, 128))
            state = (state.astype(np.float32) / 255).mean(-1)
            self.state_hist[:] = state
        else:
            state = cv2.resize(observation, (128, 128))
            state = (state.astype(np.float32) / 255).mean(-1)
            self.state_hist[self.cur_state] = state

        cur_states = self.state_hist[[self.cur_state, self.cur_state - 4, self.cur_state - 8, self.cur_state - 12]]
        action = self.dqn.get_action(cur_states)
        self.cur_state = (self.cur_state + 1) % 16
        return action
