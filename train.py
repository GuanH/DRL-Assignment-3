from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchrl.data import PrioritizedReplayBuffer, ListStorage
from tqdm import tqdm



env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

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

        self.cls1 = nn.Sequential(
                nn.Linear(147456, 1))

        self.cls2 = nn.Sequential(
                nn.Linear(147456, 12))


    # x(B, 4, H, W)
    def forward(self, x):
        B = x.shape[0]
        x = self.convs(x)
        x = x.reshape((B, -1))

        v = self.cls1(x)
        a = self.cls2(x)
        return v*1000, a
# model = QNet()
# torch.save(model.state_dict(), "model_test.pth")
# exit()

def softmax(x):
    x = np.exp(x - np.max(x))
    return x / np.sum(x)

class PER:
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.alpha = max(alpha, 1e-5)
        self.data = np.empty(capacity, dtype=np.int32)
        self.delta = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, x):
        self.data[self.pos] = x
        self.delta[self.pos] = 100
        p = self.pos
        self.pos = (self.pos+1)%self.capacity
        self.size = min(self.size+1, self.capacity)
        return p

    def sample(self, n):
        d = self.delta ** self.alpha
        p = d / np.sum(d)
        idx = np.random.choice(self.capacity, n, p=p)
        w = 1 / (p[idx] * self.size)
        return self.data[idx], idx, w

    def set_delta(self, idx, d):
        self.delta[idx] = d


class ReplayBuffer:
    def __init__(self, capacity, state_size):
        self.state_buffer_size = capacity + 64
        self.state_buffer = np.empty((self.state_buffer_size, *state_size), dtype=np.float32)
        self.state_buffer_beg = 0
        self.state_buffer_pos = np.array([-3,-2,-1,0])

        self.cur_state_buffer = np.empty((capacity, 4), dtype=np.int32)
        self.next_state_buffer = np.empty((capacity, 4), dtype=np.int32)

        self.action_buffer = np.empty(capacity, dtype=np.int64)
        self.reward_buffer = np.empty(capacity, dtype=np.float32)
        self.done_buffer = np.empty(capacity, dtype=np.float32)
        self.per = PER(capacity, 0.4)
        # self.valid_ids = np.zeros(capacity, dtype=np.float32)

        self.pos = 0
        self.capacity = capacity
        self.size = 0

        self.last_pos = -1

    def add(self, state, action, reward, done, agent):

        cur_id = (self.state_buffer_beg + self.state_buffer_pos[-1]) % self.state_buffer_size
        self.state_buffer[cur_id] = state

        self.cur_state_buffer[self.pos] = (self.state_buffer_beg + np.maximum(self.state_buffer_pos, 0)) % self.state_buffer_size
        self.next_state_buffer[self.pos] = (self.state_buffer_beg + np.maximum(self.state_buffer_pos + 1, 0)) % self.state_buffer_size
        self.state_buffer_pos = (self.state_buffer_pos + 1) % self.state_buffer_size

        self.action_buffer[self.pos] = action
        self.reward_buffer[self.pos] = reward
        self.done_buffer[self.pos] = done

        if self.last_pos != -1:
            p = self.per.add(self.last_pos)
            if self.done_buffer[self.last_pos]:
                self.per.set_delta(p, 300)
            # self.valid_ids[self.last_pos] = 1
        # self.valid_ids[self.pos] = 0
        self.last_pos = self.pos

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n):
        idx, per_idx, w = self.per.sample(n)
        # idx = np.random.choice(self.capacity, n, p=self.valid_ids/np.sum(self.valid_ids))
        states = self.state_buffer[self.cur_state_buffer[idx]]
        next_states = self.state_buffer[self.next_state_buffer[idx]]
        return states, next_states, self.action_buffer[idx], self.reward_buffer[idx], self.done_buffer[idx], per_idx, w
        # return states, next_states, self.action_buffer[idx], self.reward_buffer[idx], self.done_buffer[idx]

    def reset(self):
        self.state_buffer_beg = (self.state_buffer_beg + self.state_buffer_pos[-1] + 1) % self.state_buffer_size
        self.state_buffer_pos = np.array([-3,-2,-1,0])

class DQN:
    def __init__(self, state_size, action_size, device):
        self.action_size = action_size
        self.device = device
        self.learning_qnet = QNet()
        self.target_qnet = QNet()
        self.learning_qnet.load_state_dict(torch.load("model1.pth", weights_only=True))
        self.target_qnet.load_state_dict(torch.load("model1.pth", weights_only=True))
        self.opt1 = torch.optim.Adam(self.learning_qnet.parameters(), lr=1e-4)
        self.opt2 = torch.optim.Adam(self.target_qnet.parameters(), lr=1e-4)
        self.sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt1, 8000, 1e-7)
        self.sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt2, 8000, 1e-7)
        self.replay_buffer = ReplayBuffer(100000, state_size)

        self.learning_qnet.to(device)
        self.target_qnet.to(device)
        self.step = 0
        self.batch_size = 64
        self.acc_grad_step = 2

        self.clip_norm = 1


    def get_action(self, state, eps):
        if np.random.random() <= eps:
            return np.random.randint(0, self.action_size)
        with torch.no_grad():
            state = torch.tensor(state[None,:], dtype=torch.float32, device=self.device)
            v, a = self.learning_qnet(state)
        return torch.argmax(a, dim=-1).item()


    def train(self):
        if self.replay_buffer.size < self.batch_size:
            return 0
        states, next_states, actions, rewards, dones, per_idx, w = self.replay_buffer.sample(self.batch_size)
        # states, next_states, actions, rewards, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        w = torch.tensor(w, dtype=torch.float32, device=self.device)
       #  

        with torch.no_grad():
            v, a = self.learning_qnet(next_states)
            next_actions = a.argmax(dim=1)
            v, a = self.target_qnet(next_states)
            q = v + a - a.mean(dim=1, keepdim=True)
            target = rewards + 0.99 * q.gather(1, next_actions.unsqueeze(-1)).squeeze(-1) * (1 - dones)

        v, a = self.learning_qnet(states)
        q = v + a - a.mean(dim=1, keepdim=True)
        delta = q.gather(1, actions.unsqueeze(-1)).squeeze(-1) - target

        self.replay_buffer.per.set_delta(per_idx, delta.detach().abs().cpu().numpy())
        loss = ((delta**2)*w).mean() / self.acc_grad_step
        # loss = (delta**2).mean() / self.acc_grad_step
        ret_loss = loss.item()
        loss.backward()
        with torch.no_grad():
            v, a = self.target_qnet(next_states)
            next_actions = a.argmax(dim=1)
            v, a = self.learning_qnet(next_states)
            q = v + a - a.mean(dim=1, keepdim=True)
            target = rewards + 0.99 * q.gather(1, next_actions.unsqueeze(-1)).squeeze(-1) * (1 - dones)

        v, a = self.target_qnet(states)
        q = v + a - a.mean(dim=1, keepdim=True)
        delta = q.gather(1, actions.unsqueeze(-1)).squeeze(-1) - target
        loss = ((delta**2)*w).mean() / self.acc_grad_step
        loss = (delta**2).mean() / self.acc_grad_step
        loss.backward()


        self.step += 1
        if self.step % self.acc_grad_step == 0:
            nn.utils.clip_grad_norm_(self.learning_qnet.parameters(), self.clip_norm)
            self.opt1.step()
            self.opt1.zero_grad()

            nn.utils.clip_grad_norm_(self.target_qnet.parameters(), self.clip_norm)
            self.opt2.step()
            self.opt2.zero_grad()

            self.sch1.step()
            self.sch2.step()
            self.clip_norm = max(self.clip_norm*0.999, 0.1)

        # if self.step % 100 == 99:
        #     for target_param, learning_param in zip(self.target_qnet.parameters(), self.learning_qnet.parameters()):
        #         target_param.data.copy_(learning_param.data)

        return ret_loss



agent = DQN((128,128), env.action_space.n, "cuda")
EPOCH = 100000
reward_hist = []
step_hist = np.ones(10)*4000
step_pos = 0
loss_hist = np.zeros(20)
loss_pos = 0
max_steps = 4000
eps = 0.3
eps_min = 0.05
# eps_dec = (eps_min / eps) ** (1 / 50000)
eps_dec = 0.999
train_step = 1
highest_reward = 1500
torch.set_flush_denormal(True)
for epoch in tqdm(range(EPOCH)):
    state = env.reset()
    state = cv2.resize(state, (128, 128))
    state = (state.astype(np.float32) / 255).mean(-1)
    state_hist = np.empty((16, *state.shape), dtype=np.float32)
    state_hist[:] = state
    cur_state = 0
    done = False
    tot_reward = 0
    acc_reward = 0
    steps = 0
    agent.replay_buffer.reset()

    while not done:
        cur_states = state_hist[[cur_state, cur_state - 4, cur_state - 8, cur_state - 12]]
        cur_eps = eps if steps < np.mean(step_hist)*(np.random.random()*0.4+0.8) else eps * 2
        action = agent.get_action(cur_states, cur_eps)
        # action = agent.get_action(cur_states, eps)
        next_state, reward, done, info = env.step(action)
        tot_reward += reward
        next_state = cv2.resize(next_state, (128, 128)).mean(-1)
        if steps > max_steps or info['life'] == 1:
            done = True

        reward -= 0.1
        acc_reward += reward
        cur_state = (cur_state + 1) % 16
        next_state = next_state.astype(np.float32) / 255
        state_hist[cur_state] = next_state

        if steps % 4 == 0 or done:
            agent.replay_buffer.add(state, action, acc_reward, done, agent)
            acc_reward = 0

        train_step +=1
        state = next_state
        steps += 1
    if tot_reward > highest_reward:
        highest_reward = tot_reward
        torch.save(agent.learning_qnet.state_dict(), "model_b1.pth")
        torch.save(agent.target_qnet.state_dict(), "model_b2.pth")

    if epoch > 5:
        loss = agent.train()
        loss_hist[loss_pos] = loss
        loss_pos = (loss_pos+1)%20
        for _ in tqdm(range(20), leave=False):
            loss = agent.train()
            loss_hist[loss_pos] = loss
            loss_pos = (loss_pos+1)%20
    tqdm.write(f'Reward {tot_reward:.4f} steps {steps} eps {eps:.5f} loss {np.mean(loss_hist)}')


    step_hist[step_pos] = steps
    step_pos = (step_pos+1)%10
    eps = max(eps * eps_dec, eps_min)
    max_steps = min(max_steps+10, 4000)

    if epoch % 100 == 99:
        tqdm.write(f'Epoch:{epoch + 1} Reward:{np.mean(reward_hist):.5f} Step:{np.mean(step_hist):.2f} eps:{eps:.5f}')
        reward_hist = []
        torch.save(agent.learning_qnet.state_dict(), "model1.pth")
        torch.save(agent.target_qnet.state_dict(), "model2.pth")
    reward_hist.append(tot_reward)
