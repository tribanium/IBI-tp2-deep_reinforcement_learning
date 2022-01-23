# -*- coding: utf-8 -*-


import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import time
from skimage.transform import resize
from skimage.color import rgb2gray
from math import floor

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, num_frames, h, w, num_outputs):
        super(DQN, self).__init__()
        # 3 couches de convolution chacune suivie d'une batch normalization
        # filtres de taille 5 pixels, pas de 2
        # 16 filtres pour la première couche
        # 32 filtres pour la deuxième
        # 64 pour la troisième
        # Finir par une couche fully connected
        self.conv1 = nn.Conv2d(
            in_channels=num_frames, out_channels=16, kernel_size=5, stride=2
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc4 = nn.Linear(in_features=7 * 7 * 64, out_features=num_outputs)

    def forward(self, x):
        # Calcul de la passe avant :
        # Fonction d'activation relu pour les couches cachées
        # Fonction d'activation linéaire sur la couche de sortie

        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.fc4(torch.flatten(x, start_dim=1))

        return x


class Agent:
    def __init__(self, env):

        self.env = env
        self.batch_size = 16
        self.gamma = 0.999
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 200
        self.target_update = 10
        self.num_episodes = 200
        self.num_frames = 4

        self.im_height = 84
        self.im_width = 84

        self.n_actions = env.action_space.n
        self.actions_meaning = env.get_action_meanings()
        print(self.actions_meaning)
        # No-op, fire, right, left

        self.episode_durations = []

        self.policy_net = DQN(
            self.num_frames, self.im_height, self.im_width, self.n_actions
        ).to(device)
        self.target_net = DQN(
            self.num_frames, self.im_height, self.im_width, self.n_actions
        ).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)

        self.steps_done = 0

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        self.steps_done += 1
        if sample > eps_threshold:
            # Calcul et renvoi de l'action fournie par le réseau
            with torch.no_grad():
                return torch.tensor([[self.policy_net(state).argmax()]], device=device)
        else:
            # Calcul et renvoi d'une action choisie aléatoirement
            return torch.tensor(
                [[random.randrange(self.n_actions)]],
                device=device,
                dtype=torch.long,
            )

    def plot_durations(self):
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.title("Training...")
        plt.xlabel("Episode")
        plt.ylabel("Duration")
        plt.plot(durations_t.numpy())

        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

    def process(self, state):
        state = (
            resize(rgb2gray(state), (self.im_height, self.im_width), mode="reflect")
            * 255
        )
        state = state[np.newaxis, np.newaxis, :, :]
        return torch.tensor(state, device=device, dtype=torch.float)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        print(action_batch)
        reward_batch = torch.cat(batch.reward)

        # Calcul de Q(s_t,a) : Q pour l'état courant
        prediction = self.policy_net(state_batch)
        # print(prediction)
        state_action_values = prediction.gather(1, action_batch)
        # Le modèle calcule la Q-valeur de l'état pour toutes les actions, puis on sélectionne la Q-valeur de l'action qui a été prise.
        # Cela correspond à l'action qui aurait été prise par le policy net (avec politique epsilon-greedy).

        # Calcul de Q pour l'état suivant avec le target net
        next_state_values = torch.zeros(
            self.batch_size, device=device
        )  # Si on a un final state, la value vaut 0
        next_state_values[non_final_mask] = (
            self.target_net(non_final_next_states).max(1)[0].detach()
        )
        # max(1)[0] sélectionne la best reward
        # detach permet de ne pas actualiser les gradients avec une backprop

        # Calcul de Q future attendue cumulée
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Calcul de la fonction de perte de Huber
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimisation du modèle
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train_policy_model(self):
        for i_episode in range(self.num_episodes):
            state = self.env.reset()
            state = self.process(state)
            print(f"Episode : {i_episode}/{self.num_episodes}")

            for t in count():
                while state.size()[1] < self.num_frames:
                    action = 1  # Fire

                    new_frame, reward, done, _ = env.step(action)
                    new_frame = self.process(new_frame)

                    state = torch.cat([state, new_frame], 1)

                action = self.select_action(state)
                new_frame, reward, done, _ = self.env.step(action.item())
                new_frame = self.process(new_frame)

                if done:
                    new_state = None
                else:
                    new_state = torch.cat([state, new_frame], 1)
                    new_state = new_state[:, 1:, :, :]

                reward = torch.tensor([reward], device=device, dtype=torch.float)

                self.memory.push(state, action, new_state, reward)

                state = new_state

                self.optimize_model()

                if done:
                    self.episode_durations.append(t + 1)
                    self.plot_durations()
                    break

            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                self.save_model()

        self.save_model()
        print("Training completed")
        plt.show()

    def save_model(self):
        torch.save(self.policy_net.state_dict(), "./dqn_model")

    def load_model(self):
        self.policy_net.load_state_dict(torch.load("./dqn_model", map_location=device))

    def test(self):
        print("Testing model:")
        for i_episode in range(20):
            print(f"Test episode: {i_episode}/{20}")

            state = self.env.reset()
            state = self.process(state)
            fig = plt.figure(3)
            ax = fig.gca()
            h = ax.imshow(self.env.render(mode="rgb_array"))

            for t in count():
                h.set_data(self.env.render(mode="rgb_array"))
                plt.draw()
                plt.pause(1e-13)
                time.sleep(0.017)

                # Sélection d'une action appliquée à l'environnement
                # et mise à jour de l'état
                while state.size()[1] < self.num_frames:
                    action = 1  # Fire

                    new_frame, reward, done, _ = env.step(action)
                    new_frame = self.process(new_frame)

                    state = torch.cat([state, new_frame], 1)

                action = self.select_action(state)
                new_frame, reward, done, _ = self.env.step(action.item())
                new_frame = self.process(new_frame)

                if done:
                    new_state = None
                else:
                    new_state = torch.cat([state, new_frame], 1)
                    new_state = new_state[:, 1:, :, :]

                reward = torch.tensor([reward], device=device, dtype=torch.float)

                state = new_state

                if done:
                    self.episode_durations.append(t + 1)
                    self.plot_durations()
                    break

        print("Testing completed")


if __name__ == "__main__":

    # set up matplotlib
    is_ipython = "inline" in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    env = gym.make("BreakoutDeterministic-v4")
    env.reset()

    agent = Agent(env)

    try:
        # Training phase
        agent.train_policy_model()
    except KeyboardInterrupt:
        pass

    finally:
        # Testing phase
        agent.load_model()
        agent.test()

        env.close()
