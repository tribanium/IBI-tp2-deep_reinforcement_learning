# -*- coding: utf-8 -*-

import gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

"""
CartPole docs from https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

Observation:
    Type: Box(4)
    Num     Observation               Min                     Max
    0       Cart Position             -2.4                    2.4
    1       Cart Velocity             -Inf                    Inf
    2       Pole Angle                -0.209 rad (-12 deg)    0.209 rad (12 deg)
    3       Pole Angular Velocity     -Inf                    Inf
Actions:
    Type: Discrete(2)
    Num   Action
    0     Push cart to the left
    1     Push cart to the right
    Note: The amount the velocity that is reduced or increased is not
    fixed; it depends on the angle the pole is pointing. This is because
    the center of gravity of the pole increases the amount of energy needed
    to move the cart underneath it
Reward:
    Reward is 1 for every step taken, including the termination step
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    """Permet d'enregistrer les transitions que l'agent observe et de les réutiliser plus tard.
    En prenant un random sample dans la mémoire pour avoir un batch, les transitions formant le batch sont décorrélées."""

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


class QNet(nn.Module):
    """Réseau de neurones utiisé pour le DQN."""

    def __init__(self):
        super(QNet, self).__init__()
        d_in = 4
        d_h = 256
        d_out = 2
        self.input = nn.Linear(d_in, d_h)
        self.hidden = nn.Linear(d_h, d_out)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.input(x))
        x = self.hidden(x)

        return x


class Agent:
    def __init__(self, env):
        self.env = env
        self.batch_size = 128
        self.gamma = 0.999
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 350
        # self.target_update = 50  # Nb of iterations before updating the target net
        self.num_episodes = 360
        self.tau = 0.05

        self.n_actions = env.action_space.n  # 2
        self.episode_durations = []  # Duration <=> reward

        # Mis à jour régulièrement
        self.policy_net = QNet().to(device)
        # Mis à jour de temps en temps
        self.target_net = QNet().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=1e-2)
        self.memory = ReplayMemory(10000)

        self.steps_done = 0

        self.final_average_score = 0

    def select_action(self, state, train=False):
        """Selects an action accordingly to an epsilon greedy policy."""
        if train:
            sample = random.random()
            # Epsilon decreases exponentially over time so that the DQN agent takes a lot of random actions at the beggining and follow its policy at the end.
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
                -1.0 * self.steps_done / self.eps_decay
            )
            self.steps_done += 1
            if sample > eps_threshold:
                # Calcul et renvoi de l'action fournie par le réseau
                # indice du max sur la premiere dimension et on récupère l'indice avec les crochets
                # view permet de remettre le tout sous forme de matrice 1x1
                # on utilise no_grad car on souhaite effectuer une passe avant sans influer sur les poids
                with torch.no_grad():
                    return self.policy_net(state).max(1)[1].view(1, 1)
            else:
                # action choisie aléatoirement
                return torch.tensor(
                    [[random.randrange(self.n_actions)]],
                    device=device,
                    dtype=torch.long,
                )
        else:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)

    def process_state(self, state):
        """Converts state from numpy array to torch tensor."""
        return torch.from_numpy(state).unsqueeze(0).float().to(device)

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
            means_numpy = means.numpy()
            plt.plot(means.numpy())
            self.final_average_score = means_numpy[-1]

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return  # Pas d'optimisation tant qu'on n'a pas un batch assez grand dans la mémoire
        transitions = self.memory.sample(self.batch_size)

        # Converts batch-array of transitions to transition of batch arrays
        # ([state,action,nextstate,reward],[state,action...]) -> (state=[...],action=[...],...)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
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
        reward_batch = torch.cat(batch.reward)

        # VOTRE CODE
        ############
        # Q(s_t,a) : Q for current state
        prediction = self.policy_net(state_batch)
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

    def soft_update(self):
        for param_target, param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            param_target.data.copy_(
                param_target.data * (1.0 - self.tau) + param.data * self.tau
            )

    def train_policy_model(self):
        print("Training model:")
        for i_episode in range(self.num_episodes):
            if i_episode % 50 == 0:
                print(f"episode: {i_episode}/{self.num_episodes}")

            state = self.env.reset()

            for t in count():

                action = self.select_action(self.process_state(state), train=True)
                next_state, reward, done, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=device)

                if done:
                    next_state = None

                self.memory.push(
                    self.process_state(state),
                    action,
                    self.process_state(next_state) if not next_state is None else None,
                    reward,
                )

                state = next_state

                self.optimize_model()

                # Plots the total reward when the agent finally fails
                if done:
                    self.episode_durations.append(t + 1)
                    self.plot_durations()
                    break

            self.soft_update()

        self.save_model()
        print("Training completed")
        plt.show()

    def save_model(self):
        torch.save(self.policy_net.state_dict(), "./qlearning_model")

    def load_model(self):
        self.policy_net.load_state_dict(
            torch.load("./qlearning_model", map_location=device)
        )

    def test(self):
        print("Testing model:")
        for i_episode in range(10):
            print("episode: {}".format(i_episode))

            state = self.env.reset()

            for t in count():
                self.env.render()

                # VOTRE CODE
                ############
                # Sélection d'une action appliquée à l'environnement
                # et mise à jour de l'état
                action = self.select_action(self.process_state(state))
                next_state, reward, done, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=device)

                if done:
                    next_state = None

                state = next_state

                if done or (t > 750):
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

    seed = 42
    env = gym.make("CartPole-v1")
    torch.manual_seed(seed)
    random.seed(seed)
    env.seed(seed)
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

        print(f"Final average score : {agent.final_average_score}")

        env.close()
