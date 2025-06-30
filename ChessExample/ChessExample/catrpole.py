import gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os

# === Hyperparamètres ===
ENV_NAME = "CartPole-v1"
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
BUFFER_SIZE = 100_000
MIN_REPLAY_SIZE = 1_000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 10_000
TARGET_UPDATE_FREQ = 1000
MAX_EPISODES = 500
MODEL_PATH = "cartpole_dqn.pth"

# === Réseau de neurones DQN ===
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# === Fonction de sélection d’action epsilon-greedy ===
def select_action(state, policy_net, epsilon, env):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = policy_net(state_tensor)
            return q_values.argmax().item()

# === Entraînement ===
def train():
    env = gym.make(ENV_NAME)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = deque(maxlen=BUFFER_SIZE)

    # Initialisation du buffer
    state = env.reset()
    for _ in range(MIN_REPLAY_SIZE):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state if not done else env.reset()

    # Variables de suivi
    episode_rewards = []
    steps_done = 0
    epsilon = EPSILON_START

    for episode in range(MAX_EPISODES):
        state = env.reset()
        total_reward = 0

        while True:
            steps_done += 1
            epsilon = max(EPSILON_END, EPSILON_START - steps_done / EPSILON_DECAY)
            action = select_action(state, policy_net, epsilon, env)
            next_state, reward, done, _ = env.step(action)

            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            # Échantillonnage du batch
            batch = random.sample(replay_buffer, BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones).unsqueeze(1)

            # Q-values cibles
            with torch.no_grad():
                next_q_values = target_net(next_states).max(1, keepdim=True)[0]
                target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

            # Q-values actuelles
            current_q_values = policy_net(states).gather(1, actions)

            # Calcul de la perte et rétropropagation
            loss = nn.functional.mse_loss(current_q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Mise à jour du réseau cible
            if steps_done % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        episode_rewards.append(total_reward)
        print(f"Épisode {episode+1} — Récompense : {total_reward:.2f} — ε : {epsilon:.3f}")

        # Sauvegarde du modèle toutes les 50 épisodes
        if (episode + 1) % 50 == 0:
            torch.save(policy_net.state_dict(), MODEL_PATH)
            print(f"[INFO] Modèle sauvegardé à l’épisode {episode+1}")

    env.close()

# === Exécution du script ===
if __name__ == "__main__":
    train()
