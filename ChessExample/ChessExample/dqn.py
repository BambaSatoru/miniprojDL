import chess
import torch
import torch.nn as nn
import torch.optim as optim
import random
import sys
import os
import matplotlib.pyplot as plt

# === Constantes ===
STATE_DIM = 12 * 8 * 8
ACTION_DIM = 4672  # Nombre maximal de coups en UCI possibles

# === Réseau de neurones DQN ===
class DQN(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# === Encodage des coups ===
_MOVE_LIST = sorted({m.uci() for _ in range(20) for m in chess.Board().legal_moves})
_MOVE_TO_IDX = {uci: idx for idx, uci in enumerate(_MOVE_LIST)}
_IDX_TO_MOVE = {idx: uci for uci, idx in _MOVE_TO_IDX.items()}

def move_to_idx(move: chess.Move) -> int:
    return _MOVE_TO_IDX.get(move.uci(), 0)

def idx_to_move(idx: int) -> str:
    return _IDX_TO_MOVE.get(idx, "0000")  # move null fallback

# === Encodage d'un échiquier en tenseur ===
def board_to_tensor(board: chess.Board) -> torch.Tensor:
    tensor = torch.zeros((12, 8, 8), dtype=torch.float32)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            plane = (piece.piece_type - 1) + (0 if piece.color else 6)
            row, col = divmod(sq, 8)
            tensor[plane, 7 - row, col] = 1.0
    return tensor.view(1, -1)

# === Environnement simplifié d'échecs ===
class ChessEnv:
    def __init__(self):
        self.board = chess.Board()

    def reset(self):
        self.board.reset()
        return self.board

    def legal_moves(self):
        return list(self.board.legal_moves)

    def step(self, move):
        self.board.push(move)
        done = self.board.is_game_over()
        result = self.board.result() if done else None
        reward = 1 if result == '1-0' else -1 if result == '0-1' else 0
        return self.board, reward, done

# === Chargement d’un modèle sauvegardé ===
def load_dqn(path: str = "dqn_checkpoint.pt") -> DQN:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file '{path}' not found.")
    checkpoint = torch.load(path, map_location="cpu")
    model = DQN()
    state_dict = checkpoint.get('model', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# === Boucle principale d'entraînement ===
if __name__ == "__main__":
    checkpoint_path = "dqn_checkpoint.pt"
    policy_net = DQN()
    target_net = DQN()
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    episode = 0
    rewards = []

    # Chargement éventuel d’un checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_state = checkpoint.get('model', checkpoint)
        policy_net.load_state_dict(model_state)
        target_net.load_state_dict(model_state)
        episode = checkpoint.get('episode', 0)
        rewards = checkpoint.get('rewards', [])
        print(f"[INFO] Reprise à l'épisode {episode}")
    else:
        target_net.load_state_dict(policy_net.state_dict())
        print("[INFO] Démarrage d'un nouvel entraînement.")

    # Préparation de l'affichage du graphique
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Progression de l'entraînement DQN")
    ax.set_xlabel("Épisodes")
    ax.set_ylabel("Récompense totale")
    ax.grid(True, linestyle='--', linewidth=0.5)
    line, = ax.plot([], [], 'o-', label="Reward")
    ax.legend()

    # === Entraînement (aléatoire pour l’instant) ===
    try:
        while True:
            episode += 1
            board = env.reset()
            done = False
            total_reward = 0

            while not done:
                move = random.choice(env.legal_moves())  # à remplacer par une action du DQN plus tard
                board, reward, done = env.step(move)
                total_reward += reward

            rewards.append(total_reward)

            # Mise à jour du graphique
            x_data = list(range(1, episode + 1))
            line.set_data(x_data, rewards)
            ax.set_xlim(1, max(10, episode))
            ax.set_ylim(min(rewards) - 1, max(rewards) + 1)
            fig.canvas.draw()
            fig.canvas.flush_events()

            # Sauvegarde périodique
            if episode % 10 == 0:
                print(f"Épisode {episode} | Récompense: {total_reward}")
                torch.save({
                    'episode': episode,
                    'model': policy_net.state_dict(),
                    'rewards': rewards
                }, checkpoint_path)

    except KeyboardInterrupt:
        print(f"[INFO] Arrêt manuel à l’épisode {episode}")
        torch.save({
            'episode': episode,
            'model': policy_net.state_dict(),
            'rewards': rewards
        }, checkpoint_path)
        sys.exit()
