# ♟️ Shadow-Code

**Shadow-Code** is a feature-rich chess game that supports both local AI battles and online multiplayer. Play against a Deep Reinforcement Learning (DRL) agent trained from scratch or connect with friends over a real-time socket-based network—all with an elegant Pygame-based interface and animated visuals.

---

## 🚀 Features

- ✅ **Local Play vs AI**  
  Play against a trained Deep Q-Network (DQN)-based DRL agent.

- 🌐 **Online Multiplayer**  
  Host or join real-time matches using TCP sockets.

- 🎞️ **Smooth Animations**  
  Enjoy sliding piece animations for a polished experience.

- 👑 **Promotion Interface**  
  Promote pawns with a clean graphical selection.

- 🔊 **Sound Effects**  
  Custom sounds for moves, captures, and promotions.

- 📋 **Move History Sidebar**  
  Track your game progress with a live move list.

---

## 🛠 Installation

### 1. Clone the repository

```bash
git clone https://github.com/MrTBK/Shadow-Code.git
cd Shadow-Code
```

### 2. Install dependencies

Install the required packages via pip:

```bash
pip install -r requirements.txt
```

> **Note**: Requires Python 3.8+ and PyTorch.

---

## 🧠 Running the Game

To start the game:

```bash
python main.py
```

---

## 🎮 Modes of Play

### 🧑‍💻 Local Play (vs DRL)

- Select **Local** from the menu.
- Play against a Deep Q-Learning agent trained on legal chess states.

### 🌐 Online Multiplayer

#### Host a Game

- Select **Online** → **Host**.
- Wait for an opponent to connect.
- Game starts automatically after connection.

#### Join a Game

- Select **Online** → **Join**.
- Enter the IP address of the host machine.
- Connect and start playing.

---

## 📁 Project Structure

```
Shadow-Code/
├── assets/               # Piece images & sound files
├── dqn.py                # DQN agent definition and utilities
├── game.py               # Game logic and rendering
├── interface.py          # UI handling (board, promotion UI, sounds)
├── server.py             # Socket-based networking
├── main.py               # Entry point
├── README.md
└── requirements.txt
```

---

## 🧠 DRL Agent Training

Our AI uses a Deep Q-Network with a flat board tensor input. Training code and logs are included in the project for reproducibility. If no `dqn_checkpoint.pt` exists, training can be resumed automatically.

---

