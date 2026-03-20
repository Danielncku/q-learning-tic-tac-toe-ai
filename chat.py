import tkinter as tk
from tkinter import messagebox
import random
import pickle
import os
from collections import deque

# =========================================
# 1. Q-learning settings
# =========================================
ALPHA = 0.2          # learning rate
GAMMA = 0.95         # discount factor
EPSILON_START = 1.0  # initial exploration
EPSILON_END = 0.05   # final exploration
EPISODES = 50000     # training games

Q_X = {}  # Q-table for player X
Q_O = {}  # Q-table for player O


# =========================================
# 2. Game logic
# =========================================
WIN_LINES = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],   # rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8],   # cols
    [0, 4, 8], [2, 4, 6]               # diagonals
]


def check_winner(board):
    """Return 'X', 'O', 'Draw', or None."""
    for line in WIN_LINES:
        a, b, c = line
        if board[a] != "" and board[a] == board[b] == board[c]:
            return board[a]
    if "" not in board:
        return "Draw"
    return None


def get_valid_moves(board):
    """Return indices of empty cells."""
    return [i for i, cell in enumerate(board) if cell == ""]


def board_to_state(board):
    """Convert board list to compact string state."""
    return "".join(cell if cell != "" else "_" for cell in board)


def get_q_table(player):
    """Return the correct Q-table for the given player."""
    return Q_X if player == "X" else Q_O


def get_q_value(player, state, action):
    """Read Q-value for (state, action). Default to 0."""
    q_table = get_q_table(player)
    return q_table.get((state, action), 0.0)


def choose_action(player, state, actions, epsilon):
    """
    Epsilon-greedy policy.
    With probability epsilon -> explore
    Else -> choose best known action
    """
    if not actions:
        return None

    if random.random() < epsilon:
        return random.choice(actions)

    q_values = [get_q_value(player, state, a) for a in actions]
    max_q = max(q_values)

    # If multiple actions tie for best, choose randomly among them
    best_actions = [a for a in actions if get_q_value(player, state, a) == max_q]
    return random.choice(best_actions)


def update_q(player, state, action, reward, next_state, next_actions):
    """Standard Q-learning update."""
    q_table = get_q_table(player)
    old_q = q_table.get((state, action), 0.0)

    if next_actions:
        future_q = max(get_q_value(player, next_state, a) for a in next_actions)
    else:
        future_q = 0.0

    new_q = old_q + ALPHA * (reward + GAMMA * future_q - old_q)
    q_table[(state, action)] = new_q


def epsilon_by_episode(episode, total_episodes):
    """Linearly decay epsilon from start to end."""
    ratio = episode / total_episodes
    return EPSILON_START - (EPSILON_START - EPSILON_END) * ratio


# =========================================
# 3. Reward design
# =========================================
def terminal_reward(winner, player):
    """
    Reward from the current player's perspective.
    This is the key fix.
    """
    if winner == player:
        return 1.0
    if winner == "Draw":
        return 0.2
    return -1.0


# =========================================
# 4. Training
# =========================================
def train_ai(episodes=EPISODES, verbose_every=5000):
    """
    Train X and O separately.
    Each side learns from its own perspective.
    """
    history = deque(maxlen=1000)

    for episode in range(episodes):
        board = [""] * 9
        current_player = "X"
        epsilon = epsilon_by_episode(episode, episodes)

        # Save the last move info for each player so we can give final reward correctly
        last_state = {"X": None, "O": None}
        last_action = {"X": None, "O": None}

        while True:
            state = board_to_state(board)
            actions = get_valid_moves(board)

            action = choose_action(current_player, state, actions, epsilon)
            if action is None:
                break

            # Save current move for later update
            last_state[current_player] = state
            last_action[current_player] = action

            # Apply move
            board[action] = current_player
            next_state = board_to_state(board)
            winner = check_winner(board)

            if winner is not None:
                # Current player gets terminal reward
                update_q(
                    current_player,
                    state,
                    action,
                    terminal_reward(winner, current_player),
                    next_state,
                    []
                )

                # Other player also gets terminal reward for its last move
                other = "O" if current_player == "X" else "X"
                if last_state[other] is not None and last_action[other] is not None:
                    update_q(
                        other,
                        last_state[other],
                        last_action[other],
                        terminal_reward(winner, other),
                        next_state,
                        []
                    )

                history.append(winner)
                break

            # Non-terminal move: reward 0 for now, but future value matters
            next_actions = get_valid_moves(board)
            update_q(current_player, state, action, 0.0, next_state, next_actions)

            # Switch player
            current_player = "O" if current_player == "X" else "X"

        if (episode + 1) % verbose_every == 0:
            x_wins = sum(1 for r in history if r == "X")
            o_wins = sum(1 for r in history if r == "O")
            draws = sum(1 for r in history if r == "Draw")
            total = len(history) if history else 1

            print(f"Episode {episode + 1}/{episodes}")
            print(f"Recent 1000 games -> X win: {x_wins/total:.2%}, O win: {o_wins/total:.2%}, Draw: {draws/total:.2%}")
            print(f"Current epsilon: {epsilon:.3f}")
            print("-" * 50)

    print("Training complete!")


def save_qtables(filename="qtables_tictactoe.pkl"):
    with open(filename, "wb") as f:
        pickle.dump({"Q_X": Q_X, "Q_O": Q_O}, f)


def load_qtables(filename="qtables_tictactoe.pkl"):
    global Q_X, Q_O
    with open(filename, "rb") as f:
        data = pickle.load(f)
        Q_X = data["Q_X"]
        Q_O = data["Q_O"]


# =========================================
# 5. GUI
# =========================================
class TicTacToeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic-Tac-Toe AI (Improved Q-learning)")

        self.board = [""] * 9
        self.human = "X"
        self.ai = "O"
        self.game_over = False

        self.status_label = tk.Label(root, text="You are X. Your turn.", font=("Arial", 14))
        self.status_label.grid(row=0, column=0, columnspan=3, pady=10)

        self.buttons = []
        for i in range(9):
            btn = tk.Button(
                root,
                text="",
                width=6,
                height=3,
                font=("Arial", 24),
                command=lambda idx=i: self.player_move(idx)
            )
            btn.grid(row=1 + i // 3, column=i % 3)
            self.buttons.append(btn)

        self.reset_button = tk.Button(root, text="Reset", font=("Arial", 12), command=self.reset_game)
        self.reset_button.grid(row=4, column=0, columnspan=3, pady=10)

    def refresh_board(self):
        for i in range(9):
            self.buttons[i]["text"] = self.board[i]

    def player_move(self, idx):
        if self.game_over:
            return
        if self.board[idx] != "":
            return

        self.board[idx] = self.human
        self.refresh_board()

        result = check_winner(self.board)
        if result:
            self.end_game(result)
            return

        self.status_label.config(text="AI is thinking...")
        self.root.after(250, self.ai_move)

    def ai_move(self):
        if self.game_over:
            return

        state = board_to_state(self.board)
        actions = get_valid_moves(self.board)
        if not actions:
            self.end_game("Draw")
            return

        # During actual play, epsilon = 0 (pure exploitation)
        move = choose_action(self.ai, state, actions, epsilon=0.0)
        if move is None:
            self.end_game("Draw")
            return

        self.board[move] = self.ai
        self.refresh_board()

        result = check_winner(self.board)
        if result:
            self.end_game(result)
        else:
            self.status_label.config(text="Your turn.")

    def end_game(self, result):
        self.game_over = True
        if result == "Draw":
            self.status_label.config(text="Draw!")
            messagebox.showinfo("Game Over", "Draw!")
        elif result == self.human:
            self.status_label.config(text="You win!")
            messagebox.showinfo("Game Over", "You win!")
        else:
            self.status_label.config(text="AI wins!")
            messagebox.showinfo("Game Over", "AI wins!")

    def reset_game(self):
        self.board = [""] * 9
        self.game_over = False
        self.refresh_board()
        self.status_label.config(text="You are X. Your turn.")


# =========================================
# 6. Main
# =========================================
if __name__ == "__main__":
    model_file = "qtables_tictactoe.pkl"

    if os.path.exists(model_file):
        load_qtables(model_file)
        print("Loaded trained Q-tables.")
    else:
        print("No saved model found. Start training...")
        train_ai(EPISODES)
        save_qtables(model_file)
        print("Model saved.")

    root = tk.Tk()
    app = TicTacToeGUI(root)
    root.mainloop()