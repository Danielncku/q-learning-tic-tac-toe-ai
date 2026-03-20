

## About
- As i learned CS50 from EDX , i saw professor introduce an easy ai learning project.
- So i start to do and learn by myself
This project implements a Tic-Tac-Toe AI using Q-learning. The AI learns by playing against itself many times. At the beginning, it plays randomly, but over time it learns better strategies.

##  Key Learning Ideas
- Q-learning (reinforcement learning)
- State, action, reward
- Exploration vs exploitation
- Self-play training

## How It Works
Training loop:
play game → get reward → update Q-table → repeat

The AI stores knowledge in a Q-table:
(state, action) → score  
Higher score means a better move.

## Important Fix
Originally, I used one Q-table for both players, which caused unstable learning.  
I fixed this by:
- Using two agents (Q_X and Q_O)
- Giving rewards based on each player’s perspective  
This significantly improved performance.

##  Features
- Self-training AI (~50,000 games)
- Two separate agents (X and O)
- Simple GUI (Tkinter)
- Save/load trained model

##  Result
- AI rarely loses  
- Most games end in draw (near optimal play)

##  Future Work
- Deep Q-Learning (DQN)
- Show Q-values (explain AI decisions)
- Web version (Streamlit)

