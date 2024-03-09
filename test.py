from abstract_agent import Agent
from minimax_agent import MinimaxAgent
from environment import Environment
from board import Board
from game_runner import GameRunner
from Q_learning_agent import Q_learning_Agent
from deep_qlearning import QNetwork
import matplotlib.pyplot as plt
import torch

side = 10
size = int(side**2/2) # The size of the board
move_length = 2 # The length of a move. We do not consider longer moves for now
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


b_wins = []
for i in range(75):
    gr = GameRunner()
    q_network = QNetwork(5*size, size**move_length, nn_l1=128, nn_l2=128).to(device)
    q_network_ = torch.load(f"naive_q_network{i}.pth").to(device)
    agent_A = Agent()
    agent_B = Q_learning_Agent(q_network_)
    # gr.run_and_show(agent_A, agent_B, console = True, gif = True)
    print(f'Game {i}') 
    win_A, win_B, mean_t = gr.compare_agents(agent_A, agent_B, n=1000)
    print(f'Agent A won {win_A} times')
    print(f'Agent B won {win_B} times')
    print(f'Mean time of a game: {mean_t}')
    print()
    b_wins.append(win_B)

plt.plot(b_wins)
plt.xlabel('Training epoch')
plt.ylabel('Victory rate of Q-network-agent over random agent')
plt.legend()
plt.show()