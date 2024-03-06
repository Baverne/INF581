from abstract_agent import Agent
from environment import Environment
from board import Board
import matplotlib.pyplot as plt
import numpy as np
import os
import imageio
import shutil


class GameRunner():

    def make_gif(self, j, fps=5):
        # create a list of filenames (adjust the range to match your filenames)
        filenames = [f'temp_gif/plot_{i}.png' for i in range(j+1)]

        # read the image files and save them as a GIF
        images = [imageio.imread(filename) for filename in filenames]
        imageio.mimsave('game.gif', images,format='GIF', fps=fps)

        # delete the temp folder
        shutil.rmtree('temp_gif')

    def run(self, agent_A, agent_B):
        swap = False
        if np.random.rand() < 0.5:
            swap = True
            agent_A, agent_B = agent_B, agent_A
        board = Board()
        t = 0
        while True:
            t+=1
            # Player A
            if board.is_final():
                if swap:
                    return ('A',t)
                else:
                    return ('B',t)
            move_A = agent_A.move(board)
            board.move(move_A)
            # Player B
            board.transpose()
            if board.is_final():
                if swap:
                    return ('B',t)
                else:
                    return ('A',t)
            move_B = agent_B.move(board)
            board.move(move_B)
            board.transpose()


    def run_and_show(self, agent_A, agent_B, console = True, gif = True):
        
        board = Board()
        if gif:
            # create a temp folder
            if not os.path.exists('temp_gif'):
                os.makedirs('temp_gif')

        plt.imshow(board.get_image())
        if gif:
            i = 0
            plt.savefig(f'temp_gif/plot_{i}.png')
        if console:
            print('Game lunching...')
            print('Initial board: ')
            plt.show()
        while True:

            # Player A
            if board.is_final():
                if console:
                    print('Player B wins!')
                if gif:
                    self.make_gif(i)
                return 'A'
            move_A = agent_A.move(board)
            board.move(move_A)
            if console:
                print('Player A moved ', move_A)
            image = board.get_image()
            i += 1
            plt.imshow(image)
            if gif:
                plt.savefig(f'temp_gif/plot_{i}.png')
            if console:
                plt.show()
            
            # Player B
            board.transpose()
            if board.is_final():
                if console:
                    print('Player A wins!')
                if gif:
                    self.make_gif(i)
                return 'B'
            move_B = agent_B.move(board)
            board.move(move_B)
            if console:
                print('Player B moved ', move_B)
            board.transpose()
            image = board.get_image()
            i += 1
            plt.imshow(image)
            if gif:
                plt.savefig(f'temp_gif/plot_{i}.png')
            if console:
                plt.show()

    def compare_agents(self, agent_A, agent_B, n = 100):
        wins_A = 0
        wins_B = 0
        time = 0
        for i in range(n):
            result = self.run(agent_A, agent_B)
            if result[0] == 'A':
                wins_A += 1
            else:
                wins_B += 1
            time += result[1]
        return (wins_A / n, wins_B / n, time / n)
