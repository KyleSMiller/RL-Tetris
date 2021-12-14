import gym
import gym_tetris
import matplotlib.pyplot as plt
import numpy as np


from statistics import mean, median
from gym_tetris.ai.QNetwork import QNetwork


def main():
    env = gym.make("tetris-v1", action_mode=1)
    # network = QNetwork(state_size=9)  # Dellacherie
    network = QNetwork(state_size=7)  # EL-Tetris
    network.load()

    plt.figure(figsize=[10, 6])

    running = True
    total_games = 0
    total_steps = 0

    gamesPlayed = []
    scoreAverages = []

    while running:
        steps, rewards, scores = network.train(env, episodes=25)
        total_games += len(scores)
        total_steps += steps
        network.save()

        print("==================")
        print("* Total Games: ", total_games)
        print("* Total Steps: ", total_steps)
        print("* Epsilon: ", network.epsilon)
        print("*")
        print("* Average: ", sum(rewards) / len(rewards), "/", sum(scores) / len(scores))
        print("* Median: ", median(rewards), "/", median(scores))
        print("* Mean: ", mean(rewards), "/", mean(scores))
        print("* Min: ", min(rewards), "/", min(scores))
        print("* Max: ", max(rewards), "/", max(scores))
        print("==================")

        if total_games % 100 == 0:
            gamesPlayed.append(total_games)
            scoreAverages.append(sum(scores) / len(scores))

            print(gamesPlayed)
            print(scoreAverages)

            plt.plot(gamesPlayed, scoreAverages)
            plt.xlabel("games played")
            plt.ylabel("average score")
            plt.savefig('../../images/EL-Tetris_Reaction_2/fig_' + str(gamesPlayed[-1]))
            plt.cla()

    env.close()


if __name__ == '__main__':
    main()
