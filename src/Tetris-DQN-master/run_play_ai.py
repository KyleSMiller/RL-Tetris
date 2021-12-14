import gym
import pygame
import gym_tetris
import matplotlib.pyplot as plt

from gym_tetris.ai.QNetwork import QNetwork


def main():
    env = gym.make("tetris-v1", action_mode=1)

    network = QNetwork(discount=1, epsilon=0, epsilon_min=0, epsilon_decay=0)  # Dellacherie
    # network = QNetwork(state_size=7, discount=1, epsilon=0, epsilon_min=0, epsilon_decay=0)  # EL-Tetris
    network.load()

    obs = env.reset()
    running = True
    display = True
    total_games = 0
    scores = []

    while running:
        action, state = network.act(obs)
        obs, reward, done, info = env.step(action)

        # if display:
        #     env.render()
        #
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         running = False
        #     if event.type == pygame.KEYDOWN:
        #         if event.key == pygame.K_RETURN:
        #             display = not display

        if done:
            total_games += 1
            scores.append(env.game.score)
            print("Games run: " + str(total_games))
            print("Score: " + str(scores[-1]))

            if total_games % 5 == 0 and total_games != 0:
                plt.boxplot(scores)
                plt.ylabel("score")
                plt.savefig('../../images/_results/Dellacherie_Reaction_2/fig_' + str(total_games))
                plt.cla()

            obs = env.reset()

    env.close()


if __name__ == '__main__':
    main()
