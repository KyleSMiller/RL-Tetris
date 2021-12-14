from typing import Tuple

from tqdm import tqdm
import gym
import numpy as np


def evalAgent(agent, env, num_eval_episodes, render=False):
    """
    Evaluates the agents performance on the game of SimplifiedTetris and returns the mean score.

    :param agent: the agent to evaluate on the env.
    :param env: the env to evaluate the agent on.
    :param num_eval_episodes: the number of games to evaluate the trained agent.
    :param render: a boolean that if True renders the agent playing SimplifiedTetris after training.
    :return: the mean and std score obtained from letting the agent play num_eval_episodes games.
    """

    returns = np.zeros(num_eval_episodes)

    for episode in tqdm(range(num_eval_episodes), desc="No. of episodes completed"):

        state = env.reset()
        done = False

        while not done:

            if render:
                env.render()

            action = agent.predict(state)

            state, reward, done, _ = env.step(action)
            returns[episode] += reward

    env.close()

    mean_score = np.mean(returns)
    std_score = np.std(returns)

    return mean_score, std_score
