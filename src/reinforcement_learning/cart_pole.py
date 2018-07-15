import os
import gym
import numpy as np


# use neural network with x = [4] and y = [2]
# build neural network from scratch in numpy
env = gym.make('CartPole-v1')


def generate_train_data(no_of_episodes, total_steps):
    for i_episode in range(no_of_episodes):
        observation = env.reset()
        # y, x
        train_data = []
        for t in range(total_steps):
            # env.render()
            action = env.action_space.sample()
            # print('state: {}, action: {}'.format(observation, action))
            train_data.append([action, observation])
            observation, reward, done, info = env.step(action)
            if done:
                print('Episode finished after {} timesteps'.format(t+1))
                np.save('./train_data/cart_pole_{}'.format(i_episode), np.array(train_data, dtype=object))
                break


def train(training_files):
    for f in training_files:
        y, X = np.fromfile(f)



if __name__ == '__main__':
    generate_train_data(2000, 500)
    # train(os.listdir('./train_data'))
