import gym
import numpy as np

# use neural network with x = [4] and y = [2]
# build neural network from scratch in numpy
env = gym.make('CartPole-v0')


for i_episode in range(1000):
    observation = env.reset()
    train_data = []
    for t in range(200):
        env.render()
        action = env.action_space.sample()
        print('state: {}, action: {}'.format(observation, action))
        observation, reward, done, info = env.step(action)
        if done:
            print('Episode finished after {} timesteps'.format(t+1))
            break


