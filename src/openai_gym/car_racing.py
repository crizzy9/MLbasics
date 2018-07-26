import os
import random
from statistics import mean, median
from collections import Counter
import itertools
import gym
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard

game = 'CarRacing-v0'
env = gym.make(game)

steps = 1000
data_loc = './train_data/{}'.format(game)
model_loc = './model/{}.pickle'.format(game)


def generate_train_data(no_of_episodes):
    scores = []
    train_data = []
    for episode in range(no_of_episodes):
        observation = env.reset()
        score = 0
        game_data = []
        for t in range(steps):
            # env.render("human")
            action = env.action_space.sample()
            print('state: {}, action: {}'.format(observation, action))
            # y, X
            game_data.append([action, observation])
            observation, reward, done, info = env.step()
            score += reward
            if done:
                print('Episode {} finished after {} timesteps with score {}'.format(episode, t+1, score))
                break

        scores.append(score)
        train_data.append(game_data)
        print("Success. Saving data for episode {}!".format(episode))

    print('**********train_data************')
    print(np.array(train_data).view())
    print(np.array(train_data).shape)
    flat_data = list(itertools.chain(*train_data))
    print('**********flat_data************')
    print(np.array(flat_data).view())
    print(np.array(flat_data).shape)
    np.save(data_loc, np.array(train_data, dtype=object))
    print('Average accepted score: {}'.format(mean(scores)))
    print('Median score for accepted scores: {}'.format(median(scores)))
    print('Score counter:\n{}'.format(Counter(scores)))
    return train_data


# model ref: https://pythonprogramming.net/training-neural-network-starcraft-ii-ai-python-sc2-tutorial/
def nn_model(input_size):
    model = Sequential()
    # convert to gray scale later
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(3, activation='softmax'))

    learning_rate = 0.0001
    opt = keras.optimizers.adam(lr=learning_rate, decay=1e-6)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    tensorboard = TensorBoard(log_dir='log/stage1')

    return model, tensorboard

def train(epochs):
    for epoch in range(epochs):
        current = 0
        increment = 200
        not_max = True
        # all_files = os.listdir(data_loc)
        train_data = np.load(data_loc + '.npy')
        maximum = len(train_data)
        random.shuffle(train_data)

        while not_max:
            print("WORKING ON {}:{}".format(current, current + increment))
            steer = []
            gas = []
            brake = []

            # for t in train_data:
                
