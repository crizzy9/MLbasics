import os
import pickle
import itertools
import gym
import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.callbacks import TensorBoard
import tflearn
from tflearn.layers import input_data, fully_connected, dropout
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter

# neural network with x = [4] and y = [2]
# build neural network from scratch in numpy
game = 'CartPole-v1'
env = gym.make(game)
initial_score_requirement = 50
one_hot_action = {
    0: [1, 0],
    1: [0, 1]
}
data_loc = './train_data/{}'.format(game)
model_loc = './model/{}.pickle'.format(game)
LR = 0.001
total_steps = 500


def generate_train_data(no_of_episodes):
    scores = []
    accepted_scores = []
    train_data = []
    for ep in range(no_of_episodes):
        observation = env.reset()
        score = 0
        # y, x
        game_data = []
        for t in range(total_steps):
            # env.render()
            action = env.action_space.sample()
            # print('state: {}, action: {}'.format(observation, action))
            game_data.append([one_hot_action[action], observation])
            observation, reward, done, info = env.step(action)
            score += reward
            if done:
                print('{} Episode finished after {} timesteps with score {}'.format(ep, t+1, score))
                break

        if score >= initial_score_requirement:
            accepted_scores.append(score)
            train_data.append(game_data)
            print("Success. Saving data for episode {}!".format(ep))

        scores.append(score)
    print('**********train_data************')
    print(np.array(train_data).view())
    print(np.array(train_data).shape)
    flat_data = list(itertools.chain(*train_data))
    print('**********flat_data************')
    print(np.array(flat_data).view())
    print(np.array(flat_data).shape)
    np.save(data_loc, np.array(flat_data, dtype=object))
    print('Average accepted score: {}'.format(mean(accepted_scores)))
    print('Median score for accepted scores: {}'.format(median(accepted_scores)))
    print('Score counter:\n{}'.format(Counter(accepted_scores)))
    return flat_data


def nn_model(input_size):
    # same implementation with keras
    # model = Sequential()
    # model.add(Dense(128, input_shape=size, activation='relu'))

    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8) # meaning 0.8 will be kept, opposite in keras

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(network, tensorboard_dir='log')
    return model


def train(train_data, model=False):
    print('Train data')
    print(train_data)
    print(np.array(train_data).shape)
    X = np.array([d[1] for d in train_data]).reshape(-1, len(train_data[0][1]), 1)
    y = np.array([d[0] for d in train_data])
    print('X.shape: {}, X example: {}'.format(X.shape, X[0]))
    print('y.shape: {}, y example: {}'.format(y.shape, y[0]))
    if not model:
        model = nn_model(len(X[0]))

    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openai_learning')
    print('Saving Model')
    # model.save('./models/{}.tfl'.format(game))
    # model.load()
    # store_pickle(model, model_loc)
    return model

def store_pickle(obj, file_path):
    print("Storing pickle:", file_path)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(file_path):
    if os.path.isfile(file_path):
        print("Loading pickle:", file_path)
        if os.path.getsize(file_path) > 0:
            with open(file_path, "rb") as handle:
                dic = pickle.load(handle)
            return dic
    else:
        return FileNotFoundError


def run(model):
    scores = []
    choices = []
    for ep in range(10):
        observation = env.reset()
        score = 0
        game_data = []
        for t in range(total_steps):
            env.render()

            print('Model prediction {}'.format(model.predict(observation.reshape(-1, len(observation), 1))))

            action = np.argmax(model.predict(observation.reshape(-1, len(observation), 1))[0])
            choices.append(action)

            observation, reward, done, info = env.step(action)
            score += reward
            game_data.append([one_hot_action[action], observation])

            if done:
                print('Episode {} finished after {} timesteps with score {}'.format(ep, t+1, score))
                break

        scores.append(score)
    print('Average accepted score: {}'.format(mean(scores)))
    print('Median score for accepted scores: {}'.format(median(scores)))
    print('Choice 1: {}, Choice 2: {}'.format(choices.count(1)/len(choices), choices.count(0)/len(choices)))


if __name__ == '__main__':
    # training_data = generate_train_data(3000)
    training_data = np.load(data_loc+'.npy')
    model = train(training_data)
    # model = load_pickle(model_loc)
    run(model)


