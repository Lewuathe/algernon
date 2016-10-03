import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd

import gym

from algernon.agent import Agent


def callback(epoch, env, total_reward):
    pass
    # print("epoch: {}, total_reward: {}".format(epoch, total_reward))

def render(epoch, env, total_reward):
    env.render()

env = gym.make('Pendulum-v0')

print("Observation Space: {}".format(env.observation_space))
print("Action Space: {}".format(env.action_space))
model = Sequential()
model.add(Dense(10, input_dim=3, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(sgd(lr=.2), "mse")


agent = Agent(model=model, observation_shape=env.observation_space.shape, action_list=np.array([[-2], [2]]), episode=50, epsilon=0.7, train_interval_step=100)

agent.fit(env, render, callback)

