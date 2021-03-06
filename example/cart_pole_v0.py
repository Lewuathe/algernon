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

env = gym.make('CartPole-v0')

model = Sequential()
model.add(Dense(100, input_dim=4, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(2))
model.compile(sgd(lr=.2), "mse")
print("Observation Space: {}".format(env.observation_space))
print("Action Space: {}".format(env.action_space))


agent = Agent(model=model, observation_shape=env.observation_space.shape, action_list=np.array([0,1]), episode=50, epsilon=0.7)

agent.fit(env, callback, callback)
agent.fit(env, render, callback)

