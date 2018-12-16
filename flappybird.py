import sys
sys.path.append("game/")

import numpy as np
import cv2
import matplotlib.pyplot as plt

import game.wrapped_flappy_bird as game
from DQN_NIPS import DeepQNetworks

def preprocess(observation):	
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(observation, (80,80,1))

def playFlappyBird():
    action = 2
    brain = DeepQNetworks(action)
    flappyBird = game.GameState()
    action0 = np.array([1,0])
    observation0, reward0, terminal = flappyBird.frame_step(action0)
    observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation0 = cv2.threshold(observation0, 1, 255, cv2.THRESH_BINARY)
    brain.setInitState(observation0)

    episode = 0
    total_reward = 0
    while True:
        action = brain.getAction()
        score = flappyBird.score
        next_observation, reward, terminal = flappyBird.frame_step(action)
        next_observation = preprocess(next_observation)
        total_reward += reward
        brain.setPerception(next_observation, action, reward, terminal)
        if terminal:
            with open('scores.txt','a+') as f:
                f.write("{},{},{:.1f}\n".format(episode, score, total_reward))
            print("episode: {}, score: {}, total reward: {:.1f}".format(episode, score, total_reward))
            total_reward = 0
            episode += 1
    
if __name__ == "__main__":
    playFlappyBird()
