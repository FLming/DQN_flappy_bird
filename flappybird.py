import sys
sys.path.append("game/")

import numpy as np
import cv2

import game.wrapped_flappy_bird as game
from dueling_DQN import DeepQNetworks

def preprocess(observation):	
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation, 1, 1, cv2.THRESH_BINARY)
    return np.reshape(observation, (80,80,1))

def playFlappyBird():
    action = 2
    brain = DeepQNetworks(action)
    flappyBird = game.GameState()
    action0 = np.array([1,0])
    observation0, reward0, terminal = flappyBird.frame_step(action0)
    observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation0 = cv2.threshold(observation0, 1, 1, cv2.THRESH_BINARY)
    brain.setInitState(observation0)

    while True:
        action = brain.getAction()
        score = flappyBird.score
        next_observation, reward, terminal = flappyBird.frame_step(action)
        next_observation = preprocess(next_observation)
        brain.setPerception(next_observation, action, reward, terminal)
        if terminal:
            brain.log_score(score)
    
if __name__ == "__main__":
    playFlappyBird()
