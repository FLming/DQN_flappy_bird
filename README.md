# Deep Q-networks for flappy bird
## Roadmap
- [x] Implement 2013 NIPS paper's DQN(with replay buffer)
- [x] Implement 2015 Nature paper's DQN(add the target network)
- [ ] Implement others DQN such as Dueling DQN...
## Requirements
- tensorflow
- pygame
- opencv
``` shell
pip install tensorflow pygame opencv-python
```
## Demo
``` shell
git clone https://github.com/FLming/DQN_flappy_bird.git
cd DQN_flappybird
python flappybird.py
```
if you wanna see network architecture, and the change of variables.
``` shell
tensorboard --logdir logs
```
## Reference
- repo: [floodsung/DRL-FlappyBird](https://github.com/floodsung/DRL-FlappyBird)
- repo: [openai/baselines/deepq](https://github.com/openai/baselines/tree/master/baselines/deepq)
- repo: [MorvanZhou/Reinforcement-learning-with-tensorflow](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow)