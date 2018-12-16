# Deep Q-networks for flappy bird

## Requirements
- tensorflow
- pygame
- opencv
``` shell
pip install tensorflow pygame opencv-python
```
## Demo
``` shell
python flappybird.py
```
if you wanna see network architecture, make output_graph to True then
``` shell
tensorboard --logdir logs
```
## Roadmap
- [x] Implement 2013 paper DQN(with replay buffer)
- [ ] Implement 2015 paper DQN(add the target network)
- [ ] Implement others DQN such as Dueling DQN...
## Reference
- repo: [floodsung/DRL-FlappyBird](https://github.com/floodsung/DRL-FlappyBird)
- repo: [openai/baselines/deepq](https://github.com/openai/baselines/tree/master/baselines/deepq)
- repo: [MorvanZhou/Reinforcement-learning-with-tensorflow](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow)