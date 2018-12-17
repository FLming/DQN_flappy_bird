import random
from datetime import datetime

import numpy as np
import tensorflow as tf

import replay_buffer

class DeepQNetworks:
    def __init__(self, n_actions, learning_rate=1e-6, gamma=0.99, memory_size=50000, 
        batch_size=32,
        initial_epsion=0,
        final_epsion=0,
        n_explore=200000,
        n_observes=100,
        frame_per_action=1,
        replace_target_iter=100):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.initial_epsion = initial_epsion
        self.final_epsion = final_epsion
        self.n_explore = n_explore
        self.n_observes = n_observes
        self.frame_per_action = frame_per_action
        self.replace_target_iter = replace_target_iter
        
        self.epsion = initial_epsion
        self.time_step = 0
        self.replay_memory = replay_buffer.ReplayBuffer(memory_size)

        self.createNetwork()
        q_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Q_network')
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_network')
        self.replace_target_op = [tf.assign(t, q) for t, q in zip(t_params, q_params)]

        self.saver = tf.train.Saver()
        self.sess = tf.Session()      
        self.sess.run(tf.global_variables_initializer())
        self.sess.graph.finalize()

        ckpt = tf.train.get_checkpoint_state("saved_networks")
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Successfully loaded:", ckpt.model_checkpoint_path)
        else:
            print("Could not find old network weights")

        self.writer = tf.summary.FileWriter("logs/" + "{0:%Y-%m-%d_%H:%M:%S/}".format(datetime.now()), self.sess.graph)

    def createNetwork(self):
        self.state_input = tf.placeholder(tf.float32, [None,80,80,4], name='state_input')
        self.next_state_input = tf.placeholder(tf.float32, [None,80,80,4], name='next_state_input')
        self.y_input = tf.placeholder(tf.float32, [None], name='y_input')
        self.action_input = tf.placeholder(tf.float32, [None, self.n_actions], name='action_input')

        with tf.variable_scope("Q_network"):
            with tf.variable_scope("conv0"):
                W_con0 = tf.get_variable('W_con0', [8,8,4,32], initializer=tf.truncated_normal_initializer(stddev=0.01))
                b_con0 = tf.get_variable('b_con0', [32], initializer=tf.constant_initializer(0.01))
                conv0 = tf.nn.relu(tf.nn.conv2d(self.state_input, W_con0, strides=[1,4,4,1], padding='SAME') + b_con0)
                pool0 = tf.nn.max_pool(conv0, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            with tf.variable_scope("conv1"):
                W_con1 = tf.get_variable('W_con1', [4,4,32,64], initializer=tf.truncated_normal_initializer(stddev=0.01))
                b_con1 = tf.get_variable('b_con1', [64], initializer=tf.constant_initializer(0.01))
                conv1 = tf.nn.relu(tf.nn.conv2d(pool0, W_con1, strides=[1,2,2,1], padding='SAME') + b_con1)
            with tf.variable_scope('conv2'):
                W_con2 = tf.get_variable('W_con2', [3,3,64,64], initializer=tf.truncated_normal_initializer(stddev=0.01))
                b_con2 = tf.get_variable('b_con2', [64], initializer=tf.constant_initializer(0.01))
                conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W_con2, strides=[1,1,1,1], padding='SAME') + b_con2)
            with tf.variable_scope('fc0'):
                conv2_flat = tf.reshape(conv2, [-1,1600])
                W_fc0 = tf.get_variable('W_fc0', [1600,512], initializer=tf.truncated_normal_initializer(stddev=0.01))
                b_fc0 = tf.get_variable('b_fc0', [512], initializer=tf.constant_initializer(0.01))
                fc0 = tf.nn.relu(tf.matmul(conv2_flat, W_fc0) + b_fc0)
            with tf.variable_scope('fc1'):
                W_fc1 = tf.get_variable('W_fc1', [512,self.n_actions], initializer=tf.truncated_normal_initializer(stddev=0.01))
                b_fc1 = tf.get_variable('b_fc1', [self.n_actions], initializer=tf.constant_initializer(0.01))
                self.Q_value = tf.matmul(fc0, W_fc1) + b_fc1
            self.summary_Q_value = tf.summary.scalar('mean_Q_value', tf.reduce_mean(self.Q_value))

        with tf.variable_scope('loss'):
            Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices = 1)
            self.loss = tf.reduce_mean(tf.square(self.y_input - Q_action))
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope("target_network"):
            with tf.variable_scope("conv0"):
                W_con0 = tf.get_variable('W_con0', [8,8,4,32], initializer=tf.truncated_normal_initializer(stddev=0.01))
                b_con0 = tf.get_variable('b_con0', [32], initializer=tf.constant_initializer(0.01))
                conv0 = tf.nn.relu(tf.nn.conv2d(self.next_state_input, W_con0, strides=[1,4,4,1], padding='SAME') + b_con0)
                pool0 = tf.nn.max_pool(conv0, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            with tf.variable_scope("conv1"):
                W_con1 = tf.get_variable('W_con1', [4,4,32,64], initializer=tf.truncated_normal_initializer(stddev=0.01))
                b_con1 = tf.get_variable('b_con1', [64], initializer=tf.constant_initializer(0.01))
                conv1 = tf.nn.relu(tf.nn.conv2d(pool0, W_con1, strides=[1,2,2,1], padding='SAME') + b_con1)
            with tf.variable_scope('conv2'):
                W_con2 = tf.get_variable('W_con2', [3,3,64,64], initializer=tf.truncated_normal_initializer(stddev=0.01))
                b_con2 = tf.get_variable('b_con2', [64], initializer=tf.constant_initializer(0.01))
                conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W_con2, strides=[1,1,1,1], padding='SAME') + b_con2)
            with tf.variable_scope('fc0'):
                conv2_flat = tf.reshape(conv2, [-1,1600])
                W_fc0 = tf.get_variable('W_fc0', [1600,512], initializer=tf.truncated_normal_initializer(stddev=0.01))
                b_fc0 = tf.get_variable('b_fc0', [512], initializer=tf.constant_initializer(0.01))
                fc0 = tf.nn.relu(tf.matmul(conv2_flat, W_fc0) + b_fc0)
            with tf.variable_scope('fc1'):
                W_fc1 = tf.get_variable('W_fc1', [512,self.n_actions], initializer=tf.truncated_normal_initializer(stddev=0.01))
                b_fc1 = tf.get_variable('b_fc1', [self.n_actions], initializer=tf.constant_initializer(0.01))
                self.Q_target = tf.matmul(fc0, W_fc1) + b_fc1

    def setInitState(self, observation):
        self.current_state = np.stack((observation, observation, observation, observation), axis = 2)

    def setPerception(self, next_observation, action, reward, terminal):
        new_state = np.append(self.current_state[:,:,1:], next_observation, axis = 2)
        self.replay_memory.add(self.current_state, action, reward, new_state, terminal)

        if self.time_step > self.n_observes:
            # Train the network
            self.trainQNetwork()

        self.current_state = new_state
        self.time_step += 1

    def trainQNetwork(self):
        # Step 1: obtain random minibatch from replay memory
        state_batch, action_batch, reward_batch, nextState_batch, terminal_batch = self.replay_memory.sample(self.batch_size)
        # Step 2: calculate y 
        Q_value_batch = self.sess.run(self.Q_target, feed_dict={self.next_state_input: nextState_batch})
        y_batch = np.where(terminal_batch, reward_batch, reward_batch + self.gamma * np.max(Q_value_batch, axis=1))
                
        summary, _ = self.sess.run([self.summary_Q_value, self.train_op], feed_dict={
            self.state_input: state_batch,
            self.y_input: y_batch,
            self.action_input: action_batch})

        self.writer.add_summary(summary, self.time_step)

        if self.time_step % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)

        if self.time_step % 10000 == 0:
            self.saver.save(self.sess, 'saved_networks/' + 'Qnetwork', global_step = self.time_step)

    def getAction(self):		
        action = np.zeros(self.n_actions)
        if self.time_step % self.frame_per_action == 0:
            Q_value = self.sess.run(self.Q_value, feed_dict={self.state_input: [self.current_state]})[0]
            action_index = random.randrange(self.n_actions) if random.random() <= self.epsion else np.argmax(Q_value)
            action[action_index] = 1
        else:
            action[0] = 1 # do nothing

        if self.epsion > self.final_epsion and self.time_step > self.n_observes:
            self.epsion -= (self.initial_epsion - self.final_epsion) / self.n_explore

        return action

    def logs(self, episode, score):
        summary = tf.Summary()
        summary.value.add(tag='score', simple_value=score)
        self.writer.add_summary(summary, episode)