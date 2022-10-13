import random
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeNormal
from keras import backend as K
import tensorflow as tf

from utils.logger import Logger
tf.config.set_visible_devices([], 'GPU')

class LearningRateLoggingCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs = None):
        if (random.randint(0,5) == 0):
            lr = self.model.optimizer._decayed_lr('float32').numpy()
            logger = Logger()
            logger.write_lr_log(lr)
        
class DDQNAgent:
    def __init__(self, config):
        self.state_size = config.state_size
        self.action_size = config.action_size
        self.memory = deque(maxlen=config.memory_size)
        self.gamma = config.discount_factor  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay
        self.learning_rate = config.learning_rate
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self, nb_layers = 2, nb_node = 32):
        # Neural Net for Deep-Q learning Model
        initializer = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
        model = Sequential()
        model.add(Dense(nb_node, input_dim=self.state_size, activation='relu', kernel_initializer = initializer))
        for _ in range(nb_layers - 1):
            model.add(Dense(nb_node//2,  activation='relu', kernel_initializer = initializer))

        model.add(Dense(self.action_size, activation='linear', kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.1, maxval=0.1),
            bias_initializer=tf.keras.initializers.Constant(0.4)))

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.learning_rate,
            decay_steps=3000,
            decay_rate=0.95,
            staircase=True
        )
        
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = [i[0][0] for i in minibatch]
        next_states = [i[3][0] for i in minibatch]
        q_values_state_list = self.model.predict(np.array(states), batch_size = batch_size)
        q_values_nextstate_list = self.model.predict(np.array(next_states), batch_size = batch_size)
        q_values_target_nextstate_list = self.target_model.predict(np.array(next_states), batch_size = batch_size)
        states, targets_f = [], []
        index = 0
        
        for state, action, reward, next_state, decision in minibatch:
            target = q_values_state_list[index]
            a = q_values_nextstate_list[index]
            t = q_values_target_nextstate_list[index]
            target[action] = reward + self.gamma * t[np.argmax(a)]
            states.append(state[0])
            targets_f.append(target)
            index = index + 1
            if decision == False:
                target[0] = reward + self.gamma * t[np.argmax(a)]
                target[1] = reward + self.gamma * t[np.argmax(a)]

        self.model.fit(np.array(states), np.array(targets_f), batch_size = batch_size, epochs=1, verbose=0, callbacks=[LearningRateLoggingCallback()])

    def load(self, name):
        self.model.load_weights(name)
        self.update_target_model()

    def save(self, name):
        self.model.save_weights(name)
