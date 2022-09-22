import numpy as np
from tensorflow.keras.optimizers import RMSprop
import keras.backend as K

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


class Actor:
    def __init__(self, state_dim, action_dim, actor_lr):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(actor_lr)

    def create_model(self):
        return tf.keras.Sequential([
            Input((self.state_dim)),
            Dense(16, activation='relu'),
            Dense(16, activation='relu'),
            Dense(self.action_dim, activation='softmax')
        ])

    def compute_loss(self, actions, logits, advantages):
        #print(actions, logits, advantages)
        ce_loss = tf.keras.losses.SparseCategoricalCrossentropy()
        actions = tf.cast(actions, tf.int32)
        policy_loss = ce_loss(
            actions, logits, sample_weight=tf.stop_gradient(advantages))
        probs = tf.nn.softmax(logits)
        entropy_loss = tf.keras.losses.categorical_crossentropy(probs, probs)
        return policy_loss - entropy_loss * 0.01

    def train(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            logits = self.model(states, training=True)
            loss = self.compute_loss(
                actions, logits, advantages)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

class Critic:
    def __init__(self, state_dim, critic_lr):
        self.state_dim = state_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(critic_lr)

    def create_model(self):
        return tf.keras.Sequential([
            Input((self.state_dim)),
            Dense(16, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class A2CAgent:
    def __init__(self, state_size, action_size):
        self.state_dim = state_size
        self.action_dim = action_size
        self.actor = Actor(self.state_dim, self.action_dim, 0.0005)
        self.critic = Critic(self.state_dim, 0.0005)
        self.gamma = 0.997
        self.batch_size = 32

    def td_target(self, reward, next_state):
        v_value = self.critic.model.predict(next_state)
        if (v_value > 2000000 or reward > 2000000):
            print(next_state, v_value, reward)
        return np.reshape(reward + self.gamma * v_value[0], [1, 1])

    def advantage(self, td_targets, baselines):
        return td_targets - baselines

    def act(self, state):
        probs = self.actor.model.predict(state)
        return np.random.choice(self.action_dim, p=probs[0])
        
    def upd(self, state, action, reward, next_state):
        reward  = reward
        td_target = self.td_target(reward, next_state)
        advantage = self.advantage(td_target, self.critic.model.predict(state))
        actor_loss = self.actor.train(state, action, advantage)
        critic_loss = self.critic.train(state, td_target)
        
    def update_episode(self, states, actions, rewards):
        discount_rewards, cumul = np.zeros_like(rewards, dtype=float), 0.0
        for t in reversed(range(0, len(rewards))):
            cumul = rewards[t] + self.gamma * cumul
            discount_rewards[t] = cumul

        #states = tf.stack(states)
        state_values = self.critic.model.predict(np.array(states))
        state_values = np.reshape(state_values, len(state_values))
        discount_rewards =  np.reshape(discount_rewards, len(rewards))
        advantages = discount_rewards - state_values
        #Training on batches
        indices = np.arange(len(states))
        np.random.shuffle(indices)
        for batch in np.split(indices, np.arange(self.batch_size,len(indices),self.batch_size)):
            batch_states = list(np.array(states)[batch])
            batch_actions = list(np.array(actions)[batch].reshape([len(batch), 1]))
            batch_advantages = advantages[batch]
            batch_rewards = discount_rewards[batch]
            batch_rewards = np.reshape(batch_rewards, [len(batch_rewards), 1])
            batch_states = tf.stack(batch_states)
            actor_loss = self.actor.train(batch_states, batch_actions, batch_advantages)
            critic_loss = self.critic.train(batch_states, batch_rewards)
    
    def load(self, name):
        self.actor.model.load_weights(name + "_actor.h5")
        self.critic.model.load_weights(name + "_critic.h5")

    def save(self, name):
        self.actor.model.save_weights(name + "_actor.h5")
        self.critic.model.save_weights(name + "_critic.h5")
