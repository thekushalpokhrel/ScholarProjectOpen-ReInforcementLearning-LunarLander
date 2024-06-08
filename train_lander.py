import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
import random

# Initialize the environment
env = gym.make('LunarLander-v2' rendermode="human")

# Define the neural network model
def create_q_model():
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(env.observation_space.shape[0],)),
        layers.Dense(128, activation='relu'),
        layers.Dense(env.action_space.n, activation='linear')
    ])
    return model

# Hyperparameters
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
batch_size = 64
learning_rate = 0.001
target_update_freq = 10

# Initialize models
model = create_q_model()
target_model = create_q_model()
target_model.set_weights(model.get_weights())

optimizer = tf.keras.optimizers.Adam(learning_rate)
loss_function = tf.keras.losses.Huber()

# Experience replay buffer
replay_buffer = deque(maxlen=100000)

# Training function
def train_step(batch):
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

    state_batch = np.array(state_batch)
    next_state_batch = np.array(next_state_batch)
    reward_batch = np.array(reward_batch)
    done_batch = np.array(done_batch)

    future_rewards = target_model.predict(next_state_batch)
    updated_q_values = reward_batch + gamma * np.max(future_rewards, axis=1) * (1 - done_batch)

    masks = tf.one_hot(action_batch, env.action_space.n)

    with tf.GradientTape() as tape:
        q_values = model(state_batch)
        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
        loss = loss_function(updated_q_values, q_action)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

# Main training loop
num_episodes = 100
evaluation_frequency = 10

for episode in range(num_episodes):
    state = env.reset()[0]
    episode_reward = 0

    for step in range(100):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(np.array([state]))[0])

        next_state, reward, done, _, _ = env.step(action)
        episode_reward += reward

        replay_buffer.append((state, action, reward, next_state, done))

        state = next_state

        if done:
            break

    if len(replay_buffer) > batch_size:
        batch = random.sample(replay_buffer, batch_size)
        train_step(batch)

    if episode % target_update_freq == 0:
        target_model.set_weights(model.get_weights())

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    print(f'Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}, Epsilon: {epsilon}')

env.close()
