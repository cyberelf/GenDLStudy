import numpy as np
import gym

env = gym.make('CartPole-v1')
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]

# Initialize policy parameters
theta = np.random.rand(n_states, n_actions)

def softmax(x):
    z = x - np.max(x)
    return np.exp(z) / np.sum(np.exp(z), axis=0)

def policy(state, theta):
    z = state.dot(theta)
    return softmax(z)

def generate_episode(env, policy, theta):
    states, actions, rewards = [], [], []
    state = env.reset()
    done = False
    while not done:
        probs = policy(state, theta)
        action = np.random.choice(len(probs), p=probs)
        next_state, reward, done, _ = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state
    return states, actions, rewards

def compute_returns(rewards, gamma=0.99):
    returns = np.zeros_like(rewards)
    G = 0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns

# Training loop
alpha = 0.01  # Learning rate
for episode in range(1000):
    states, actions, rewards = generate_episode(env, policy, theta)
    returns = compute_returns(rewards)
    for t in range(len(states)):
        state, action, G = states[t], actions[t], returns[t]
        theta[:, action] += alpha * G * (1 - policy(state, theta)[action]) * state

# Test the trained policy
state = env.reset()
done = False
while not done:
    env.render()
    action = np.argmax(policy(state, theta))
    state, reward, done, _ = env.step(action)
env.close()
