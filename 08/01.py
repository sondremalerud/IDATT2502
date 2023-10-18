import gym
import numpy as np
import pickle
import matplotlib.pyplot as plt

# DOCUMENTATION https://gymnasium.farama.org/environments/classic_control/cart_pole/

# Q-learning parameters
learning_rate = 0.1
discount_factor = 0.99
exploration_prob = 1 # Probability of exploring a random action
episodes = 100000 # Will likely stop earlier if mean rewards is good enough

rewards_per_episode = []


env = gym.make("CartPole-v1", render_mode=None)

# Devide position, velocity, angle and angle velocity into bigger segments
# These can get tweaked, remember min and max of all variables are found in the documentation
pos_space = np.linspace(-2.4, 2.4, 10) # 10 bins
vel_space = np.linspace(-4, 4, 10)
angle_space = np.linspace(-0.2095, 0.2095, 10)
angle_vel_space = np.linspace(-4, 4, 10)

# Initializing 11x11x11x11x2 array
q_table = np.zeros((len(pos_space)+1, len(vel_space)+1, len(angle_space)+1, len(angle_vel_space)+1, env.action_space.n)) 

for episode in range(episodes):
    state, info = env.reset() # starter miljøet

    # Assigns each value to a bin (created with np.linspace above)
    state_pos = np.digitize(state[0], pos_space)
    state_vel = np.digitize(state[1], vel_space)
    state_angle = np.digitize(state[2], angle_space)
    state_angle_vel = np.digitize(state[3], angle_vel_space)

    terminated = False
    rewards = 0
    #print(q_table[state_pos, state_vel, state_angle, state_angle_vel, :])

    while(not terminated):

        # Exploration
        if np.random.default_rng().random() < exploration_prob:
            action = env.action_space.sample() # Random -> explore
        else:
            action = np.argmax(q_table[state_pos, state_vel, state_angle, state_angle_vel, :])

        # Take the step
        next_state, reward, terminated, truncated, info = env.step(action) 

        next_state_pos = np.digitize(next_state[0], pos_space)
        next_state_vel = np.digitize(next_state[1], vel_space)
        next_state_angle = np.digitize(next_state[2], angle_space)
        next_state_angle_vel = np.digitize(next_state[3], angle_vel_space)

        #env.render()

        # update Q-value. State and action will provide a new q-value
        q_value = q_table[state_pos, state_vel, state_angle, state_angle_vel, action]

        max_q_next = np.max(q_table[next_state_pos, next_state_vel, next_state_angle, next_state_angle_vel, :])

        # Q(s,a):
        new_q_value = (1 - learning_rate) * q_value + learning_rate * (reward + discount_factor * max_q_next)
        q_table[state_pos, state_vel, state_angle, state_angle_vel, action] = new_q_value

        state = next_state
        state_pos = next_state_pos
        state_vel = next_state_vel
        state_angle = next_state_angle
        state_angle_vel = next_state_angle_vel
        
        rewards+=reward


    rewards_per_episode.append(rewards)
    mean_rewards = np.mean(rewards_per_episode[len(rewards_per_episode)-100:]) # Mean of last 100 episodes

    if (episode % 100 == 0):
        print(f'Episode: {episode} {rewards}  Mean rewards {mean_rewards:0.1f}')
        
    # Good enogh :)
    if mean_rewards>1000:
        break

    exploration_prob = max(exploration_prob - 0.00001, 0)


env.close() # closes env and render()-window

mean_rewards = []

for t in range(len(rewards_per_episode)):
    mean_rewards.append(np.mean(rewards_per_episode[max(0, t-100):(t+1)]))
plt.plot(mean_rewards)
plt.xlabel('Episode')
plt.ylabel('Mean Reward')
plt.savefig('./08/cartpole.png')
