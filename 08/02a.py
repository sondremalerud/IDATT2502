import numpy as np
import pygame
import sys

class GridWorld:
    def __init__(self, width, height, goal_pos):
        self.width = width
        self.height = height
        self.goal_pos = goal_pos
        self.agent_pos = (0, 0)
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        self.state = (0, 0)
        self.done = False

    def reset(self):
        self.agent_pos = (0, 0)
        self.state = (0, 0)
        self.done = False
        return self.state

    def step(self, action):
        if self.done:
            raise ValueError("Episode is done. Call reset() to restart.")

        new_pos = (self.agent_pos[0] + self.actions[action][0], self.agent_pos[1] + self.actions[action][1])

        # Check if move is within grid world and updates pos
        if 0 <= new_pos[0] < self.width and 0 <= new_pos[1] < self.height:
            self.agent_pos = new_pos
        else:
            reward = -1 # crashed into a wall
            return self.agent_pos, reward, self.done


        # Check if the agent reached the goal
        if self.agent_pos == self.goal_pos:
            self.done = True
            reward = 1
        else:
            reward = 0

        return self.agent_pos, reward, self.done

class QLearningAgent:
    def __init__(self, action_space_size):
        self.action_space_size = action_space_size
        self.q_table = np.zeros((width, height, action_space_size))
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.99  # Discount factor
        self.epsilon = 0.1  # Exploration-exploitation trade-off (Exploration probability)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            # Generates random sample from action_space_size
            return np.random.choice(self.action_space_size)
        else:
            return np.argmax(self.q_table[state[0], state[1]])

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state[0], next_state[1]])
        current_q_value = self.q_table[state[0], state[1], action]
        next_q_value = self.q_table[next_state[0], next_state[1], best_next_action]
        new_q_value = (1 - self.alpha) * current_q_value + self.alpha * (reward + self.gamma * next_q_value)
        self.q_table[state[0], state[1], action] = new_q_value

def draw_q_values(q_table):
    for i in range(width):
        for j in range(height):
            actions = q_table[i, j]
            best_action = np.argmax(actions)
            color = 255 - int(actions[best_action] * 255)
            pygame.draw.rect(screen, (color, color, color), (i * cell_size, j * cell_size, cell_size, cell_size))

            # Adjust the angle based on the best action
            angle = 90
            if best_action == 0:
                angle = -270  # Up
            elif best_action == 1:
                angle = -180  # Down
            elif best_action == 2:
                angle = 0  # Left

            

            # Draw arrow
            pygame.draw.line(screen, (0, 255, 0), (i * cell_size + cell_size // 2, j * cell_size + cell_size // 2),
                             (i * cell_size + cell_size // 2 - np.cos(np.radians(angle)) * actions[best_action] * 20,
                              j * cell_size + cell_size // 2 + np.sin(np.radians(angle)) * actions[best_action] * 20), 2)

def draw_agent(current_position):

    for i in range(width):
        for j in range(height):
            # Draw a dot for the current position
            if (i, j) == current_position:
                pygame.draw.circle(screen, (0, 0, 255), (i * cell_size + cell_size // 2, j * cell_size + cell_size // 2), 10)



# Main loop
width, height = 5, 5
goal_pos = (width - 1, height - 1)

env = GridWorld(width, height, goal_pos)
agent = QLearningAgent(action_space_size=4)

cell_size = 50
pygame.init()
screen = pygame.display.set_mode((width * cell_size, height * cell_size))

episode = 0

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    state = env.reset()
    total_reward = 0

    while not env.done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update_q_table(state, action, reward, next_state)

        state = next_state
        total_reward += reward
        draw_agent(env.agent_pos)
        pygame.display.flip()



    if episode % 100 == 0:
        print(f"Episode: {episode}, Total Reward: {total_reward}")

    screen.fill((255, 255, 255))
    draw_q_values(agent.q_table)
    pygame.display.flip()

    episode += 1

pygame.quit()
