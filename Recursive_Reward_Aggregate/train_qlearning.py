import gym
import numpy as np
import pickle as pkl

cliffEnv = gym.make("CliffWalking-v0")

q_table = np.zeros(shape=(48, 4))

# Parameters
EPSILON = 0.2  # 0.1
ALPHA = 0.1
GAMMA = 0.9
MAX_EPISODES = 1000

NUM_EPISODES = 500


def policy(state, explore=0.0):
    action = int(np.argmax(q_table[state]))
    if np.random.random() <= explore:
        action = int(np.random.randint(low=0, high=4, size=1))

    return action


GOAL_STATE = 47

def reward_shaping(state, reward):
    x, y = state // 12, state % 12
    goal_x, goal_y = GOAL_STATE // 12, GOAL_STATE % 12
    distance_to_goal = abs(goal_x - x) + abs(goal_y - y)

    shaped_reward = reward - 0.1 * distance_to_goal
    return shaped_reward


for episode in range(NUM_EPISODES):

    done = False

    total_reward = 0
    episode_length = 0

    state = cliffEnv.reset()

    while not done:
        action = policy(state, EPSILON)
        next_state, reward, done, _ = cliffEnv.step(action)
        next_action = policy(next_state)

        shaped_reward = reward_shaping(next_state, reward)
        if done:
            shaped_reward = 100
            # reward = 100

        q_table[state][action] += ALPHA * (shaped_reward + GAMMA * q_table[next_state][next_action] - q_table[state][action])  # sum
        # q_table[state][action] += ALPHA * max(shaped_reward, GAMMA * q_table[next_state][next_action] - q_table[state][action])  # max

        state = next_state

        total_reward += reward
        episode_length += 1

        if done:
            print("FINISHED!!")

        if reward == -100:
            done = True

        if episode_length == MAX_EPISODES:
            done = True
            print("Episode terminated due to step limit reached.")


    print("Episode:", episode, "Episode Length:",
          episode_length, "Total Reward:", total_reward)


cliffEnv.close()

pkl.dump(q_table, open("q_learning_q_table.pkl", "wb"))

print("Taining Complete, Q Table Saved :)")