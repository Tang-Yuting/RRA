import sys
sys.path.insert(0, "./stable_baselines3")

import gymnasium as gym
from stable_baselines3 import PPO, MAX_PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

import numpy as np

env = gym.make("CartPole-v1", render_mode="rgb_array")
env = Monitor(env)

model = MAX_PPO("CustomActorCriticPolicy", env, verbose=1, learning_rate=0.000001,
            n_steps=2048, batch_size=64, gamma=0.99, recursive_type="extended_max_new")

model.learn(total_timesteps=100000)

model.save("ppo_CartPole")

num_episodes = 10
max_steps_per_episode = 500
episode_rewards = []
failures = 0

for episode in range(num_episodes):
    obs, _ = env.reset()
    extend_state = 0
    total_reward = 0
    steps = 0

    while steps < max_steps_per_episode:

        action, _ = model.predict(obs, extend_state=extend_state, deterministic=True)

        new_obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        steps += 1

        extend_state = max(reward, model.gamma * extend_state)

        obs = new_obs
        if done:
            failures += 1
            break

    episode_rewards.append(total_reward)
    print(f"Episode {episode + 1}: Steps = {steps}, Reward = {total_reward}")


# 计算评估数据
avg_reward = sum(episode_rewards) / num_episodes
print("\n===== Evaluation Summary =====")
print(f"Total Episodes: {num_episodes}")
print(f"Failures (Episodes terminated early): {failures}")
print(f"Average Reward: {avg_reward:.2f}")
print(f"Best Reward: {max(episode_rewards):.2f}")
print(f"Worst Reward: {min(episode_rewards):.2f}")

# 关闭环境
env.close()
