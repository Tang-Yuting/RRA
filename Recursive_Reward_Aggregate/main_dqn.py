import sys
sys.path.insert(0, "./stable_baselines3")

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.evaluation import evaluate_policy

# 包装环境为 DummyVecEnv (SB3 推荐)
def make_env():
    return gym.make("CartPole-v1", render_mode="rgb_array")

# 创建向量化环境
env = DummyVecEnv([make_env])

# 使用 VecVideoRecorder 保存视频
video_folder = "./videos"
env = VecVideoRecorder(env, video_folder=video_folder,
                       record_video_trigger=lambda x: x == 0, # 只录制第一个回合
                       video_length=500, # 每段视频最大长度
                       name_prefix="dqn_CartPole")

# 使用 DQN 训练
model = DQN(
    "MlpPolicy", env, verbose=1,
    learning_rate=0.0001,  # 降低学习率，提高训练稳定性
    exploration_fraction=0.5,  # 探索时间更长
    exploration_final_eps=0.05,  # 最终更少随机探索
    buffer_size=200000,  # 增加经验缓冲区
    target_update_interval=5000,  # 更快地更新目标网络
)

model.learn(total_timesteps=200000)

# 保存模型
model.save("dqn_CP")

# 测试模型并录制视频
num_episodes = 10  # 设定测试回合数
max_steps_per_episode = 500  # 每个回合最多 500 步
episode_rewards = []  # 记录每个回合的总奖励
failures = 0  # 记录失败次数

for episode in range(num_episodes):
    obs = env.reset()
    total_reward = 0
    steps = 0

    while steps < max_steps_per_episode:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

        if done:
            failures += 1  # 失败计数
            break  # 结束当前回合

    episode_rewards.append(total_reward)
    print(f"Episode {episode + 1}: Steps = {steps}, Reward = {total_reward}")

# 计算统计信息
avg_reward = sum(episode_rewards) / num_episodes
print("\n===== Evaluation Summary =====")
print(f"Total Episodes: {num_episodes}")
print(f"Failures (Episodes terminated early): {failures}")
print(f"Average Reward: {avg_reward:.2f}")
print(f"Best Reward: {max(episode_rewards):.2f}")
print(f"Worst Reward: {min(episode_rewards):.2f}")

# 关闭环境
env.close()


