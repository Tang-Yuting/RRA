import numpy as np
import gymnasium as gym
from tqdm import tqdm

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_q(q):
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set_aspect('equal')
    ax.set_xlim(-.5, 11.5)
    ax.set_ylim(-.5, 3.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()

    norm = mpl.colors.Normalize(vmin=q.min(), vmax=q.max())
    cmap = mpl.colormaps['RdYlGn']

    da = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]) * .5
    db = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]]) * .3

    for i in range(12):
        for j in range(4):
            observation = j * 12 + i
            # ax.text(i, j, str(observation), color='w', ha='center', va='center')
            m = q[observation].max()
            for k in range(4):
                center = np.array([i, j])
                ax.add_patch(Polygon([center, center + da[k], center + da[k + 1]],
                                     fc=cmap(norm(q[observation, k])),
                                     ec='k' if q[observation, k] == m else 'none', lw=2,
                                     zorder=1 if q[observation, k] == m else 0))
                ax.text(*(center + db[k]), s=f'{q[observation, k]:.1f}',
                        ha='center', va='center',
                        zorder=2)

    # colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size='2%', pad=0.05)  # 5% width, 0.05 pad
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)

    fig.tight_layout()
    fig.show()


def epsilon_greedy(qs, epsilon=0.3):
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.randint(len(qs))
    else:
        action = int(np.argmax(qs))
    return action


def update(tau, observation, action, reward, next_observation, act, post, lr=0.01):
    next_action = post(act(reward, tau[next_observation])).argmax()
    estimation = act(reward, tau[next_observation, next_action])
    tau[observation, action] += lr * (estimation - tau[observation, action])


env = gym.make('CliffWalking-v0')

# discounted sum
# init = np.array([0.])
# act = lambda r, s: r + .95 * s
# post = np.squeeze

# discounted max
# init = np.array([-500.])
# act = lambda r, s: np.maximum(r, 0.95 * s)
# post = np.squeeze

# mean
init = np.array([1e-3, 0.])
act = lambda r, tau: tau + np.array([1, r])
post = lambda tau: tau[..., 1] / tau[..., 0]

tau = np.repeat(np.repeat(init[None, None,], env.action_space.n, axis=1), env.observation_space.n, axis=0)

lr = 1

for episode in tqdm(range(100)):
    if episode % 10 == 0:
        plot_q(post(tau))

    observation, info = env.reset()
    done = False
    while not done:
        action = epsilon_greedy(post(tau[observation]))
        next_observation, reward, terminated, truncated, info = env.step(action)

        # boundary
        # if observation == next_observation:
        #     reward = -100
        # goal
        if terminated:
            reward = 100

        update(tau, observation, action, reward, next_observation, act, post, lr=lr)
        done = terminated or truncated
        observation = next_observation
env.close()
