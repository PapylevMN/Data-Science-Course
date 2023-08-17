import random
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import warnings
import statistics


class QLearning:
    def __init__(self, state_n, action_n, alpha=0.5, gamma=0.99, epsilon=0.1):
        self.state_n = state_n
        self.action_n = action_n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Инициализация q-функции нулями
        self.qfunction = np.zeros((self.state_n, self.action_n))

    def get_action(self, state):
        # Выбор действия на основе эпсилон-жадной стратегии
        if random.random() < self.epsilon:
            return random.randint(0, self.action_n - 1)
        else:
            return np.argmax(self.qfunction[state])

    def update_qfunction(self, state, action, reward, next_state):
        # Обновление q-функции с использованием алгоритма Q-learning
        current_q = self.qfunction[state][action]
        next_q = np.max(self.qfunction[next_state])
        updated_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
        self.qfunction[state][action] = updated_q

def main():
    warnings.filterwarnings('ignore')

    # Создание окружения сетки
    env = gym.make("Taxi-v3")

    state_n = env.observation_space.n
    action_n = env.action_space.n
       
    # Создание агента Q-learning с определенным количеством состояний и действий
    agent = QLearning(state_n, action_n)
    episode_n = 500
    trajectory_len = 200
    total_reward = []
    last_reward = []
    for i in range(episode_n):
        obs = env.reset()
        state = obs[0]
        episode_reward = 0
        for _ in range(trajectory_len):
            # Выбор действия на основе текущего состояния
            action = agent.get_action(state)
            
            # Выполнение выбранного действия и получение нового состояния и вознаграждения
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward
        
            # Обновление q-функции по алгоритму Q-learning
            agent.update_qfunction(state, action, reward, next_state)

            # Переход к следующему состоянию
            state = next_state

            if done:
                break
        
        if i >= episode_n - 100:
            last_reward.append(episode_reward)
            total_reward.append(statistics.mean(last_reward))
        else:
            total_reward.append(episode_reward)
    
    print(f'Сумма вознаграждений последних 100 эпизодов: {round(sum(total_reward[-100:]),1)}')
    
    plt.plot(total_reward)
    plt.show()
    

if __name__ == '__main__':
    main()
