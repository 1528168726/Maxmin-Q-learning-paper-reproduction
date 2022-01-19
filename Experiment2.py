import math

import numpy as np
import gym
import tqdm
import datetime
import multiprocessing as mp
import matplotlib.pyplot as plt

epsilon = 0.1  # 贪婪系数
discountFactor = 1  # 折扣因子
learningRate = 0.2  # 学习率
tryEpisodes = 1000
maxStep = 1000
run_time = 1
num_cores = 6
replay_buffer_size = 100


class ReplyBuffer:
    def __init__(self, size):
        self.buffer = [None for i in range(replay_buffer_size)]
        self.index = 0
        self.isFull = False
        self.shape = (replay_buffer_size, size)

    def push(self, data):
        self.buffer[self.index] = data
        self.index = (self.index + 1) % self.shape[0]
        if not self.isFull and self.index == 0:
            self.isFull = True

    def sample(self):
        if self.isFull:
            return self.buffer[np.random.randint(0, self.shape[0])]
        else:
            return self.buffer[np.random.randint(0, self.index)]

    def clear(self):
        self.index = 0
        self.isFull = False


class Qlearning:
    def __init__(self, sigma):
        self.sigma = sigma
        self.env = gym.make('MountainCar-v0').env
        self.actions = self.env.action_space
        self.num_pos = 20  # 将位置分为num_pos份
        self.num_vel = 15  # 将速度分为num_vel份
        self.Q = np.random.normal(loc=0, scale=0.01, size=(self.num_pos * self.num_vel, self.actions.n))
        self.pos_bins = self.toBins(-1.2, 0.6, self.num_pos)
        self.vel_bins = self.toBins(-0.07, 0.07, self.num_vel)

    def learn(self):
        countStep = np.zeros(tryEpisodes)
        replyBuffer = ReplyBuffer(5)
        for i in range(tryEpisodes):
            observation = self.env.reset()
            state = self.digitizeState(observation)
            # replyBuffer.clear()
            done=False
            for t in range(maxStep):
                action = self.chooseAction(state)
                nextState, reward, done = self.stepAction(action)

                # push and sample
                # replyBuffer.push([state, action, reward, nextState, done])
                # [state, action, reward, nextState, done] = replyBuffer.sample()

                # 更新Q
                tdError = reward + discountFactor * self.Qmax(nextState) - self.Q[state, action]
                self.Q[state, action] += learningRate * tdError

                # 更新状态
                state = nextState
                if done:
                    countStep[i] = t
                    break
            if not done:
                countStep[i] = maxStep
        return countStep

    def chooseAction(self, state):
        if np.random.random() < epsilon:
            return self.actions.sample()
        else:
            return np.argmax(self.Q[state])

    def stepAction(self, action):
        observation, reward, done, info = self.env.step(action)
        reward = np.random.normal(loc=-1, scale=self.sigma)
        nextState = self.digitizeState(observation)
        return nextState, reward, done

    def Qmax(self, nextState):
        return np.max(self.Q[nextState])

    # 分箱处理函数，把[clip_min,clip_max]区间平均分为num段，
    def toBins(self, clip_min, clip_max, num):
        return np.linspace(clip_min, clip_max, num + 1)

    def digit(self, x, bin):
        n = np.digitize(x, bins=bin)
        if x == bin[-1]:
            n = n - 1
        return n

    # 将观测值observation离散化处理
    def digitizeState(self, observation):
        # 将矢量打散回连续特征值
        cart_pos, cart_v = observation
        # 分别对各个连续特征值进行离散化（分箱处理）
        digitized = [self.digit(cart_pos, self.pos_bins),
                     self.digit(cart_v, self.vel_bins), ]
        # 将4个离散值再组合为一个离散值，作为最终结果
        return (digitized[1] - 1) * self.num_pos + digitized[0] - 1

    def clear(self):
        self.Q = np.random.normal(loc=0, scale=0.01, size=(self.num_pos * self.num_vel, self.actions.n))


class MaxminQlearning:
    def __init__(self, sigma, N=2):
        self.sigma = sigma
        self.N = N
        self.env = gym.make('MountainCar-v0').env
        self.actions = self.env.action_space
        self.num_pos = 20  # 将位置分为num_pos份
        self.num_vel = 15  # 将速度分为num_vel份
        self.Q = np.random.normal(loc=0, scale=0.01, size=(self.N, self.num_pos * self.num_vel, self.actions.n))
        self.pos_bins = self.toBins(-1.2, 0.6, self.num_pos)
        self.vel_bins = self.toBins(-0.07, 0.07, self.num_vel)

    def learn(self):
        countStep = np.zeros(tryEpisodes)
        replyBuffer = ReplyBuffer(5)
        for i in range(tryEpisodes):
            observation = self.env.reset()
            state = self.digitizeState(observation)
            replyBuffer.clear()
            for t in range(maxStep):
                minQ = self.selectQ(state)
                action = self.chooseAction(minQ, state)
                nextState, reward, done = self.stepAction(action)

                # push and sample
                # replyBuffer.push([state, action, reward, nextState, done])
                # if not done:
                #     [state, action, reward, nextState, done] = replyBuffer.sample()

                # 更新Q
                updateQ = np.random.randint(0, self.N)
                tdError = reward + discountFactor * self.Qmax(minQ, nextState) - self.Q[updateQ, state, action]
                self.Q[updateQ, state, action] += learningRate * tdError

                if done:
                    countStep[i] = t
                    break
                # 更新状态
                state = nextState
            if countStep[i] == 0:
                countStep[i] = maxStep
        return countStep

    def selectQ(self, state):
        return int(np.argmin(self.Q[:, state, :]) / self.actions.n)

    def chooseAction(self, curQ, state):
        if np.random.random() < epsilon:
            return self.actions.sample()
        else:
            return np.argmax(self.Q[curQ, state])

    def stepAction(self, action):
        observation, reward, done, info = self.env.step(action)
        reward = np.random.normal(loc=-1, scale=self.sigma)
        nextState = self.digitizeState(observation)
        return nextState, reward, done

    def Qmax(self, minQ, nextState):
        return np.max(self.Q[minQ, nextState])

    # 分箱处理函数，把[clip_min,clip_max]区间平均分为num段，
    def toBins(self, clip_min, clip_max, num):
        return np.linspace(clip_min, clip_max, num + 1)

    def digit(self, x, bin):
        n = np.digitize(x, bins=bin)
        if x == bin[-1]:
            n = n - 1
        return n

    # 将观测值observation离散化处理
    def digitizeState(self, observation):
        # 将矢量打散回连续特征值
        cart_pos, cart_v = observation
        # 分别对各个连续特征值进行离散化（分箱处理）
        digitized = [self.digit(cart_pos, self.pos_bins),
                     self.digit(cart_v, self.vel_bins), ]
        # 将4个离散值再组合为一个离散值，作为最终结果
        return (digitized[1] - 1) * self.num_pos + digitized[0] - 1

    def clear(self):
        self.Q = np.random.normal(loc=0, scale=0.01, size=(self.num_pos * self.num_vel, self.actions.n))

class DoubleQlearning:
    def __init__(self, sigma):
        self.sigma = sigma
        self.env = gym.make('MountainCar-v0').env
        self.actions = self.env.action_space
        self.num_pos = 20  # 将位置分为num_pos份
        self.num_vel = 15  # 将速度分为num_vel份
        self.QA = np.random.normal(loc=0, scale=0.01, size=(self.num_pos * self.num_vel, self.actions.n))
        self.QB = np.random.normal(loc=0, scale=0.01, size=(self.num_pos * self.num_vel, self.actions.n))
        self.pos_bins = self.toBins(-1.2, 0.6, self.num_pos)
        self.vel_bins = self.toBins(-0.07, 0.07, self.num_vel)

    def learn(self):
        countStep = np.zeros(tryEpisodes)
        replyBuffer = ReplyBuffer(5)
        for i in range(tryEpisodes):
            observation = self.env.reset()
            state = self.digitizeState(observation)
            # replyBuffer.clear()
            for t in range(maxStep):
                action = self.chooseAction(state)
                nextState, reward, done = self.stepAction(action)

                # push and sample
                # replyBuffer.push([state, action, reward, nextState, done])
                # [state, action, reward, nextState, done] = replyBuffer.sample()

                # 更新Q
                if np.random.rand() < 0.5:
                    tdError = reward + discountFactor * self.Qmax('A',nextState) - self.QA[state, action]
                    self.QA[state, action] += learningRate * tdError
                else:
                    tdError = reward + discountFactor * self.Qmax('B',nextState) - self.QB[state, action]
                    self.QB[state, action] += learningRate * tdError

                # 更新状态
                state = nextState
                if done:
                    countStep[i] = t
                    break
            if countStep[i] == 0:
                countStep[i] = maxStep
        return countStep

    def chooseAction(self, state):
        if np.random.random() < epsilon:
            return self.actions.sample()
        else:
            return np.argmax(self.QA[state]+self.QB[state])

    def stepAction(self, action):
        observation, reward, done, info = self.env.step(action)
        reward = np.random.normal(loc=-1, scale=self.sigma)
        nextState = self.digitizeState(observation)
        return nextState, reward, done

    def Qmax(self, choice,nextState):
        if choice == 'A':
            pos = np.argmax(self.QA[nextState])
            return self.QB[nextState, pos]
        else:
            pos = np.argmax(self.QB[nextState])
            return self.QA[nextState, pos]

    # 分箱处理函数，把[clip_min,clip_max]区间平均分为num段，
    def toBins(self, clip_min, clip_max, num):
        return np.linspace(clip_min, clip_max, num + 1)

    def digit(self, x, bin):
        n = np.digitize(x, bins=bin)
        if x == bin[-1]:
            n = n - 1
        return n

    # 将观测值observation离散化处理
    def digitizeState(self, observation):
        # 将矢量打散回连续特征值
        cart_pos, cart_v = observation
        # 分别对各个连续特征值进行离散化（分箱处理）
        digitized = [self.digit(cart_pos, self.pos_bins),
                     self.digit(cart_v, self.vel_bins), ]
        # 将4个离散值再组合为一个离散值，作为最终结果
        return (digitized[1] - 1) * self.num_pos + digitized[0] - 1

    def clear(self):
        self.Q = np.random.normal(loc=0, scale=0.01, size=(self.num_pos * self.num_vel, self.actions.n))

def Run(i, lock, array,func,sigma):
    pbar = tqdm.tqdm(total=run_time, postfix=i)
    pbar.set_description("processing " + str(i) + " :")
    count_choice = np.zeros(tryEpisodes)
    for i in range(0, run_time):
        learning = func(sigma)
        count_choice+=learning.learn()
        pbar.update(1)
    with lock:
        for i in range(0, tryEpisodes):
            array[i] += int(count_choice[i])
    return

def MlutCoreRun(func, sigma, info=""):
    print(info)
    manager = mp.Manager()
    managed_locker = manager.Lock()
    managed_array = manager.Array('i', [0 for i in range(tryEpisodes)])

    jobs = []
    for i in range(0, num_cores):
        p = mp.Process(target=Run, args=(i, managed_locker, managed_array,func,sigma))
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()
    return np.array(managed_array)

def figure2_1():

    resQ=[]
    resMaxmin=[]
    resDouble=[]
    for i in range(6):
        sigma=math.sqrt(i*10)
        res=MlutCoreRun(Qlearning, sigma=sigma, info="Qlearning sigma=%lf" %sigma )
        resQ.append(np.mean(res[-21:-1])/(run_time * num_cores))
    for i in range(6):
        sigma=math.sqrt(i*10)
        res=MlutCoreRun(MaxminQlearning, sigma=sigma, info="Qlearning sigma=%lf" %sigma )
        resMaxmin.append(np.mean(res[-21:-1])/(run_time * num_cores))
    for i in range(6):
        sigma=math.sqrt(i*10)
        res=MlutCoreRun(DoubleQlearning, sigma=sigma, info="Qlearning sigma=%lf" %sigma )
        resDouble.append(np.mean(res[-21:-1])/(run_time * num_cores))

    plt.xlabel('Sigma^2')
    plt.ylabel('Steps')
    plt.plot(range(0, 60, 10), resQ, label="Qlearning")
    plt.plot(range(0, 60, 10), resMaxmin, label="Maxmin N=2")
    plt.plot(range(0, 60, 10), resDouble, label="DoubleQ")
    plt.legend()
    plt.savefig("2.1.jpg")
    plt.show()
    return

def figure2_2():
    sigma=math.sqrt(10)
    # q = Qlearning(sigma=3)
    plt.figure()
    plt.title("sigma^2=%lf" % (sigma*sigma))
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    Q=MlutCoreRun(Qlearning, sigma=sigma, info="Qlearning sigma=%lf" %sigma )
    MaxminQ=MlutCoreRun(MaxminQlearning, sigma=sigma, info="Maxmin Qlearning sigma=%lf" %sigma )
    DoubleQ=MlutCoreRun(DoubleQlearning, sigma=sigma, info="Double Qlearning sigma=%lf" %sigma )
    plt.plot(range(0, tryEpisodes, 10), (Q/(run_time * num_cores))[::10], label="Qlearning")
    plt.plot(range(0, tryEpisodes, 10), (MaxminQ/(run_time * num_cores))[::10], label="Maxmin N=2")
    plt.plot(range(0, tryEpisodes, 10), (DoubleQ/(run_time * num_cores))[::10], label="DoubleQ")
    plt.legend()
    plt.savefig("sigma^2=%lf.jpg"%(sigma*sigma))
    plt.show()

if __name__ == '__main__':
    figure2_1()
