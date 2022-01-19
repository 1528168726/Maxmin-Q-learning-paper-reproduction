import numpy as np
import tqdm
import datetime
import multiprocessing as mp
import matplotlib.pyplot as plt

epsilon = 0.1  # 贪婪系数
discountFactor = 1  # 折扣因子
learningRate = 0.01  # 学习率
tryStep = 10000
run_time = 2500
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
    def __init__(self, u):
        self.Q = np.random.normal(loc=0, scale=0.01, size=(2, 8))
        self.u = u

    def learn(self, count_choice):
        totalStep = 0
        replyBuffer = ReplyBuffer(5)
        while totalStep < tryStep:
            state = 1
            reachEnd = False
            while not reachEnd:
                # 选择决策
                action = self.chooseAction(state)
                # 得到决策的下一状态
                nextState, reward, reachEnd = self.stepAction(state, action)

                # push and sample
                replyBuffer.push([state, action, reward, nextState, reachEnd])
                [state, action, reward, nextState, reachEnd] = replyBuffer.sample()
                # 更新Q
                tdError = reward + discountFactor * self.Qmax(nextState, action) - self.Q[state, action]
                self.Q[state, action] += learningRate * tdError

                # 更新状态
                state = nextState

            self.update_count_choice(count_choice, totalStep, state)
            totalStep += 1


    def chooseAction(self, state):
        if state == 1:
            if np.random.rand() < epsilon:
                return np.random.randint(0, 2)
            return np.argmax(self.Q[state, 0:2])
        else:
            if np.random.rand() < epsilon:
                return np.random.randint(0, 8)
            else:
                return np.argmax(self.Q[state])

    def stepAction(self, state, action):
        if state == 1:
            if action == 0:
                return 0, 0, False
            return 2, 0, True
        else:
            reward = self.u + np.random.uniform(-1, 1)
            return -1, reward, True

    def Qmax(self, nextState, action):
        if nextState == 0:
            return np.max(self.Q[0])
        return 0

    def update_count_choice(self, count_choice, totalStep, state):
        if state == 2:
            count_choice[totalStep] += 1


class DoubleQlearning:
    def __init__(self, u):
        self.QA = np.random.normal(loc=0, scale=0.1, size=(2, 8))
        self.QB = np.random.normal(loc=0, scale=0.1, size=(2, 8))
        self.u = u

    def learn(self, count_choice):
        totalStep = 0
        replyBuffer = ReplyBuffer(5)
        while totalStep < tryStep:
            state = 1
            reachEnd = False
            while not reachEnd:
                # 选择决策
                action = self.chooseAction(state)
                # 得到决策的下一状态
                nextState, reward, reachEnd = self.stepAction(state, action)
                # push and sample
                replyBuffer.push([state, action, reward, nextState, reachEnd])
                [state, action, reward, nextState, reachEnd] = replyBuffer.sample()
                # 更新Q
                if np.random.rand() < 0.5:
                    tdError = reward + discountFactor * self.Qmax('A', nextState) - self.QA[state, action]
                    self.QA[state, action] += learningRate * tdError
                else:
                    tdError = reward + discountFactor * self.Qmax('B', nextState) - self.QB[state, action]
                    self.QB[state, action] += learningRate * tdError

                # 更新状态
                state = nextState

            self.update_count_choice(count_choice, totalStep, state)
            totalStep += 1

    def chooseAction(self, state):
        if state == 1:
            if np.random.rand() < epsilon:
                return np.random.randint(0, 2)
            return np.argmax(self.QA[state, 0:2] + self.QB[state, 0:2])
        else:
            if np.random.rand() < epsilon:
                return np.random.randint(0, 8)
            else:
                return np.argmax(self.QA[state, 0:8] + self.QB[state, 0:8])

    def stepAction(self, state, action):
        if state == 1:
            if action == 0:
                return 0, 0, False
            return 2, 0, True
        else:
            reward = self.u + np.random.uniform(-1, 1)
            return -1, reward, True

    def Qmax(self, choice, nextState):
        if nextState == 0:
            if choice == 'A':
                pos = np.argmax(self.QA[nextState])
                return self.QB[nextState, pos]
            else:
                pos = np.argmax(self.QB[nextState])
                return self.QA[nextState, pos]
        return 0

    def update_count_choice(self, count_choice, totalStep, state):
        if state == 2:
            count_choice[totalStep] += 1


class MaxminQlearning:
    def __init__(self, u, N=1):
        self.Q = np.random.normal(loc=0, scale=0.01, size=(N, 2, 8))
        self.N = N
        self.u = u

    def learn(self, count_choice):
        totalStep = 0
        replyBuffer = ReplyBuffer(5)
        while totalStep < tryStep:
            state = 1
            reachEnd = False
            while not reachEnd:
                minQ = self.selectQ(state)
                # 选择决策
                action = self.chooseAction(minQ, state)
                # 得到决策的下一状态
                nextState, reward, reachEnd = self.stepAction(state, action)

                # push and sample
                replyBuffer.push([state, action, reward, nextState, reachEnd])
                [state, action, reward, nextState, reachEnd] = replyBuffer.sample()
                # 更新Q
                updateQ = np.random.randint(0, self.N)
                tdError = reward + discountFactor * self.Qmax(minQ, nextState) - self.Q[updateQ, state, action]
                self.Q[updateQ, state, action] += learningRate * tdError

                # 更新状态
                state = nextState

            self.update_count_choice(count_choice, totalStep, state)
            totalStep += 1


    def selectQ(self, state):
        if state == 1:
            return int(np.argmin(self.Q[:, state, :2]) / 2)
        else:
            return int(np.argmin(self.Q[:, state, :]) / 8)

    def chooseAction(self, curQ, state):
        if state == 1:
            if np.random.rand() < epsilon:
                return np.random.randint(0, 2)
            return np.argmax(self.Q[curQ, state, 0:2])
        else:
            if np.random.rand() < epsilon:
                return np.random.randint(0, 8)
            else:
                return np.argmax(self.Q[curQ, state])

    def stepAction(self, state, action):
        if state == 1:
            if action == 0:
                return 0, 0, False
            return 2, 0, True
        else:
            reward = self.u + np.random.uniform(-1, 1)
            return -1, reward, True

    def Qmax(self, minQ, nextState):
        if nextState == 0:
            return np.max(self.Q[minQ, 0])
        return 0

    def update_count_choice(self, count_choice, totalStep, state):
        if state == 2:
            count_choice[totalStep] += 1


def QlearningMain(i, lock, array, u, N):
    pbar = tqdm.tqdm(total=run_time, postfix=i)
    pbar.set_description("processing " + str(i) + " :")
    count_choice = np.zeros(tryStep)
    for i in range(0, run_time):
        learning = Qlearning(u)
        learning.learn(count_choice)
        pbar.update(1)
    with lock:
        for i in range(0, tryStep):
            array[i] += int(count_choice[i])
    return


def DoubleQlearningMain(i, lock, array, u, N):
    pbar = tqdm.tqdm(total=run_time, postfix=i)
    pbar.set_description("processing " + str(i) + " :")
    count_choice = np.zeros(tryStep)
    for i in range(0, run_time):
        learning = DoubleQlearning(u)
        learning.learn(count_choice)
        pbar.update(1)
    with lock:
        for i in range(0, tryStep):
            array[i] += int(count_choice[i])
    return


def MaxminQlearningMain(i, lock, array, u, N):
    pbar = tqdm.tqdm(total=run_time, postfix=i)
    pbar.set_description("processing " + str(i) + " :")
    count_choice = np.zeros(tryStep)
    for i in range(0, run_time):
        learning = MaxminQlearning(u, N=N)
        learning.learn(count_choice)
        pbar.update(1)
    with lock:
        for i in range(0, tryStep):
            array[i] += int(count_choice[i])
    return


def MlutCoreRun(func, u, N=1, info=""):
    print(info)
    manager = mp.Manager()
    managed_locker = manager.Lock()
    managed_array = manager.Array('i', [0 for i in range(tryStep)])

    jobs = []
    for i in range(0, num_cores):
        p = mp.Process(target=func, args=(i, managed_locker, managed_array, u, N))
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()
    return np.array(managed_array)


def test(u):
    if (u > 0):
        target = epsilon / 2
    else:
        target = (-epsilon / 2 + 1)
    plt.figure()
    plt.title("u=%lf" % u)
    Qp = MlutCoreRun(QlearningMain, u, info="Qlearning u=%lf" % u)
    DoubleQ = MlutCoreRun(DoubleQlearningMain, u, info="DoubleQ u=%lf" % u)
    MaxminQ2 = MlutCoreRun(MaxminQlearningMain, u, N=2, info="Maxmin N=2 u=%lf" % u)
    MaxminQ4 = MlutCoreRun(MaxminQlearningMain, u, N=4, info="Maxmin N=4 u=%lf" % u)
    MaxminQ6 = MlutCoreRun(MaxminQlearningMain, u, N=6, info="Maxmin N=6 u=%lf" % u)
    MaxminQ8 = MlutCoreRun(MaxminQlearningMain, u, N=8, info="Maxmin N=8 u=%lf" % u)
    plt.plot(range(0, tryStep, 10), np.abs(Qp / (run_time * num_cores) - target)[::10], label="Qlearning")
    plt.plot(range(0, tryStep, 10), np.abs(DoubleQ / (run_time * num_cores) - target)[::10], label="DoubleQ")
    plt.plot(range(0, tryStep, 10), np.abs(MaxminQ2 / (run_time * num_cores) - target)[::10], label="Maxmin N=2")
    plt.plot(range(0, tryStep, 10), np.abs(MaxminQ4 / (run_time * num_cores) - target)[::10], label="Maxmin N=4")
    plt.plot(range(0, tryStep, 10), np.abs(MaxminQ6 / (run_time * num_cores) - target)[::10], label="Maxmin N=6")
    plt.plot(range(0, tryStep, 10), np.abs(MaxminQ8 / (run_time * num_cores) - target)[::10], label="Maxmin N=8")
    plt.legend()
    plt.savefig("u=%lf.jpg"%u)
    plt.show()


if __name__ == '__main__':
    test(0.1)
    test(-0.1)
    # for i in np.arange(0.04,0.11,0.01):
    #     test(i)
    # for i in np.arange(-0.10,-0.04,0.01):
    #     test(i)

    # pbar = tqdm.tqdm(total=run_time)
    # pbar.set_description("processing " + " :")
    # count_choice = np.zeros(tryStep)
    # learning = MaxminQlearning(0.1,N=2)
    # learning.learn(count_choice, pbar)

    print(1)
