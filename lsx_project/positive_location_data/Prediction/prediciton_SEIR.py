# -*- codeing = utf-8 -*- 
# @Time :2022/5/8 21:47 
# @Author:Liujie 
# @File : SEIR.py 
# @Software: PyCharm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pylab


# 计算SEIR的值
def calc(T):
    for i in range(0, len(T) - 1):
        S.append(S[i] - r * beta * S[i] * I[i] / N - r2 * beta2 * S[i] * E[i] / N)
        E.append(E[i] + r * beta * S[i] * I[i] / N + r2 * beta2 * S[i] * E[i] / N - alpha * E[i])
        I.append(I[i] + alpha * E[i] - gamma * I[i])  # 计算累计确诊人数
        R.append(R[i] + gamma * I[i])


# 画图
def plot(T, S, E, I, R):
    plt.figure()
    plt.title("SEIR-Time Curve of Virus Transmission")
    plt.plot(T, S, color='r', label='Susceptible')
    plt.plot(T, E, color='k', label='Exposed')
    plt.plot(T, I, color='b', label='Infected')
    plt.plot(T, R, color='g', label='Recovered')

    plt.grid(False)
    plt.legend()
    plt.xlabel("Time(day)")
    plt.ylabel("Population")
    plt.show()


S, E, I, R = [], [], [], []
N = 10000  # 人口总数
I.append(1)
S.append(N - I[0])
E.append(0)
R.append(0)
r = 20  # 传染者接触人数
r2 = 30
beta = 0.03  # 传染者传染概率
beta2 = 0.03  # 易感染者被潜伏者感染的概率
alpha = 0.14  # 潜伏者患病概率 1/7
gamma = 0.1  # 康复概率

pylab.rcParams['figure.figsize'] = (12.0, 7.0)
T = [i for i in range(0, 100)]
calc(T)
plot(T, S, E, I, R)

'''
————————————————
版权声明：本文为CSDN博主「久菜河子」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_45347184/article/details/124719806
'''