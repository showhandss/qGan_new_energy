import pandas
import time
from scipy.stats import lognorm
import numpy as np
seed = 71
np.random.seed = seed

import matplotlib.pyplot as plt
from scipy import stats

from qiskit import Aer, QuantumRegister, QuantumCircuit, BasicAer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import TwoLocal, UniformDistribution, NormalDistribution

from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms import NumPyDiscriminator, QGAN

from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.aqua.algorithms import IterativeAmplitudeEstimation
from qiskit.finance.applications import EuropeanCallExpectedValue

def plot_qgan_loss(qgan,epochs):
    t_steps = np.arange(epochs)
    plt.figure(figsize=(6,5))
    plt.title("Progress in the loss function")
    plt.plot(t_steps, qgan.g_loss, label='Generator loss function', color='mediumvioletred', linewidth=2)
    plt.plot(t_steps, qgan.d_loss, label='Discriminator loss function', color='rebeccapurple', linewidth=2)
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('time steps')
    plt.ylabel('loss')
    plt.show()
    # print(qgan.g_loss)
    # print(qgan.d_loss)

def plot_Relative_Entropy(qgan,epochs):
    # 繪製基於相對熵的進展
    plt.figure(figsize=(6,5))
    plt.title('Relative Entropy')
    plt.plot(np.linspace(0, epochs, len(qgan.rel_entr)), qgan.rel_entr, color='mediumblue', lw=4, ls=':')
    plt.grid()
    plt.xlabel('time steps')
    plt.ylabel('relative entropy')
    plt.show()

def plot_qgan_CDF(qgan,load_bounds):
    #繪製結果分布對於目標分布，也就是對數正態分布，的累積分布函數
    log_normal = np.random.lognormal(mean=1, sigma=1, size=100000)
    log_normal = np.round(log_normal)
    log_normal = log_normal[log_normal <= load_bounds[1]]
    temp = []
    for i in range(int(load_bounds[1] + 1)):
        temp += [np.sum(log_normal==i)]
    log_normal = np.array(temp / sum(temp))

    plt.figure(figsize=(6,5))
    plt.title('CDF (Cumulative Distribution Function)')
    samples_g, prob_g = qgan.generator.get_output(qgan.quantum_instance, shots=10000)
    samples_g = np.array(samples_g)
    samples_g = samples_g.flatten()
    num_bins = len(prob_g)
    plt.bar(samples_g,  np.cumsum(prob_g), color='royalblue', width= 0.8, label='simulation')
    plt.plot( np.cumsum(log_normal),'-o', label='log-normal', color='deepskyblue', linewidth=4, markersize=12)
    plt.xticks(np.arange(min(samples_g), max(samples_g)+1, 1.0))
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.legend(loc='best')
    plt.show()

def plot_distribution(x,y,log_normal_samples):
    # 繪製分布
    plt.bar(x, y, width=0.2, label='trained distribution', color='royalblue')
    plt.xticks(x, size=15, rotation=90)
    plt.yticks(size=15)
    plt.grid()
    plt.xlabel('Spot Price at Maturity $S_T$ (\$)', size=15)
    plt.ylabel('Probability ($\%$)', size=15)
    plt.plot(log_normal_samples,'-o', color ='deepskyblue', label='target distribution', linewidth=4, markersize=12)
    plt.legend(loc='best')
    plt.show()