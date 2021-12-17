# LOOP_MAIN_qubit
# version:0.1.2
import pandas
import time
from scipy.stats import lognorm
from sklearn.preprocessing import StandardScaler
import numpy as np
seed = 83
np.random.seed = seed

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

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

from PLOT_qubit import *

### 基础数据
N = 2000                # 样本数
epoch = 8               # 轮数
batch = int(N/epoch)    # 每轮样本
strike_price = 2        # 設置執行價格(應該在不確定性的最低值和最高值範圍內)
c_approx = 0.25         # 設置成本函數的近似縮放
epsilon = 0.01          # 目標精度
alpha = 0.05            # 信賴區間

### 导入数据
OUTPUT_DATA = []
origin_data = pandas.read_csv("ttt.csv", names=["price"])
scaler = StandardScaler()
data_list = scaler.fit_transform(origin_data) # 将数据更改使得数据可以被fit
data_arr = np.array(data_list)
actual_data = data_arr.astype(float)
sigma,loc,scale = stats.lognorm.fit(actual_data)

# 正规化
estimated_mu = np.log(scale)
estimated_sigma = sigma
real_data = np.random.lognormal(mean=estimated_mu, sigma=estimated_sigma, size=N)
Dmin = real_data.min()
Dmax = real_data.max()

algorithm_globals.random_seed = seed

#for qubit_num in range(3, 8 + 1):
for qubit_num in range(3,4):
    new_data = []
    for i in real_data:
        new = (i - Dmin) / (Dmax - Dmin) * (2 ** qubit_num - 1)
        new_data.append(new)
    # print(new_data)

    # 根據資料的維度設置量子位元為 k 個量子位元串列[#q_0,...,#q_k-1]
    load_num_qubits = [qubit_num]
    k = len(load_num_qubits)

    # 設置數據解析度
    # 設置最高和最低資料數值作為 k 最小/最大數據值的串列
    load_bounds = np.array([0, (2 ** load_num_qubits[0]) -1])

    # 此处每次跑1000次比较各种数据
    for TMS in range(1000):
        THIS_DATA = []                  # this data 将记录本轮数据的结果
        THIS_DATA.append(TMS)           # this data 加入次数
        THIS_DATA.append(qubit_num)     # this data 加入qubit num数
        THIS_DATA.append(N)             # this data 加入N
        THIS_DATA.append(epoch)         # this data 加入epoch
        THIS_DATA.append(batch)         # this data 加入batch
        THIS_DATA.append(strike_price)  # this data 加入strike_price
        THIS_DATA.append(c_approx)      # this data 加入c_approx
        THIS_DATA.append(epsilon)       # this data 加入epsilon
        THIS_DATA.append(alpha)         # this data 加入alpha
        
        start = time.process_time()
        # 初始化qGAN
        qgan = QGAN(new_data, load_bounds, load_num_qubits, batch, epoch, snapshot_dir=None)
        qgan.seed = 86

        # 设置 quantum instance 以执行量子生成器
        quantum_instance = QuantumInstance(backend=BasicAer.get_backend('statevector_simulator'),
                                       seed_transpiler=seed, seed_simulator=seed)

        init_dist = UniformDistribution(sum(load_num_qubits))
        ansatz = TwoLocal(int(np.sum(load_num_qubits)), 'ry', 'cz', 'linear', reps=1, insert_barriers=True)

        # 設置初始參數用於減少訓練時間進而降低DG的執行時間
        # 你可以提升人工智慧訓練型樣的數量並使用隨機初始參數
        init_params = np.random.rand(ansatz.num_parameters_settable) * 2 * np.pi
        load_g_circuit = ansatz.compose(init_dist, front=True)

        # 設置量子生成器
        qgan.set_generator(generator_circuit=load_g_circuit, generator_init_params=init_params)

        # 參數具有順序問題下列是暫時的解決方法
        qgan._generator._free_parameters = sorted(load_g_circuit.parameters, key=lambda p: p.name)
        
        # 設置經典判別器神經網路
        discriminator = NumPyDiscriminator(len(load_num_qubits))
        qgan.set_discriminator(discriminator)

        #执行qGAN
        result = qgan.run(quantum_instance)

        # 显示结果D和G的系数
        print('Training results: {} times'.format(TMS))
        for key, value in result.items():
            print(f'  {key} : {value}')

        # plot_qgan_loss(qgan,epoch)

        # plot_Relative_Entropy(qgan,epoch)

        # plot_qgan_CDF(qgan,load_bounds,)
        
        # THIS_DATA.append(result['params_g'])
        THIS_DATA.append(result['loss_d'])
        THIS_DATA.append(result['loss_g'])
        
        g_params = result['params_g']
        init_dist = NormalDistribution(qubit_num, mu=estimated_mu, sigma=estimated_sigma, bounds=load_bounds)

        # 構造變分形式
        var_form = TwoLocal(qubit_num, 'ry', 'cz', entanglement='circular', reps=1)

        # 保留一個參數列表，以便我們可以將它們與數值列表關聯起來
        # (否則我們需要一個字典)
        theta = var_form.ordered_parameters

        # 組成生成器電路，這是加載不確定性模型的電路
        op_g_circuit = init_dist.compose(var_form)

        # 為成本函數構造電路
        european_call_objective = EuropeanCallExpectedValue(
            qubit_num,
            strike_price=strike_price,
            rescaling_factor=c_approx,
            bounds=load_bounds
        )

        # 評估訓練過的概率分佈
        values = [load_bounds[0] + (load_bounds[1] - load_bounds[0]) * x / (2 ** qubit_num - 1) for x in range(2**qubit_num)]
        uncertainty_model = op_g_circuit.assign_parameters(dict(zip(theta, g_params)))
        amplitudes = Statevector.from_instruction(uncertainty_model).data
        
        x = np.array(values)
        # print(x)
        y = np.abs(amplitudes) ** 2
        # print(y)

        # 從目標概率分佈中抽取樣本
        N = 10000
        log_normal = np.random.lognormal(mean=estimated_mu, sigma=estimated_sigma, size=N)
        print('1:',log_normal)
        log_normal = np.round(log_normal)
        print('2:',log_normal)
        log_normal = log_normal[log_normal <= (2 ** qubit_num) -1]
        print('3:',log_normal)
        log_normal_samples = []
        for i in range(2 ** qubit_num):
            log_normal_samples += [np.sum(log_normal==i)]
        log_normal_samples = np.array(log_normal_samples / sum(log_normal_samples))
        print(log_normal_samples)
        
        # plot_distribution(x,y,log_normal_samples)

        # 評估不同發行版本的收益
        # print(log_normal_samples)
        # print(len(y))
        payoff = []
        for i in range(2 ** qubit_num):
            if i <= strike_price:
                payoff.append(0)
            else:
                payoff.append(i - strike_price)
        payoff = np.array(payoff)
        # print(payoff)

        ep = np.dot(log_normal_samples, payoff)
        print("Analytically calculated expected payoff w.r.t. the target distribution:  %.4f" % ep)
        ep_trained = np.dot(y, payoff)
        print("Analytically calculated expected payoff w.r.t. the trained distribution: %.4f" % ep_trained)
        
        # 导入预期
        THIS_DATA.append(ep)
        THIS_DATA.append(ep_trained)

        # plot_distribution(x,y,log_normal_samples)

        # 為QAE構造一個操作符
        european_call = european_call_objective.compose(uncertainty_model, front=True)

        # 構造振幅估計
        ae = IterativeAmplitudeEstimation(epsilon=epsilon, alpha=alpha,
                                        state_preparation=european_call,
                                        objective_qubits=[qubit_num],
                                        post_processing=european_call_objective.post_processing)

        End_result = ae.run(quantum_instance=Aer.get_backend('qasm_simulator'), shots=1000)
        
        conf_int = np.array(End_result['confidence_interval'])
        print('Exact value:        \t%.4f' % ep_trained)
        print('Estimated value:    \t%.4f' % (End_result['estimation']))
        print('Confidence interval:\t[%.4f, %.4f]' % tuple(conf_int))
        
        THIS_DATA.append(End_result['estimation'])
        THIS_DATA.append(conf_int)
        

        end = time.process_time()

        print("執行時間：%f 秒" % (end - start))
        
        THIS_DATA.append(end - start)
        print(THIS_DATA)
        
        OUTPUT_DATA.append(THIS_DATA)