import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np

from qiskit import Aer, QuantumCircuit
from qiskit.aqua.algorithms import IterativeAmplitudeEstimation
from qiskit.circuit.library import LogNormalDistribution, LinearAmplitudeFunction

# 表示不确定性的量子位的数量
num_uncertainty_qubits = 3

# 相对随机分布的参数
S = 2.0       # 最初的现货价格
vol = 0.4     # 40%的波动
r = 0.05      # 无风险利率年化为5%
T = 40 / 252  # 40天到期

# 得到的参数为对数正态分布
mu = ((r - 0.5 * vol**2) * T + np.log(S))                     # 平均值
sigma = vol * np.sqrt(T)                                      # 波动率
mean = np.exp(mu + sigma**2/2)                                # 期望值 见下方markdown
variance = (np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2)   # 方差 一组数据内部的离散程度
stddev = np.sqrt(variance)                                    # 标准差

# 考虑现货价格的最低和最高价值;在中间，考虑等距离散化。
low  = np.maximum(0, mean - 3*stddev)
high = mean + 3*stddev

# 构造一个用于QAE的操作符
# 组成不确定性模型和目标
uncertainty_model = LogNormalDistribution(num_uncertainty_qubits, mu=mu, sigma=sigma**2, bounds=(low, high))

# 画出概率分布
x = uncertainty_model.values
y = uncertainty_model.probabilities
# plt.bar(x, y, width=0.2)
# plt.xticks(x, size=15, rotation=90)
# plt.yticks(size=15)
plt.bar(x, y)
plt.xticks(x, size = 5)
plt.yticks(size = 15)
plt.grid()
plt.xlabel('Spot Price at Maturity $S_T$ (\$)', size=15)
plt.ylabel('Probability ($\%$)', size=15)
plt.show()
y_list = y.tolist()
max_pro = max(y)

# 设定执行价格(应在不确定性的低值和高值范围内)
strike_price = round(x[y_list.index(max_pro)], 3)

# 设置成本函数的近似缩放
c_approx = 0.25

# 建立分段线性目标函数
breakpoints = [low, strike_price]
slopes = [0, 1]
offsets = [0, 0]
f_min = 0
f_max = high - strike_price
european_call_objective = LinearAmplitudeFunction(
    num_uncertainty_qubits,
    slopes,
    offsets,
    domain=(low, high),
    image=(f_min, f_max),
    breakpoints=breakpoints,
    rescaling_factor=c_approx
)

# 为支付函数构造一个QAE算子
# 组成不确定性模型和目标
num_qubits = european_call_objective.num_qubits
european_call = QuantumCircuit(num_qubits)
european_call.append(uncertainty_model, range(num_uncertainty_qubits))
european_call.append(european_call_objective, range(num_qubits))

european_call.draw()

y_1 = np.maximum(0, x - strike_price)
plt.plot(x, y_1, 'ro-')
plt.grid()
plt.title('Payoff Function', size=15)
plt.xlabel('Spot Price', size=15)
plt.ylabel('Payoff', size=15)
plt.xticks(x, size=15, rotation=90)
plt.yticks(size=15)
plt.show()

# 计算精确的期望值(归一化到[0,1]区间)
exact_value = np.dot(uncertainty_model.probabilities, y_1)
exact_delta = sum(uncertainty_model.probabilities[x >= strike_price])
print('exact expected value:\t%.4f' % exact_value)
print('exact delta value:   \t%.4f' % exact_delta)

from qiskit.finance.applications import EuropeanCallExpectedValue

european_call_objective = EuropeanCallExpectedValue(num_uncertainty_qubits,
                                                    strike_price,
                                                    rescaling_factor=c_approx,
                                                    bounds=(low, high))

# 将不确定性模型附加到前面
european_call = european_call_objective.compose(uncertainty_model, front=True)

# 设定目标精度和置信度
epsilon = 0.01
alpha = 0.05

# 构造幅度估计
ae = IterativeAmplitudeEstimation(epsilon=epsilon, alpha=alpha,
                                  state_preparation=european_call,
                                  objective_qubits=[3],
                                  post_processing=european_call_objective.post_processing)
result = ae.run(quantum_instance=Aer.get_backend('qasm_simulator'), shots=1000)

conf_int = np.array(result['confidence_interval'])
print('Exact value:        \t%.4f' % exact_value)
print('Estimated value:    \t%.4f' % (result['estimation']))
print('Confidence interval:\t[%.4f, %.4f]' % tuple(conf_int))

from qiskit.finance.applications import EuropeanCallDelta

european_call_delta = EuropeanCallDelta(num_uncertainty_qubits, strike_price, bounds=(low, high))

european_call_delta.decompose().draw()

state_preparation = QuantumCircuit(european_call_delta.num_qubits)
state_preparation.append(uncertainty_model, range(uncertainty_model.num_qubits))
state_preparation.append(european_call_delta, range(european_call_delta.num_qubits))
state_preparation.draw()

# set target precision and confidence level
epsilon = 0.01
alpha = 0.05

# construct amplitude estimation
ae_delta = IterativeAmplitudeEstimation(epsilon=epsilon, alpha=alpha,
                                        state_preparation=state_preparation,
                                        objective_qubits=[num_uncertainty_qubits])

result_delta = ae_delta.run(quantum_instance=Aer.get_backend('qasm_simulator'), shots=100000)

conf_int = np.array(result_delta['confidence_interval'])
print('Exact delta:    \t%.4f' % exact_delta)
print('Esimated value: \t%.4f' % result_delta['estimation'])
print('Confidence interval: \t[%.4f, %.4f]' % tuple(conf_int))