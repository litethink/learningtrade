import math
import numpy as np
np.random.seed(20000)
t0 = time()

#S0: float
#        标的物初始价格水平
#    K: float
#       行权价格
#    T: float
#       到期日
#    r: float
#       固定无风险短期利率
#    sigma: float 不能直接观察
#       波动因子
import numpy as np
from math import sqrt, log
from scipy import stats
#
# 欧式期权BSM定价公式

def bsm_call_value(S0, K, T, r, sigma):

    S0 = float(S0)
    d1 = (np.log(S0 /K) + (r + 0.5 * sigma**2) * T )/(sigma * np.sqrt(T))
    d2 = (np.log(S0 /K) + (r - 0.5 * sigma**2) * T )/(sigma * np.sqrt(T))
    value = (S0 * stats.norm.cdf(d1, 0, 1) - K * np.exp(-r * T) * stats.norm.cdf(d2, 0, 1))
    return value
    
def bsm_vega(S0, K, T, r, sigma):
    """
    Vega 计算
    """
    S0 = float(S0)
    d1 = (np.log(S0/K)) + (r+0.5*sigma**2)*T /(sigma*sqrt(T))
    vega = S0 * stats.norm.cdf(d1, 0, 1) * np.sqrt(T)
    return vega

def bsm_call_imp_vol(S0, K, T, r, C0, sigma_est, it=100):
    for i in range(it):
        sigma_est -= ((bsm_call_value(S0, K, T, r, sigma_est) - C0)
                     / bsm_vega(S0, K, T, r, sigma_est))
    return sigma_est




S0 = 100; K=105; T=1.; r=0.05; sigma=0.2
S = S0 * np.exp(np.cumsum((r-0.5*sigma**2)*dt + sigma * math.sqrt(dt)
                       * np.random.standard_normal((M+1, I)), axis=0
                      ))
S[0] = S0

#欧式期权定价
C0 = math.exp(-r*T) * np.sum(np.maximum(S[-1]-K, 0))/I

print(f'欧式期权定价 {C0}.')
print(f'共计花费时间 {np.round(time()-t0,1) }s.')

plt.hist(S[-1], bins=50)
plt.grid(True)
plt.xlabel('index level')
plt.ylabel('frequency')
