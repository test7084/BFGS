from  FedCSL_LinR_CI import generate_data,federated_CSL
from FedSGD_LinR_CI import federated_sgd
from FedBFGS_LinR_CI import federated_BFGS
from LocalSGD_LinR_CI import local_sgd
import numpy as np
import pandas as pd
import time
from autograd import grad, hessian
from scipy.stats import norm




N, p = 10000, 5
num_clients = 10
initial_alpha = 0.1
batch_size = 10
max_iter = 400
noise = 1   #生成数据的噪声


true_ws = []  # 用于存储每个数据集对应的 true_w




# 生成随机数据

true_w = np.random.uniform(-0.5, 0.5, size=p)
# print(true_w)
X, y, true_covariance = generate_data(N, p,true_w, noise)

print(true_w)
true_ws.append(true_w)
# print(true_ws)


initial_model = np.random.normal(0, 0.1, size=p)
print(initial_model)




l2_norm_history_localsgd, ail_history_localsgd, srr_history_localsgd, cp_history_localsgd = local_sgd(X, y, true_w, num_clients, initial_model,
                                                                      true_covariance, max_iter)

l2_norm_history_FedSGD, ail_history_FedSGD, srr_history_FedSGD, cp_history_FedSGD = federated_sgd(X, y, true_w, num_clients, initial_model,
                                                                      true_covariance, max_iter)
l2_norm_history_FedCSL, ail_history_FedCSL, srr_history_FedCSL, cp_history_FedCSL = federated_CSL(X, y, true_w, num_clients, initial_model,
                                                                      true_covariance, max_iter)
l2_norm_history_BFGS, ail_history_BFGS, srr_history_BFGS, cp_history_BFGS = federated_BFGS(X, y, true_w, num_clients, initial_model,true_covariance, max_iter)

# 创建一个字典用于存储数据
data = {
    'Iteration': range(1, max_iter + 1),  # 假设每次迭代都有一个对应的值
    'L2 Norm (Local SGD)': l2_norm_history_localsgd,
    'AIL (Local SGD)': ail_history_localsgd,
    'SRR (Local SGD)': srr_history_localsgd,
    'CP (Local SGD)': cp_history_localsgd,

    'L2 Norm (Federated SGD)': l2_norm_history_FedSGD,
    'AIL (Federated SGD)': ail_history_FedSGD,
    'SRR (Federated SGD)': srr_history_FedSGD,
    'CP (Federated SGD)': cp_history_FedSGD,

    'L2 Norm (Federated CSL)': l2_norm_history_FedCSL,
    'AIL (Federated CSL)': ail_history_FedCSL,
    'SRR (Federated CSL)': srr_history_FedCSL,
    'CP (Federated CSL)': cp_history_FedCSL,

    'L2 Norm (Federated BFGS)': l2_norm_history_BFGS,
    'AIL (Federated BFGS)': ail_history_BFGS,
    'SRR (Federated BFGS)': srr_history_BFGS,
    'CP (Federated BFGS)': cp_history_BFGS,
}

# 创建 DataFrame
df = pd.DataFrame(data)

# 保存为 Excel 文件
df.to_excel('results_LinR.xlsx', index=False)