# import numpy as np
import pandas as pd
import time
from autograd import grad, hessian
from scipy.stats import norm
from joblib import Parallel, delayed
import autograd.numpy as np


def generate_logistic_data(N, p,true_w, noise=1):
    # 初始化均值和协方差矩阵
    mean = np.zeros(p)
    covariance = np.full((p, p), 0.2)
    np.fill_diagonal(covariance, 1)
    # 随机生成真实的权重向量 true_w
    # true_w = np.random.uniform(-0.5, 0.5, size=p)
    # 生成特征数据 X
    X = np.random.multivariate_normal(mean, covariance, size=N)
    # 计算线性组合 Z = X * true_w + 噪声
    z = np.dot(X, true_w) + np.random.normal(0, noise, size=N)
    # 使用 sigmoid 函数计算概率
    probability = 1 / (1 + np.exp(-z))
    # 生成二元的响应变量 y，p(y=1) = probability
    y = np.random.binomial(1, probability)

    # 计算真实协方差矩阵
    X_covariance = np.dot(X.T, X)
    X_covariance_inv = np.linalg.inv(X_covariance)
    true_covariance_matrix = noise**2 * X_covariance_inv
    return X, y, true_covariance_matrix



# def loss_function(w, X, y):
#     # 计算预测概率，直接使用 sigmoid 公式
#     z = X @ w
#     predictions = 1 / (1 + np.exp(-z))
#     # 计算逻辑回归的负对数似然损失
#     loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
#     return loss

# # 逻辑回归损失和梯度计算
# def logistic_loss_and_gradient(w, X, y):
#     # 计算预测概率，直接使用 sigmoid 公式
#     z = X @ w
#     predictions = 1 / (1 + np.exp(-z))

#     # 计算逻辑回归的负对数似然损失
#     loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))

#     # 计算梯度
#     gradient = X.T @ (predictions - y) / len(y)

#     return loss, gradient


def safe_log(x):
    epsilon = 1e-8  # 一个非常小的常数，避免 log(0)
    return np.log(np.maximum(x, epsilon))

def sigmoid(z):
    # 防止输入太大导致溢出，确保计算时不出现 nan 或 inf
    # z = np.clip(z, -500, 500)  # 限制 z 的值范围
    return 1 / (1 + np.exp(-z))

def loss_function(w, X, y):
    # 计算预测概率，直接使用 sigmoid 公式
    z = X @ w
    predictions = sigmoid(z)  # 使用改进的 sigmoid 函数
    # 计算逻辑回归的负对数似然损失
    loss = -np.mean(y * safe_log(predictions) + (1 - y) * safe_log(1 - predictions))
    return loss

def logistic_loss_and_gradient(w, X, y):
    # 计算预测概率，使用稳定版的sigmoid函数
    z = X @ w
    predictions = sigmoid(z)  # 使用改进的 sigmoid 函数
    # 计算逻辑回归的负对数似然损失
    loss = -np.mean(y * safe_log(predictions) + (1 - y) * safe_log(1 - predictions))
    # 计算梯度
    gradient = X.T @ (predictions - y) / len(y)
    return loss, gradient

def local_sgd(X, y, true_w, num_clients, initial_model, true_covariance, max_iter=500, tol=1e-6, batch_size=10,
              initial_alpha=0.9, decay_rate=0.99):
    N, p = X.shape
    alpha = initial_alpha
    local_models = [initial_model.copy() for _ in range(num_clients)]
    global_model = np.mean(local_models, axis=0)
    l2_norm_history = []
    ail_history = []  # 用于存储每一轮的置信区间长度
    cp_history = []
    cp01_history = np.zeros(max_iter)
    srr_history = []
    z_975 = norm.ppf(0.975)  # 计算标准正态分布的 0.975 分位数，即 Φ^(-1)(0.975)
    # 置信区间覆盖率的初始化
    coverage_count = 0
    # num_experiments = 100

    machine_data_size = N // num_clients
    data_splits = [(X[i * machine_data_size:(i + 1) * machine_data_size],
                    y[i * machine_data_size:(i + 1) * machine_data_size])
                   for i in range(num_clients)]

    w = np.ones(p) / np.sqrt(p)  # 单位向量

    # 定义工具函数，用于生成 Hessian 矩阵的计算函数
    def hessian_loss_function(X_m, y_m):
        return lambda model: loss_function(model, X_m, y_m)

    # 定义并行计算 Hessian 的任务
    def compute_hessian(m):
        X_m, y_m = data_splits[m]
        loss_fn = hessian_loss_function(X_m, y_m)
        return hessian(loss_fn)(local_models[m])  # 计算 Hessian

    # 定义并行计算损失和梯度的任务
    def compute_loss_and_gradient(m):
        X_m, y_m = data_splits[m]
        # 计算当前批次的起始和结束索引
        start_idx = (iteration * batch_size) % len(X_m)
        end_idx = start_idx + batch_size
        # 按顺序选择当前批次数据
        X_batch, y_batch = X_m[start_idx:end_idx], y_m[start_idx:end_idx]

        local_loss_m, grad = logistic_loss_and_gradient(local_models[m], X_batch, y_batch)
        return local_loss_m, grad  # 返回损失和梯度

    for iteration in range(max_iter):
        gradients = []  # 用于存储每个客户端的梯度
        local_hessians = []  # 用于存储每个客户端的 Hessian 矩阵
        local_losses = []

        results = Parallel(n_jobs=num_clients)(delayed(compute_loss_and_gradient)(m) for m in range(num_clients))
        # 分离损失和梯度
        local_losses, gradients = zip(*results)

        # 并行计算 Hessian 矩阵
        local_hessians = Parallel(n_jobs=num_clients)(delayed(compute_hessian)(m) for m in range(num_clients))

        # 更新本地模型
        for m in range(num_clients):
            local_models[m] -= alpha * gradients[m]

        alpha *= decay_rate

        # 全局聚合
        global_model = np.mean(local_models, axis=0)
        global_hessian = np.mean(local_hessians, axis=0)  # 聚合全局 Hessian 矩阵

        # 计算损失、L2 范数等指标
        current_loss = np.mean(local_losses)
        l2_norm = np.linalg.norm(global_model - true_w)
        l2_norm_history.append(l2_norm)

        # 计算全局协方差矩阵 Sigma
        g_outer_avg = np.zeros((gradients[0].shape[0], gradients[0].shape[0]))
        for m in range(num_clients):
            g_outer_avg += np.outer(gradients[m], gradients[m])
        g_outer_avg /= num_clients

        aveT_H_global_inv_w = np.linalg.solve(global_hessian, w)  # 计算所有训练时刻全局海森矩阵的逆
        est_var = aveT_H_global_inv_w.T @ g_outer_avg @ aveT_H_global_inv_w  # 计算全局协方差矩阵

        # est_var = calculate_global_covariance(gradients, global_hessian, num_clients)

        # **计算置信区间长度**
        # 计算置信区间

        number_samples = num_clients * batch_size
        ci_length = z_975 * np.sqrt((est_var) / number_samples)
        ci_lower = w.T @ global_model - ci_length
        ci_upper = w.T @ global_model + ci_length
        # # 检查覆盖情况
        if (w.T @ true_w >= ci_lower) and (w.T @ true_w <= ci_upper):
            coverage_count += 1
            cp01_history[iteration] = 1
        # AIL
        ail = 2 * ci_length
        ail_history.append(ail)  # 保存每一轮的 CI 长度

        # 计算 SRR
        # srr = np.sqrt(est_var / w.T @ true_covariance @ w)
        # srr_history.append(srr)

        # CP覆盖率
        coverage_rate = coverage_count / (iteration + 1)
        cp_history.append((coverage_rate))

        print(
            f"Local SGD {iteration + 1}: Loss = {current_loss:.4f}, L2 Norm = {l2_norm:.4f}, CP={coverage_rate:.4f}, final_CP={cp01_history[iteration]}, AIL={ail:.4f}")

        # print(f"真实参数: {true_w} 估计参数: {global_model}")
        if l2_norm < tol:
            print(f"模型收敛于第 {iteration + 1} 次全局更新")
            break

    # 计算 SRR
    # srr = calculate_srr(w, est_var, var)  # 用全局海森矩阵作为真实协方差矩阵
    # 计算平均置信区间长度
    average_ail = np.mean(ail_history)
    print(f"Average Interval Length (AIL) = {average_ail:.4f}")
    # print(f"CI Coverage = {coverage_count / num_experiments:.4f}")  # 输出置信区间覆盖率

    return l2_norm_history, ail_history,  cp_history  # 返回覆盖率


if __name__ == "__main__":
    start_time = time.time()

    N, p = 200000, 100
    num_clients = 20
    initial_alpha = 0.5
    batch_size = 10
    max_iter = 300
   #  true_w =np.array([-0.09064027, - 0.30717682,
   # 0.10312984,
   # 0.02839971,
   # 0.40539251, - 0.30477423,
   # 0.20827367, - 0.49457332, - 0.25870951, - 0.48844876])
    true_w = np.random.uniform(-0.5, 0.5, size=p)
    print("true_w:",true_w)
    X, y, true_covariance = generate_logistic_data(N, p, true_w)
    # true_w = [-0.43149495,
    #           0.39185201,
    #           0.47517927,
    #           0.42632228, - 0.16891591,
    #           0.24172592,
    #           0.45885961,
    #           0.45927549,
    #           0.00421632,
    #           0.37175014]
    X, y, true_covariance = generate_logistic_data(N, p, true_w)
    initial_model = np.random.normal(0, 0.1, size=p)
    print("initial_model", initial_model)
    # initial_model = [-0.43149495,
    #                  0.39185201,
    #                  0.47517927,
    #                  0.42632228, - 0.16891591,
    #                  0.24172592,
    #                  0.45885961,
    #                  0.45927549,
    #                  0.00421632,
    #                  0.37175014]
#     initial_model =  np.array([ 0.07440937,  0.06920027,  0.08988633, -0.06392738, -0.0477463,  -0.00403305,
# -0.15490423, -0.02027027, -0.21273082,  0.01995764])

    l2_norm_history, ail_history, cp_history = local_sgd(X, y, true_w, num_clients, initial_model,true_covariance, max_iter)

    # 创建一个DataFrame来存储这些历史数据
    df = pd.DataFrame({
        'L2 Norm History': l2_norm_history,
        'AIL History': ail_history,
        'CP History': cp_history
    })

    # 将DataFrame写入Excel文件
    df.to_excel('LocalSGD_LogR_CI_N200000k20p100.xlsx', index=False)

    # 计算指标
    # Calculate metrics
    last_l2_norm = l2_norm_history[-1]
    min_l2_norm = np.min(l2_norm_history)
    avg_l2_norm = np.mean(l2_norm_history)

    last_ail = ail_history[-1]
    min_ail = np.min(ail_history)
    avg_ail = np.mean(ail_history)



    last_cp = cp_history[-1]
    min_cp = np.min(cp_history)
    avg_cp = np.mean(cp_history)

    print(f"最后迭代的 L2 Norm: {last_l2_norm:.4f}")
    print(f"所有迭代中的最小 L2 Norm: {min_l2_norm:.4f}")
    print(f"所有迭代的平均 L2 Norm: {avg_l2_norm:.4f}")

    print(f"最后迭代的 AIL: {last_ail:.4f}")
    print(f"所有迭代中的最小 AIL: {min_ail:.4f}")
    print(f"所有迭代的平均 AIL: {avg_ail:.4f}")


    print(f"最后迭代的 CP: {last_cp:.4f}")
    print(f"所有迭代中的最小 CP: {min_cp:.4f}")
    print(f"所有迭代的平均 CP: {avg_cp:.4f}")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"程序运行时间: {execution_time:.4f} 秒")
    print("通信开销：", (p * p + p) * num_clients)