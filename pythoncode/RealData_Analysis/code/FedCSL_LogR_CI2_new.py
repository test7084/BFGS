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


# 逻辑回归损失和梯度计算
def logistic_loss_and_gradient(w, X, y):
    # 计算预测概率，直接使用 sigmoid 公式
    z = X @ w
    predictions = 1 / (1 + np.exp(-z))

    # 计算逻辑回归的负对数似然损失
    loss = -np.mean(y * safe_log(predictions+ 1e-8) + (1 - y) * safe_log(1 - predictions+ 1e-8))

    # 计算梯度
    gradient = X.T @ (predictions - y) / len(y)

    return loss, gradient

def surrogate_loss_and_gradient(theta, theta_t, X, y, global_grad_t):
    # 计算 f_k(θ) 和 ∇f_k(θ)
    residuals = X @ theta - y
    loss_f_k = 0.5 * np.mean(residuals ** 2)
    grad_f_k = X.T @ residuals / len(y)

    # 计算 ∇f_k(θ_t)
    _, grad_f_k_t = logistic_loss_and_gradient(theta_t, X, y)

    # 定义新的损失函数 f_k(θ) - <∇f_k(θ_t) − ∇f(θ_t), θ>
    inner_product = (grad_f_k_t - global_grad_t) @ theta
    surrogate_loss = loss_f_k - inner_product

    # 计算新的梯度 ∇f_k(θ) - (∇f_k(θ_t) - ∇f(θ_t))
    surrogate_gradient = grad_f_k - (grad_f_k_t - global_grad_t)

    return surrogate_loss, surrogate_gradient


def calculate_srr(w, Sigma, Sigma_true):
    numerator = w.T @ Sigma @ w
    denominator = w.T @ Sigma_true @ w
    return np.sqrt(numerator / denominator)


# 代理损失和梯度计算
def surrogate_loss_and_gradient(theta, localgrad, X, y, global_grad_t):
    residuals = X @ theta - y
    loss_f_k = 0.5 * np.mean(residuals ** 2)
    grad_f_k = X.T @ residuals / len(y)

    # 定义新的损失函数 f_k(θ) - <∇f_k(θ_t) − ∇f(θ_t), θ>
    inner_product = (localgrad - global_grad_t) @ theta
    surrogate_loss = loss_f_k - inner_product

    # 计算新的梯度 ∇f_k(θ) - (∇f_k(θ_t) - ∇f(θ_t))
    surrogate_gradient = grad_f_k - (localgrad - global_grad_t)

    return surrogate_loss, surrogate_gradient


# 客户端训练函数
def client_training(X_m, y_m, local_model, global_model, global_grad_t, alpha, num_local_steps):
    for _ in range(num_local_steps):
        indices = np.random.choice(len(X_m), 10, replace=False)  # 使用固定的batch_size
        X_batch, y_batch = X_m[indices], y_m[indices]
        surrogate_loss, surrogate_grad = surrogate_loss_and_gradient(local_model, global_model, X_batch, y_batch,
                                                                     global_grad_t)
        local_model -= alpha * surrogate_grad
    return local_model


# 服务器聚合模型
def server_aggregate(local_models):
    return np.mean(local_models, axis=0)


# 联邦训练函数
def federated_CSL(X, y, true_w, num_clients, intial_model, true_covariance, max_iter=500, tol=1e-6, batch_size=10,
                  initial_alpha=0.2,
                  decay_rate=0.99, num_local_steps=10):
    N, p = X.shape
    alpha = initial_alpha

    # 初始化每个机器的本地模型
    local_models = [intial_model for _ in range(num_clients)]
    global_model = np.mean(local_models, axis=0)
    l2_norm_history = []
    ail_history = []  # 用于存储每一轮的置信区间长度
    cp_history = []
    cp01_history = np.zeros(max_iter)
    loss_history = []
    srr_history = []
    z_975 = norm.ppf(0.975)  # 计算标准正态分布的 0.975 分位数，即 Φ^(-1)(0.975)
    # 置信区间覆盖率的初始化
    coverage_count = 0

    # 将数据划分给每台机器
    client_data_size = N // num_clients
    data_splits = [(X[i * client_data_size:(i + 1) * client_data_size],
                    y[i * client_data_size:(i + 1) * client_data_size])
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

    def compute_gradients_and_hessians(m):
        X_m, y_m = data_splits[m]

        # 计算当前批次的起始和结束索引
        start_idx = iteration * batch_size
        end_idx = start_idx + batch_size
        # 按顺序选择当前批次数据
        X_batch, y_batch = X_m[start_idx:end_idx], y_m[start_idx:end_idx]
        # 计算梯度
        _, grad = logistic_loss_and_gradient(local_models[m], X_batch, y_batch)
        # 在本地计算 Hessian 矩阵
        local_hessian = hessian(hessian_loss_function(X_m, y_m))(local_models[m])

        return grad, local_hessian

    for iteration in range(max_iter):
        gradients = []  # 用于存储每个客户端的梯度
        local_hessians = []  # 用于存储每个客户端的 Hessian 矩阵
        # 每台机器计算局部梯度并发送到中心处理器
        local_grads = []

        results = Parallel(n_jobs=num_clients)(delayed(compute_loss_and_gradient)(m) for m in range(num_clients))
        # 分离损失和梯度
        local_losses, local_grads = zip(*results)

        # 中央处理器计算平均梯度并广播
        global_grad = np.mean(local_grads, axis=0)

        # # 并行计算每个客户端的梯度和 Hessian
        # results = Parallel(n_jobs=num_clients)(delayed(compute_gradients_and_hessians)(m) for m in range(num_clients))

        # # 分离梯度和 Hessian
        # local_grads, local_hessians = zip(*results)

        # 每台机器根据目标函数最小化并发送更新到中央处理器
        def update_local_model(m):
            # 复制当前全局模型作为初始局部模型
            local_model = global_model.copy()
            alpha0 = alpha
            for _ in range(num_local_steps):
                # 使用 local_grad 和 global_grad 进行更新
                surrogate_loss, custom_grad = surrogate_loss_and_gradient(local_model, local_grads[m],
                                                                          data_splits[m][0], data_splits[m][1],
                                                                          global_grad)

                local_model -= alpha0 * custom_grad
                alpha0 *= 0.95

            return local_model, custom_grad

        # 并行更新每个机器的本地模型
        updated_results = Parallel(n_jobs=num_clients)(delayed(update_local_model)(m) for m in range(num_clients))

        alpha = 0.1 / (iteration + 1)
        # 更新本地模型和梯度
        for m, (local_model, custom_grad) in enumerate(updated_results):
            local_models[m] = local_model
            gradients.append(custom_grad)  # 存储每个客户端的梯度

        local_hessians = Parallel(n_jobs=num_clients)(delayed(compute_hessian)(m) for m in range(num_clients))
        alpha *= decay_rate

        global_model = np.mean(local_models, axis=0)
        global_hessian = np.mean(local_hessians, axis=0)  # 聚合全局 Hessian 矩阵

        # 计算损失、L2 范数等指标
        current_loss = np.mean(local_losses, axis=0)
        l2_norm = np.linalg.norm(global_model - true_w)
        l2_norm_history.append(l2_norm)
        loss_history.append(current_loss)

        g_outer_sum = np.zeros((gradients[0].shape[0], gradients[0].shape[0]))  # 外积和初始化
        # 计算全局协方差矩阵 Sigma
        for gi in gradients:
            g_outer_sum += np.outer(gi, gi)
        g_outer_avg = g_outer_sum / num_clients

        aveT_H_global_inv = np.linalg.inv(global_hessian)  # 计算所有训练时刻全局海森矩阵的逆
        # aveT_H_global_inv_w = np.linalg.solve(global_hessian, w)
        est_var = aveT_H_global_inv @ (g_outer_avg) @ aveT_H_global_inv  # 计算全局协方差矩阵
        # est_var = aveT_H_global_inv_w.T @ g_outer_avg @ aveT_H_global_inv_w

        # **计算置信区间长度**
        # 计算置信区间
        number_samples = num_clients * batch_size
        ci_length = z_975 * np.sqrt((w.T @ est_var @ w) / number_samples)
        ci_lower = w.T @ global_model - ci_length
        ci_upper = w.T @ global_model + ci_length
        # # 检查覆盖情况
        if (w.T @ true_w >= ci_lower) and (w.T @ true_w <= ci_upper):
            coverage_count += 1
            cp01_history[iteration] = 1
        # AIL
        ail = ci_length
        ail_history.append(ail)  # 保存每一轮的 CI 长度

        # # 计算 SRR
        # srr = calculate_srr(w, est_var, true_covariance)  # 用全局海森矩阵作为真实协方差矩阵
        # srr_history.append(srr)

        # CP覆盖率
        coverage_rate = coverage_count / (iteration + 1)
        cp_history.append((coverage_rate))

        print(
            f"FedCSL {iteration + 1}: Loss = {current_loss:.4f}, L2 Norm = {l2_norm:.4f}, CP={coverage_rate:.4f}, final_CP={cp01_history[iteration]}, AIL={ail:.4f}")

        if l2_norm < tol:
            print(f"模型收敛于第 {iteration + 1} 次全局更新")
            break

        # 计算 SRR
        # srr = calculate_srr(w, est_var, var)  # 用全局海森矩阵作为真实协方差矩阵
        # 计算平均置信区间长度
    average_ail = np.mean(ail_history)
    print(f"Average Interval Length (AIL) = {average_ail:.4f}")
    # print(f"CI Coverage = {coverage_count / num_experiments:.4f}")  # 输出置信区间覆盖率

    return loss_history, l2_norm_history, ail_history, cp_history  # 返回覆盖率


# 主函数
if __name__ == "__main__":
    start_time = time.time()

    N, p = 200000, 100
    num_clients = 20
    initial_alpha = 0.00001
    batch_size = 10
    max_iter = 300
    true_w = np.random.uniform(-0.5, 0.5, size=p)
    X, y, true_covariance = generate_logistic_data(N, p, true_w)
    initial_model = np.random.normal(0, 0.1, size=p)
    # true_w = np.array([-0.09064027, - 0.30717682,
    #                    0.10312984,
    #                    0.02839971,
    #                    0.40539251, - 0.30477423,
    #                    0.20827367, - 0.49457332, - 0.25870951, - 0.48844876])
    # true_w = np.array([-0.0565327, 0.25099069, -0.37631736, -0.40326699, 0.37086017, 0.28294391,
    #           0.25751959, 0.00661737, -0.02463554, 0.10878637, -0.10427602, 0.27973042,
    #           -0.10310403, 0.20214087, -0.47362878, 0.3482037, -0.18057214, 0.38954681,
    #           0.16493052, 0.33260615, -0.02315252, 0.1871603, -0.1877714, -0.02474511,
    #           0.00828701, 0.1191221, -0.16720248, -0.19095845, -0.05693608, 0.17292977,
    #           -0.2187786, 0.05576835, -0.41241086, -0.24647714, -0.30771062, 0.08476041,
    #           0.178626, -0.00465841, 0.23009452, -0.27514723, -0.15821784, -0.0613083,
    #           0.17213351, 0.17280699, -0.36259588, -0.4945381, 0.29758038, 0.26513314,
    #           -0.23988742, 0.03797761, 0.30286003, 0.2476939, -0.17031656, -0.0688906,
    #           -0.30484922, 0.2954227, 0.49519552, 0.19307402, 0.14021994, 0.13616312,
    #           -0.0902053, -0.1544647, -0.22394054, -0.22230975, 0.20717917, -0.03187367,
    #           -0.47420139, 0.1968122, 0.38816358, -0.20264843, 0.17297111, -0.3485448,
    #           -0.22958891, -0.07204058, 0.4909282, -0.30674114, -0.41451482, -0.27800917,
    #           -0.20312731, -0.46017351, 0.00780113, -0.2153871, -0.37317276, -0.04986955,
    #           -0.14736823, 0.2148469, -0.31051706, -0.04299646, -0.43555296, -0.0811036,
    #           0.33460475, -0.34463609, 0.14315483, 0.17100154, 0.44674999, 0.13924505,
    #           0.06239016, -0.33690692, 0.33658157, 0.07228054])
    # X, y, true_covariance = generate_logistic_data(N, p, true_w)
    # initial_model = np.array([-0.0565327, 0.25099069, -0.37631736, -0.40326699, 0.37086017, 0.28294391,
    #                  0.25751959, 0.00661737, -0.02463554, 0.10878637, -0.10427602, 0.27973042,
    #                  -0.10310403, 0.20214087, -0.47362878, 0.3482037, -0.18057214, 0.38954681,
    #                  0.16493052, 0.33260615, -0.02315252, 0.1871603, -0.1877714, -0.02474511,
    #                  0.00828701, 0.1191221, -0.16720248, -0.19095845, -0.05693608, 0.17292977,
    #                  -0.2187786, 0.05576835, -0.41241086, -0.24647714, -0.30771062, 0.08476041,
    #                  0.178626, -0.00465841, 0.23009452, -0.27514723, -0.15821784, -0.0613083,
    #                  0.17213351, 0.17280699, -0.36259588, -0.4945381, 0.29758038, 0.26513314,
    #                  -0.23988742, 0.03797761, 0.30286003, 0.2476939, -0.17031656, -0.0688906,
    #                  -0.30484922, 0.2954227, 0.49519552, 0.19307402, 0.14021994, 0.13616312,
    #                  -0.0902053, -0.1544647, -0.22394054, -0.22230975, 0.20717917, -0.03187367,
    #                  -0.47420139, 0.1968122, 0.38816358, -0.20264843, 0.17297111, -0.3485448,
    #                  -0.22958891, -0.07204058, 0.4909282, -0.30674114, -0.41451482, -0.27800917,
    #                  -0.20312731, -0.46017351, 0.00780113, -0.2153871, -0.37317276, -0.04986955,
    #                  -0.14736823, 0.2148469, -0.31051706, -0.04299646, -0.43555296, -0.0811036,
    #                  0.33460475, -0.34463609, 0.14315483, 0.17100154, 0.44674999, 0.13924505,
    #                  0.06239016, -0.33690692, 0.33658157, 0.07228054])
    loss_history, l2_norm_history, ail_history, cp_history = federated_CSL(X, y, true_w, num_clients, initial_model,
                                                                           true_covariance, max_iter)

    # 创建一个DataFrame来存储这些历史数据
    df = pd.DataFrame({
        'L2 Norm History': l2_norm_history,
        'AIL History': ail_history,
        'CP History': cp_history
    })

    # 将DataFrame写入Excel文件
    df.to_excel('FedCSL_LogR_CI_N200000k20p100.xlsx', index=False)

    # 计算指标
    # Calculate metrics
    last_loss = loss_history[-1]
    min_loss = np.min(loss_history)
    avg_loss = np.mean(loss_history)

    last_l2_norm = l2_norm_history[-1]
    min_l2_norm = np.min(l2_norm_history)
    avg_l2_norm = np.mean(l2_norm_history)

    last_ail = ail_history[-1]
    min_ail = np.min(ail_history)
    avg_ail = np.mean(ail_history)

    last_cp = cp_history[-1]
    min_cp = np.min(cp_history)
    avg_cp = np.mean(cp_history)

    print(f"最后迭代的 loss: {last_loss:.4f}")
    print(f"所有迭代中的最小 loss: {min_loss:.4f}")
    print(f"所有迭代的平均 loss: {avg_loss:.4f}")

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