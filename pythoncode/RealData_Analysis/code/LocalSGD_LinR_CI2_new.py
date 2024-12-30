import numpy as np
import pandas as pd
import time
from autograd import grad, hessian
from scipy.stats import norm
from joblib import Parallel, delayed


def generate_data(N, p, true_w, noise=1):
    mean = np.zeros(p)
    covariance = np.full((p, p), 0.2)
    np.fill_diagonal(covariance, 1)
    X = np.random.multivariate_normal(mean, covariance, size=N)
    ei = np.random.normal(0, noise, size=N)
    y = np.dot(X, true_w) + ei

    # 计算真实协方差矩阵
    # sigma_squared = noise**2  # 噪声方差
    # true_covariance_matrix = sigma_squared * np.linalg.inv(covariance)
    X_covariance = np.dot(X.T, X)
    X_covariance_inv = np.linalg.inv(X_covariance)
    true_covariance_matrix = noise ** 2 * X_covariance_inv
    return X, y, true_covariance_matrix


def loss_function(w, X, y):
    residuals = X @ w - y
    return 0.5 * sum(residuals ** 2) / len(y)  # 使用内置 sum


def loss_and_gradient(w, X, y):
    residuals = X @ w - y
    loss = 0.5 * sum(residuals ** 2) / len(y)  # 使用内置 sum
    gradient = X.T @ residuals / len(y)
    return loss, gradient


# def calculate_srr(w, Sigma, Sigma_true):
#     numerator = w.T @ Sigma @ w
#     denominator = w.T @ Sigma_true @ w
#     return np.sqrt(numerator / denominator)


# def calculate_global_covariance(g, H_global, M):
#     g_outer_sum = np.zeros((g[0].shape[0], g[0].shape[0]))  # 外积和初始化

#     for gi in g:
#         g_outer_sum += np.outer(gi, gi)

#     g_outer_avg = g_outer_sum / M  # 计算外积的平均
#     H_global_inv = np.linalg.inv(H_global)  # 计算全局海森矩阵的逆
#     est_var = H_global_inv @ g_outer_avg @ H_global_inv  # 计算全局协方差矩阵

#     return est_var

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
        
        local_loss_m, grad = loss_and_gradient(local_models[m], X_batch, y_batch)
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
        ail = 2*ci_length
        ail_history.append(ail)  # 保存每一轮的 CI 长度

        # 计算 SRR
        srr = np.sqrt(est_var / w.T @ true_covariance @ w)
        srr_history.append(srr)

        # CP覆盖率
        coverage_rate = coverage_count / (iteration + 1)
        cp_history.append((coverage_rate))

        print(
            f"Local SGD {iteration + 1}: Loss = {current_loss:.4f}, L2 Norm = {l2_norm:.4f}, CP={coverage_rate:.4f}, final_CP={cp01_history[iteration]}, AIL={ail:.4f}, SRR = {srr:.4f}")

        print(f"真实参数: {true_w} 估计参数: {global_model}")
        if l2_norm < tol:
            print(f"模型收敛于第 {iteration + 1} 次全局更新")
            break

    # 计算 SRR
    # srr = calculate_srr(w, est_var, var)  # 用全局海森矩阵作为真实协方差矩阵
    # 计算平均置信区间长度
    average_ail = np.mean(ail_history)
    print(f"Average Interval Length (AIL) = {average_ail:.4f}")
    # print(f"CI Coverage = {coverage_count / num_experiments:.4f}")  # 输出置信区间覆盖率

    return l2_norm_history, ail_history, srr_history, cp_history  # 返回覆盖率


if __name__ == "__main__":
    start_time = time.time()

    N, p = 100000, 10
    num_clients = 100
    initial_alpha = 1
    batch_size = 10
    max_iter = 400
    true_w = np.random.uniform(-0.5, 0.5, size=p)
    X, y, true_covariance = generate_data(N, p, true_w)
    initial_model = np.random.normal(0, 0.1, size=p)

    l2_norm_history, ail_history, srr_history, cp_history = local_sgd(X, y, true_w, num_clients, initial_model,
                                                                      true_covariance, max_iter)

    # 计算指标
    # Calculate metrics
    last_l2_norm = l2_norm_history[-1]
    min_l2_norm = np.min(l2_norm_history)
    avg_l2_norm = np.mean(l2_norm_history)

    last_ail = ail_history[-1]
    min_ail = np.min(ail_history)
    avg_ail = np.mean(ail_history)

    last_srr = srr_history[-1]
    min_srr = np.min(srr_history)
    avg_srr = np.mean(srr_history)

    last_cp = cp_history[-1]
    min_cp = np.min(cp_history)
    avg_cp = np.mean(cp_history)

    print(f"最后迭代的 L2 Norm: {last_l2_norm:.4f}")
    print(f"所有迭代中的最小 L2 Norm: {min_l2_norm:.4f}")
    print(f"所有迭代的平均 L2 Norm: {avg_l2_norm:.4f}")

    print(f"最后迭代的 AIL: {last_ail:.4f}")
    print(f"所有迭代中的最小 AIL: {min_ail:.4f}")
    print(f"所有迭代的平均 AIL: {avg_ail:.4f}")

    print(f"最后迭代的 SRR: {last_srr:.4f}")
    print(f"所有迭代中的最小 SRR: {min_srr:.4f}")
    print(f"所有迭代的平均 SRR: {avg_srr:.4f}")

    print(f"最后迭代的 CP: {last_cp:.4f}")
    print(f"所有迭代中的最小 CP: {min_cp:.4f}")
    print(f"所有迭代的平均 CP: {avg_cp:.4f}")


    end_time = time.time()
    execution_time = end_time - start_time
    print(f"程序运行时间: {execution_time:.4f} 秒")
    print("通信开销：", (p * p + p) * num_clients)