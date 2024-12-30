import autograd.numpy as np
from autograd import grad
import pandas as pd
import time
from autograd import grad, hessian
from scipy.stats import norm


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


def loss_function(w, X, y):
    # 计算预测概率，直接使用 sigmoid 公式
    z = X @ w
    predictions = 1 / (1 + np.exp(-z))
    # 计算逻辑回归的负对数似然损失
    loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return loss

# 逻辑回归损失和梯度计算
def logistic_loss_and_gradient(w, X, y):
    # 计算预测概率，直接使用 sigmoid 公式
    z = X @ w
    predictions = 1 / (1 + np.exp(-z))

    # 计算逻辑回归的负对数似然损失
    loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))

    # 计算梯度
    gradient = X.T @ (predictions - y) / len(y)

    return loss, gradient

# Wolfe条件的线性搜索函数
def line_search_wolfe(w, direction, X, y, grad, c1=0.01, c2=0.5, max_iter=10, alpha=1.0):
    for _ in range(max_iter):
        w_new = w + alpha * direction
        loss_new, grad_new = logistic_loss_and_gradient(w_new, X, y)
        loss, _ = logistic_loss_and_gradient(w, X, y)

        # 检查Wolfe条件
        if loss_new > loss + c1 * alpha * np.dot(grad, direction):
            alpha *= 0.5
        elif np.dot(grad_new, direction) < c2 * np.dot(grad, direction):
            alpha *= 0.5
        else:
            break
    return alpha

# # 单批次更新
# def batch_update(w, H, X, y, tol=1e-6):
#     loss, grad = loss_and_gradient(w, X, y)
#     if np.linalg.norm(grad) < tol:
#         return w, grad, H

#     # 使用Hessian矩阵计算更新方向
#     direction = -np.linalg.solve(H, grad)  # 用 np.linalg.solve 解方程 H @ direction = -grad

#     # 线性搜索找最佳步长 alpha
#     alpha = line_search_wolfe(w, direction, X, y, grad)
#     w_new = w + alpha * direction

#     # 更新 BFGS 近似
#     s = (w_new - w).reshape(-1, 1)
#     _, grad_new = loss_and_gradient(w_new, X, y)
#     y_vec = (grad_new - grad).reshape(-1, 1)
#     I = np.eye(w.shape[0])

#     if s.T @ y_vec > 0:
#         H = H + (y_vec @ y_vec.T) / (y_vec.T @ s) - (H @ s @ s.T @ H) / (s.T @ H @ s)

#     return w_new, grad, H


#  BFGS 逆公式更新单批次更新
def batch_update(w, H, X, y, tol=1e-6):
    loss, grad = logistic_loss_and_gradient(w, X, y)
    if np.linalg.norm(grad) < tol:
        return w, grad, H

    # 使用Hessian矩阵计算更新方向
    # direction = -np.linalg.solve(H, grad)  # 用 np.linalg.solve 解方程 H @ direction = -grad

    direction = -np.dot(H, grad)
    # 线性搜索找最佳步长 alpha
    alpha = line_search_wolfe(w, direction, X, y, grad)
    w_new = w + alpha * direction

    # 更新 BFGS 近似
    s = (w_new - w).reshape(-1, 1)
    loss, grad_new = logistic_loss_and_gradient(w_new, X, y)
    y_vec = (grad_new - grad).reshape(-1, 1)

    # if s.T @ y_vec > 0:
    #     H = H + (y_vec @ y_vec.T) / (y_vec.T @ s) - (H @ s @ s.T @ H) / (s.T @ H @ s)

    # BFGS 逆公式更新
    if s.T @ y_vec > 1e-8:  # 检查正定条件
        rho = 1.0 / (y_vec.T @ s)
        V = np.eye(len(w)) - rho * (s @ y_vec.T)
        H = V.T @ H @ V + rho * (s @ s.T)

    return w_new, grad, H, loss


# 客户端训练函数
# def client_training(X_m, y_m, local_model, alpha, batch_size,num_local_steps):
#     H = np.eye(len(local_model))  # 初始化Hessian矩阵
#     for _ in range(num_local_steps):
#         # 随机选择一个batch
#         indices = np.random.choice(len(X_m), 10, replace=False)
#         X_batch = X_m[indices]
#         y_batch = y_m[indices]
#         local_model, grad, H = batch_update(local_model, H, X_batch, y_batch, alpha)
#     return local_model

def client_training(X_m, y_m, H, local_model, batch_size):
    # 随机选择一个batch
    indices = np.random.choice(len(X_m), batch_size, replace=False)
    X_batch = X_m[indices]
    y_batch = y_m[indices]
    local_model, grad, H, loss = batch_update(local_model, H, X_batch, y_batch)
    return local_model, grad, H, loss


# 服务器聚合
def server_aggregate(local_models):
    return np.mean(local_models, axis=0)


def calculate_srr(w, Sigma, Sigma_true):
    numerator = w.T @ Sigma @ w
    denominator = w.T @ Sigma_true @ w
    return np.sqrt(numerator / denominator)


# 联邦训练函数
def federated_BFGS(X, y, true_w, num_clients, intial_model, true_covariance, max_iter=500, tol=1e-6, batch_size=10,
                   initial_alpha=0.2, decay_rate=0.95):
    N, p = X.shape
    alpha = initial_alpha
    # local_models = [intial_model.copy() for _ in range(num_clients)]
    # global_model = np.mean(local_models, axis=0)
    l2_norm_history = []
    ail_history = []  # 用于存储每一轮的置信区间长度
    cp_history = []
    cp01_history = np.zeros(max_iter)
    srr_history = []
    z_975 = norm.ppf(0.975)  # 计算标准正态分布的 0.975 分位数，即 Φ^(-1)(0.975)
    # 置信区间覆盖率的初始化
    coverage_count = 0

    # 将数据划分给每个客户端
    client_data_size = N // num_clients
    data_splits = [(X[i * client_data_size:(i + 1) * client_data_size],
                    y[i * client_data_size:(i + 1) * client_data_size])
                   for i in range(num_clients)]

    w = np.ones(p) / np.sqrt(p)  # 单位向量

    # 初始化全局模型
    # global_model = np.random.normal(0, 0.1, size=p)
    global_model = intial_model
    global_hessian = np.eye(len(global_model))  # 初始化Hessian矩阵
    for iteration in range(max_iter):
        local_models = []
        gradients = []  # 用于存储每个客户端的梯度
        local_hessians = []  # 用于存储每个客户端的 Hessian 矩阵
        local_losses = []

        # 每个客户端进行本地训练
        for m in range(num_clients):
            X_m, y_m = data_splits[m]
            local_model, grad, local_hessian, local_loss = client_training(X_m, y_m, global_hessian,
                                                                           global_model.copy(), batch_size)
            local_models.append(local_model)
            gradients.append(grad)  # 存储每个客户端的梯度
            local_hessians.append(local_hessian)
            local_losses.append(local_loss)
        # 更新学习率
        alpha *= decay_rate
        # 服务器聚合模型
        global_model = server_aggregate(local_models)
        # 全局hessian
        global_hessian = np.mean(local_hessians, axis=0)  # 聚合全局 Hessian 矩阵

        # 计算损失、L2 范数等指标
        # current_loss, _ = loss_and_gradient(global_model, X, y)
        current_loss = np.mean(local_losses, axis=0)
        l2_norm = np.linalg.norm(global_model - true_w)
        l2_norm_history.append(l2_norm)

        g_outer_sum = np.zeros((gradients[0].shape[0], gradients[0].shape[0]))  # 外积和初始化
        # 计算全局协方差矩阵 Sigma
        for gi in gradients:
            g_outer_sum += np.outer(gi, gi)
        g_outer_avg = g_outer_sum / num_clients

        # aveT_H_global_inv = np.linalg.inv(global_hessian)  # 计算所有训练时刻全局海森矩阵的逆
        # est_var = aveT_H_global_inv @ g_outer_avg @ aveT_H_global_inv  # 计算全局协方差矩阵
        est_var = global_hessian @ g_outer_avg @ global_hessian  # 计算全局协方差矩阵

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
            f"FedBFGS{iteration + 1}: Loss = {current_loss:.4f}, L2 Norm = {l2_norm:.4f}, CP={coverage_rate:.4f}, final_CP={cp01_history[iteration]}, AIL={ail:.4f}")

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


# 主函数
if __name__ == "__main__":
    start_time = time.time()

    N, p = 200000, 100
    num_clients = 20
    initial_alpha = 0.1
    batch_size = 20
    max_iter = 300
    true_w = np.random.uniform(-0.5, 0.5, size=p)
    # print("true_w",true_w)
    X, y, true_covariance = generate_logistic_data(N, p, true_w)
    initial_model = np.random.normal(0, 0.1, size=p)
    # print("intial_model", initial_model)
    # true_w = [-0.0565327, 0.25099069, -0.37631736, -0.40326699, 0.37086017, 0.28294391,
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
    #           0.06239016, -0.33690692, 0.33658157, 0.07228054]
    X, y, true_covariance = generate_logistic_data(N, p, true_w)
    # initial_model = [-0.0565327, 0.25099069, -0.37631736, -0.40326699, 0.37086017, 0.28294391,
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
    #                  0.06239016, -0.33690692, 0.33658157, 0.07228054]

    l2_norm_history, ail_history, cp_history = federated_BFGS(X, y, true_w, num_clients, initial_model,
                                                                           true_covariance, max_iter)

    # 创建一个DataFrame来存储这些历史数据
    df = pd.DataFrame({
        'L2 Norm History': l2_norm_history,
        'AIL History': ail_history,
        'CP History': cp_history
    })

    # 将DataFrame写入Excel文件
    df.to_excel('FedBFGS_LogR_CI_N200000k20p100.xlsx', index=False)

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