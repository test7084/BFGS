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

# 本地SGD训练函数
def local_sgd(X, y, local_model, batch_size, alpha):
    # indices = np.random.choice(len(X), batch_size, replace=False)
    # X_batch, y_batch = X[indices], y[indices]
    loss, grad = logistic_loss_and_gradient(local_model, X, y)
    local_model -= alpha * grad  # 更新本地模型参数
    return local_model,grad,loss


# 服务端聚合
def server_aggregate(local_models):
    return np.mean(local_models, axis=0)



def calculate_srr(w, Sigma, Sigma_true):
    numerator = w.T @ Sigma @ w
    denominator = w.T @ Sigma_true @ w
    return np.sqrt(numerator / denominator)


def federated_sgd(X, y, true_w, num_clients, initial_model, true_covariance, max_iter=500, tol=1e-6, batch_size=10,
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


    for iteration in range(max_iter):
        gradients = []  # 用于存储每个客户端的梯度
        local_hessians = []  # 用于存储每个客户端的 Hessian 矩阵
        local_losses=[]

        # for m in range(num_clients):
        #     X_m, y_m = data_splits[m]
        #     local_models[m],grad = local_sgd(X_m, y_m, local_models[m], batch_size, alpha)
        #     # 在本地计算 Hessian 矩阵
        #     local_hessian = hessian(hessian_loss_function(X_m, y_m))(local_models[m])
        #     local_hessians.append(local_hessian)
        #     gradients.append(grad)

        # 并行计算每个客户端的模型更新和 Hessian
        def update_client_model(m):
            X_m, y_m = data_splits[m]
            # 计算当前批次的起始和结束索引
            start_idx = (iteration * batch_size) % len(X_m)
            end_idx = start_idx + batch_size
            # 按顺序选择当前批次数据
            X_batch, y_batch = X_m[start_idx:end_idx], y_m[start_idx:end_idx]
            # 执行局部SGD
            updated_model, grad,localloss = local_sgd(X_batch,  y_batch, local_models[m], batch_size, alpha)
            # 计算 Hessian 矩阵
            local_hessian = hessian(hessian_loss_function(X_m, y_m))(updated_model)

            return updated_model, grad, local_hessian,localloss

        # 使用并行处理来更新每个客户端的模型
        results = Parallel(n_jobs=num_clients)(delayed(update_client_model)(m) for m in range(num_clients))

        # 分离更新后的模型、梯度和 Hessian
        # local_losses, local_grads = zip(*results)
        for m, (updated_model, grad, local_hessian,localloss) in enumerate(results):
            local_models[m] = updated_model
            gradients.append(grad)  # 存储每个客户端的梯度
            local_hessians.append(local_hessian)  # 存储每个客户端的 Hessian 矩阵
            local_losses.append(localloss)

        alpha *= decay_rate

        # 全局聚合
        global_model = np.mean(local_models, axis=0)
        global_hessian = np.mean(local_hessians, axis=0)  # 聚合全局 Hessian 矩阵

        # # 下发全局模型给所有客户端
        # for m in range(num_clients):
        #     local_models[m] = global_model.copy()  # 更新本地模型为全局模型
        # 下发全局模型给所有客户端
        def update_client_model(m):
            return global_model.copy()  # 复制全局模型
        # 使用并行处理来更新每个客户端的本地模型
        local_models = Parallel(n_jobs=num_clients)(delayed(update_client_model)(m) for m in range(num_clients))

        # 计算损失、L2 范数等指标
        current_loss = np.mean(local_losses, axis=0)
        l2_norm = np.linalg.norm(global_model - true_w)
        l2_norm_history.append(l2_norm)

        g_outer_sum = np.zeros((gradients[0].shape[0], gradients[0].shape[0]))  # 外积和初始化
        # 计算全局协方差矩阵 Sigma
        for gi in gradients:
            g_outer_sum += np.outer(gi, gi)
        g_outer_avg = g_outer_sum / num_clients


        aveT_H_global_inv = np.linalg.inv(global_hessian)  # 计算所有训练时刻全局海森矩阵的逆
        # aveT_H_global_inv_w = np.linalg.solve(global_hessian, w)
        # est_var = aveT_H_global_inv_w.T @ g_outer_avg @ aveT_H_global_inv_w # 计算全局协方差矩阵
        est_var = aveT_H_global_inv @ (g_outer_avg) @ aveT_H_global_inv  # 计算全局协方差矩阵

        # est_var = calculate_global_covariance(gradients, global_hessian, num_clients)

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
            f"FedSGD {iteration + 1}: Loss = {current_loss:.4f}, L2 Norm = {l2_norm:.4f}, CP={coverage_rate:.4f}, final_CP={cp01_history[iteration]}, AIL={ail:.4f}")

        if l2_norm < tol:
            print(f"模型收敛于第 {iteration + 1} 次全局更新")
            break


    # 计算平均置信区间长度
    average_ail = np.mean(ail_history)
    print(f"Average Interval Length (AIL) = {average_ail:.4f}")
    # print(f"CI Coverage = {coverage_count / num_experiments:.4f}")  # 输出置信区间覆盖率

    return l2_norm_history, ail_history,  cp_history  # 返回覆盖率


if __name__ == "__main__":
    start_time = time.time()

    N, p = 200000, 100
    num_clients = 20
    initial_alpha = 1
    batch_size = 10
    max_iter = 300
 #    true_w=[-0.0565327,   0.25099069, -0.37631736, -0.40326699,  0.37086017,  0.28294391,
 #  0.25751959,  0.00661737, -0.02463554,  0.10878637, -0.10427602,  0.27973042,
 # -0.10310403 , 0.20214087, -0.47362878,  0.3482037,  -0.18057214,  0.38954681,
 #  0.16493052,  0.33260615, -0.02315252,  0.1871603,  -0.1877714,  -0.02474511,
 #  0.00828701,  0.1191221,  -0.16720248, -0.19095845, -0.05693608,  0.17292977,
 # -0.2187786 ,  0.05576835, -0.41241086, -0.24647714, -0.30771062,  0.08476041,
 #  0.178626,   -0.00465841,  0.23009452, -0.27514723, -0.15821784, -0.0613083,
 #  0.17213351,  0.17280699, -0.36259588, -0.4945381,   0.29758038,  0.26513314,
 # -0.23988742,  0.03797761,  0.30286003,  0.2476939,  -0.17031656, -0.0688906,
 # -0.30484922,  0.2954227 ,  0.49519552,  0.19307402,  0.14021994,  0.13616312,
 # -0.0902053,  -0.1544647,  -0.22394054, -0.22230975,  0.20717917, -0.03187367,
 # -0.47420139,  0.1968122,   0.38816358, -0.20264843,  0.17297111, -0.3485448,
 # -0.22958891, -0.07204058,  0.4909282,  -0.30674114, -0.41451482, -0.27800917,
 # -0.20312731, -0.46017351,  0.00780113, -0.2153871,  -0.37317276, -0.04986955,
 # -0.14736823,  0.2148469,  -0.31051706, -0.04299646, -0.43555296, -0.0811036,
 #  0.33460475, -0.34463609,  0.14315483,  0.17100154,  0.44674999,  0.13924505,
 #  0.06239016, -0.33690692,  0.33658157,  0.07228054]
    true_w = np.random.uniform(-0.5, 0.5, size=p)
    # true_w = [0.43038294, - 0.42494823, - 0.04939066, - 0.37626503, - 0.22343397, - 0.05183448,
    #           - 0.26021582, 0.02671197, 0.32387738, 0.19558738]
    # true_w=[-0.19959704,0.03292176,0.46165019, - 0.27262339, - 0.11848485, - 0.06060108,
    # - 0.46191334,
    # 0.42206347, - 0.29233146, - 0.05537873, - 0.24955772,
    # 0.14157733,
    # 0.44567857, - 0.26011995,
    # 0.32175989,
    # 0.05082601, - 0.44654817, - 0.39066483,
    # 0.12026831, - 0.38707489, - 0.3909619, - 0.30499284,
    # 0.20611066,
    # 0.14419278,
    # 0.21602679,
    # 0.1191478, - 0.3690436,
    # 0.13610334,
    # 0.05834804,
    # 0.29217816,
    # 0.23812041, - 0.04528584,
    # 0.43034167 ,- 0.08670919, - 0.30285717,
    # 0.13421821,
    # 0.17290639, - 0.37454122, - 0.25547405, - 0.17652117,
    # 0.05074379, - 0.15380182,
    # - 0.03030872,
    # 0.01081423, - 0.29261155 ,- 0.30637989,
    # 0.37201026,
    # 0.41434878,
    # - 0.39517973,
    # 0.04414078,
    # 0.14545502,
    # 0.34886834, - 0.22966552, - 0.25839423,
    # 0.39728734,
    # 0.30041016,
    # 0.24496605,
    # 0.22404809, - 0.34298225, - 0.49707576,
    # - 0.22112436, - 0.09619379, - 0.4046903,
    # 0.18218657,
    # 0.0899265,
    # 0.00135533,
    # 0.43414943, - 0.30376737, - 0.20926222, - 0.30580375, - 0.14279564,
    # 0.11478333,
    # 0.47015967,
    # 0.36040241, - 0.48624241,
    # 0.1861162,
    # 0.47205363 ,- 0.00918217,
    # 0.33549356,
    # 0.03717893 ,- 0.00339029 ,- 0.25675137,
    # 0.45996918,
    # 0.34051527,
    # 0.00966093, - 0.18936049,
    # 0.27229147,
    # 0.45008121 ,- 0.34992265, - 0.31028691,
    # - 0.45870503 ,- 0.32075527,
    # 0.28070963, - 0.37757784, - 0.4250515, - 0.03189259,
    # 0.01218871,
    # 0.17991677, - 0.02318789,
    # 0.24144037]
    # true_w = np.array(true_w)
    # true_w = [-0.09064027, - 0.30717682,
    # 0.10312984,
    # 0.02839971,
    # 0.40539251, - 0.30477423,
    # 0.20827367, - 0.49457332, - 0.25870951, - 0.48844876]
    X, y, true_covariance = generate_logistic_data(N, p, true_w)
    initial_model = np.random.normal(0, 0.1, size=p)
 #    initial_model = [-0.0565327,   0.25099069, -0.37631736, -0.40326699,  0.37086017,  0.28294391,
 #  0.25751959,  0.00661737, -0.02463554,  0.10878637, -0.10427602,  0.27973042,
 # -0.10310403,  0.20214087, -0.47362878 , 0.3482037,  -0.18057214,  0.38954681,
 #  0.16493052 , 0.33260615, -0.02315252,  0.1871603,  -0.1877714,  -0.02474511,
 #  0.00828701,  0.1191221,  -0.16720248, -0.19095845, -0.05693608,  0.17292977,
 # -0.2187786 ,  0.05576835, -0.41241086, -0.24647714, -0.30771062,  0.08476041,
 #  0.178626,   -0.00465841,  0.23009452, -0.27514723, -0.15821784, -0.0613083,
 #  0.17213351 , 0.17280699, -0.36259588, -0.4945381,   0.29758038,  0.26513314,
 # -0.23988742,  0.03797761,  0.30286003,  0.2476939,  -0.17031656, -0.0688906,
 # -0.30484922,  0.2954227 ,  0.49519552,  0.19307402,  0.14021994,  0.13616312,
 # -0.0902053,  -0.1544647 , -0.22394054, -0.22230975,  0.20717917, -0.03187367,
 # -0.47420139,  0.1968122,   0.38816358, -0.20264843,  0.17297111, -0.3485448,
 # -0.22958891, -0.07204058,  0.4909282,  -0.30674114, -0.41451482, -0.27800917,
 # -0.20312731, -0.46017351,  0.00780113, -0.2153871,  -0.37317276, -0.04986955,
 # -0.14736823 , 0.2148469,  -0.31051706, -0.04299646, -0.43555296, -0.0811036,
 #  0.33460475, -0.34463609,  0.14315483,  0.17100154,  0.44674999,  0.13924505,
 #  0.06239016, -0.33690692,  0.33658157,  0.07228054]
    # initial_model = [0.09662693, 0.03003041, 0.01761898, 0.02250416, -0.10999331, 0.01580522,
    #                  -0.0531954, 0.0756385, 0.01685449, -0.0626304]
 #    initial_model = [ 1.11968489e-01, -3.91280835e-02,  2.79782111e-02, -8.85612856e-02,
 #  5.42420845e-03 , 1.21132775e-01, -8.14969449e-02, -8.68893088e-03,
 # -1.51077851e-01,  2.21992856e-02, -2.42288710e-03,  6.60845600e-02,
 #  4.52181099e-02, -4.17150436e-02 ,-2.82085676e-02, -2.31133668e-01,
 #  1.19338858e-01,  4.28293514e-02, -2.30085460e-02,  1.04148478e-01,
 # -3.29353983e-02, -3.63115230e-02 , 5.12652712e-02, -1.49487256e-01,
 #  1.93672956e-01, -7.22920797e-02, -1.98434806e-04, -9.02452439e-02,
 #  4.67146479e-02, -2.83040301e-02,  1.36180374e-02, -1.69480790e-01,
 #  1.71951717e-02, -1.87443298e-02, -7.14029635e-02,  8.00656932e-02,
 # -6.09515273e-02 , 2.45142031e-02,  5.47392146e-02, -2.00300011e-01,
 #  1.74419333e-02 , 4.21982059e-03 , 1.14546900e-01 , 7.75178838e-02,
 # -2.92260281e-02 , 1.34749279e-01,  8.10457199e-02 , 8.19659842e-02,
 # -2.05604822e-02,  4.82444740e-02 , 6.68873031e-02, -2.95941246e-03,
 #  3.92897459e-02 , 2.19389593e-03, -1.14845901e-01 , 6.09062509e-02,
 #  6.53454174e-02,  6.01825864e-02, -1.30360306e-01 , 9.73384768e-02,
 # -1.23319983e-01,  4.54672222e-02 , 4.03184858e-02, -2.50941530e-02,
 # -1.45359327e-01,  1.69612974e-02,  1.82698276e-02,  8.18835529e-02,
 #  3.03577795e-03 , 1.00577321e-01,  6.29997587e-02 , 9.44465825e-02,
 # -5.79690137e-03,  4.27771777e-02,  8.89525389e-03, -2.25691839e-02,
 #  2.70129471e-02,  7.28663642e-02, -9.26568425e-02,  1.29830157e-02,
 #  2.63695674e-02,  1.46719707e-01,  9.05040533e-02, -7.05392766e-02,
 # -9.47040687e-02,  6.03672054e-02, -2.99735427e-02,  9.97661652e-02,
 # -1.56295217e-01, -8.23938772e-02, -1.07127851e-01,  1.93960220e-01,
 #  1.11221924e-02 , 5.25730246e-02 , 1.25830450e-03, -1.23682672e-01,
 # -1.23120552e-02 , 6.37359577e-02, -1.46159899e-01,  2.79179639e-02]
 #    initial_model = np.array(initial_model)

    l2_norm_history, ail_history,  cp_history = federated_sgd(X, y, true_w, num_clients, initial_model,
                                                                      true_covariance, max_iter)

    # 创建一个DataFrame来存储这些历史数据
    df = pd.DataFrame({
        'L2 Norm History': l2_norm_history,
        'AIL History': ail_history,
        'CP History': cp_history
    })

    # 将DataFrame写入Excel文件
    df.to_excel('FedSGD_LogR_CI_N200000k20p100.xlsx', index=False)

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