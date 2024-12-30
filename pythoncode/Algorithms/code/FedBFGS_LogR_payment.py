import autograd.numpy as np
from autograd import grad
import pandas as pd
import time
from autograd import grad, hessian
from scipy.stats import norm
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score


def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset."""
    # 读取数据
    data = pd.read_csv(file_path)

    # 分离标签列
    labels = data['isFraud'].values  # 目标标签

    # 移除非数值列 'nameOrig', 'nameDest'
    features = data.drop(['isFraud', 'nameOrig', 'nameDest'], axis=1)

    # 对类别列 'type' 进行独热编码
    features = pd.get_dummies(features, columns=['type'], drop_first=True)

    # 标准化数值特征
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return features, labels


def cal_truevariance(X):
    # 计算真实协方差矩阵
    X_covariance = np.dot(X.T, X)
    X_covariance_inv = np.linalg.inv(X_covariance)
    true_covariance_matrix = X_covariance_inv
    return true_covariance_matrix

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
    return loss, gradient,predictions


# def loss_function(w, X, y):
#     # 计算预测概率，直接使用 sigmoid 公式
#     z = X @ w
#     predictions = 1 / (1 + np.exp(-z))
#     # 计算逻辑回归的负对数似然损失
#     loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
#     return loss

# 预测函数
def predict(X, w):
    z = X @ w
    return 1 / (1 + np.exp(-z)) >= 0.5  # 返回布尔值

# 逻辑回归损失和梯度计算
# def logistic_loss_and_gradient(w, X, y):
#     # 计算预测概率，直接使用 sigmoid 公式
#     z = X @ w
#     predictions = 1 / (1 + np.exp(-z))
#
#     # 计算逻辑回归的负对数似然损失
#     loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
#
#     # 计算梯度
#     gradient = X.T @ (predictions - y) / len(y)
#
#     return loss, gradient,predictions


# #额外的中心化算法，使用全部的的数据，进行梯度下降，得到true_w
# def centralized_gradient_descent(X, y, lr=0.01, max_iter=1000, tol=1e-6):
#     """
#     使用全部数据进行梯度下降训练，寻找最优解 true_w。
#     :param X: 特征矩阵
#     :param y: 标签
#     :param lr: 学习率
#     :param max_iter: 最大迭代次数
#     :param tol: 收敛阈值
#     :return: 最优参数 true_w，损失历史 loss_history
#     """
#     n_samples, n_features = X.shape
#     w = np.zeros(n_features)  # 初始化参数为零
#     # np.random.uniform(-0.5, 0.5, size=p)
#     loss_history = []
#
#     for iteration in range(max_iter):
#         # 计算预测值
#         z = np.dot(X, w)
#         predictions = 1 / (1 + np.exp(-z))
#         # 计算损失 (负对数似然函数)
#         loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
#         loss_history.append(loss)
#         # 计算梯度
#         gradient = np.dot(X.T, (predictions - y)) / n_samples
#         # 更新参数
#         w -= lr * gradient
#         # 检查收敛
#         if np.linalg.norm(gradient) < tol:
#             print(f"中心化梯度下降在第 {iteration} 次迭代时收敛。")
#             break
#
#         # 输出每迭代的损失
#         if iteration % 100 == 0 or iteration == max_iter - 1:
#             print(f"Iteration {iteration}: Loss = {loss}")
#
#     return w

# Wolfe条件的线性搜索函数
def line_search_wolfe(w, direction, X, y, grad, c1=0.01, c2=0.5, max_iter=10, alpha=0.7):
    for _ in range(max_iter):
        w_new = w + alpha * direction
        loss_new, grad_new,_ = logistic_loss_and_gradient(w_new, X, y)
        loss, _,_ = logistic_loss_and_gradient(w, X, y)

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
    loss, grad, predictions = logistic_loss_and_gradient(w, X, y)
    if np.linalg.norm(grad) < tol:
        return w, grad, H,loss

    # 使用Hessian矩阵计算更新方向
    # direction = -np.linalg.solve(H, grad)  # 用 np.linalg.solve 解方程 H @ direction = -grad

    direction = -np.dot(H, grad)
    # 线性搜索找最佳步长 alpha
    alpha = line_search_wolfe(w, direction, X, y, grad)
    w_new = w + alpha * direction

    # 更新 BFGS 近似
    s = (w_new - w).reshape(-1, 1)
    loss, grad_new,_ = logistic_loss_and_gradient(w_new, X, y)
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

# def client_training(X_m, y_m, H, local_model, batch_size):
#     # 随机选择一个batch
#     indices = np.random.choice(len(X_m), batch_size, replace=False)
#     X_batch = X_m[indices]
#     y_batch = y_m[indices]
#     local_model, grad, H, loss ,predictions= batch_update(local_model, H, X_batch, y_batch)
#     return local_model, grad, H, loss,predictions

def client_training(X_batch, y_batch, H, local_model, batch_size):
    # 随机选择一个batch
    # indices = np.random.choice(len(X_m), batch_size, replace=False)
    # X_batch = X_m[indices]
    # y_batch = y_m[indices]
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
    loss_history = []
    cp_history = []
    cp01_history = np.zeros(max_iter)
    srr_history = []
    z_975 = norm.ppf(0.975)  # 计算标准正态分布的 0.975 分位数，即 Φ^(-1)(0.975)
    # 置信区间覆盖率的初始化
    coverage_count = 0
    accuarcy_history = []
    f1_history = []

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
        local_f1=[]
        local_accuracy=[]

        # 每个客户端进行本地训练
        for m in range(num_clients):
            X_m, y_m = data_splits[m]
            # 计算当前批次的起始和结束索引
            start_idx = iteration * batch_size
            end_idx = start_idx + batch_size
            # 按顺序选择当前批次数据
            X_batch, y_batch = X_m[start_idx:end_idx], y_m[start_idx:end_idx]
            local_model, grad, local_hessian, local_loss = client_training(X_batch, y_batch, global_hessian, global_model.copy(), batch_size)
            local_models.append(local_model)
            gradients.append(grad)  # 存储每个客户端的梯度
            local_hessians.append(local_hessian) #每个客户端的本地hessian
            local_losses.append(local_loss)  #每个客户端的本地loss
            y_pred = predict(X_batch, local_model)
            localf1 = f1_score(y_batch, y_pred,zero_division=1)
            local_f1.append(localf1)
            # predicted_classes = (predictions > 0.5).astype(int)  # 假设阈值为 0.5
            # local_f1.append(f1_score(y_batch, predicted_classes)) #每个客户端的本地F1
            # accuracy = accuracy_score(y, y_pred)
            local_accuracy.append(accuracy_score(y_batch, y_pred)) #每个客户端的本地acc
        # 更新学习率
        alpha *= decay_rate
        # 服务器聚合模型
        global_model = server_aggregate(local_models)
        # 全局hessian
        global_hessian = np.mean(local_hessians, axis=0)  # 聚合全局 Hessian 矩阵

        # 计算损失、L2 范数等指标
        # current_loss, _ = loss_and_gradient(global_model, X, y)
        current_loss = np.mean(local_losses, axis=0)
        loss_history.append(current_loss)
        l2_norm = np.linalg.norm(global_model - true_w)
        l2_norm_history.append(l2_norm)
        #acc和F1 score
        accuracy = np.mean(local_accuracy,axis=0)
        accuarcy_history.append(accuracy)
        f1 = np.mean(local_f1,axis=0)
        f1_history.append(f1)


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
        # 计算准确率
        # predictions = predict(X, global_model)
        # accuracy = accuracy_score(y, predictions)
        # accuracy_history.append(accuracy)

        # # CP覆盖率
        # coverage_rate = coverage_count / (iteration + 1)
        # cp_history.append((coverage_rate))

        print(
            f"FedBFGS{iteration + 1}: Loss = {current_loss:.4f}, L2 Norm = {l2_norm:.4f},  AIL={ail:.4f}, accuracy={accuracy:.4f}, f1_score={f1:4f}")

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

    return loss_history ,l2_norm_history, ail_history,  f1_history, accuarcy_history  # 返回覆盖率


# 主函数
if __name__ == "__main__":
    start_time = time.time()


    num_clients = 20
    initial_alpha = 1
    batch_size = 10
    max_iter = 300
    file_path = 'payment_fraud_dataset.csv'
    X, y = load_and_preprocess_data(file_path)
    N, p = X.shape
    true_covariance=cal_truevariance(X)

    true_w = np.array([ 5.6783423e-02,  6.3214862e-01, 1.0839984e+01, -1.0987664e+01, 3.6733518e+00,-3.9949934e+00,
  2.7498127e+03,  -1.4874005e-01, -2.5315860e-02,  -1.6703321e-01, -3.4908421e-02])
    # true_w = centralized_gradient_descent(X, y)
    # print("true_w",true_w)

    initial_model = np.random.normal(0, 0.1, size=p)
    # print("intial_model", initial_model)

    loss_history ,l2_norm_history, ail_history,  f1_history, accuarcy_history = federated_BFGS(X, y, true_w, num_clients, initial_model,true_covariance, max_iter)

    # 创建一个DataFrame来存储这些历史数据
    df = pd.DataFrame({
        'Loss History': loss_history,
        'L2 Norm History': l2_norm_history,
        'AIL History': ail_history,
        'F1_score': f1_history,
        'Accuracy': accuarcy_history
    })

    # 将DataFrame写入Excel文件
    df.to_excel('FedBFGS_LogR_payment.xlsx', index=False)

    # 计算指标
    # Calculate metrics
    last_loss = loss_history[-1]
    min_loss = np.min(loss_history)
    avg_loss = np.mean(loss_history)
    max_loss = np.max(loss_history)

    last_l2_norm = l2_norm_history[-1]
    min_l2_norm = np.min(l2_norm_history)
    avg_l2_norm = np.mean(l2_norm_history)
    max_l2_norm = np.max(l2_norm_history)

    last_ail = ail_history[-1]
    min_ail = np.min(ail_history)
    avg_ail = np.mean(ail_history)
    max_ail = np.max(ail_history)

    last_f1 = f1_history[-1]
    min_f1= np.min(f1_history)
    avg_f1 = np.mean(f1_history)
    max_f1 = np.max(f1_history)



    last_acc = accuarcy_history[-1]
    min_acc = np.min(accuarcy_history)
    avg_acc = np.mean(accuarcy_history)
    max_acc = np.max(accuarcy_history)

    print(f"最后迭代的 Loss: {last_loss:.4f}")
    print(f"所有迭代的平均 Loss: {avg_loss:.4f}")
    print(f"所有迭代中的最小 Loss: {min_loss:.4f}")
    print(f"所有迭代中的最大 Loss: {max_loss:.4f}")


    print(f"最后迭代的 L2 Norm: {last_l2_norm:.4f}")
    print(f"所有迭代的平均 L2 Norm: {avg_l2_norm:.4f}")
    print(f"所有迭代中的最小 L2 Norm: {min_l2_norm:.4f}")
    print(f"所有迭代中的最大 L2 Norm: {max_l2_norm:.4f}")


    print(f"最后迭代的 AIL: {last_ail:.4f}")
    print(f"所有迭代的平均 AIL: {avg_ail:.4f}")
    print(f"所有迭代中的最小 AIL: {min_ail:.4f}")
    print(f"所有迭代中的最大 AIL: {max_ail:.4f}")

    print(f"最后迭代的 acc: {last_acc:.4f}")
    print(f"所有迭代的平均 acc: {avg_acc:.4f}")
    print(f"所有迭代中的最小 acc: {min_acc:.4f}")
    print(f"所有迭代中的最大 acc: {max_acc:.4f}")

    print(f"最后迭代的 F1_score: {last_f1:.4f}")
    print(f"所有迭代的平均 F1_score: {avg_f1:.4f}")
    print(f"所有迭代中的最小 F1_score: {min_f1:.4f}")
    print(f"所有迭代中的最大 F1_score: {max_f1:.4f}")


    end_time = time.time()
    execution_time = end_time - start_time
    print(f"程序运行时间: {execution_time:.4f} 秒")
    print("通信开销：", (p * p + p) * num_clients)