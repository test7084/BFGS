import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,log_loss
from scipy.optimize import minimize
import torch
import numpy as np

def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset."""
    data = pd.read_csv(file_path)
    labels = data['isFraud'].values
    features = data.drop(['isFraud', 'nameOrig', 'nameDest'], axis=1)
    features = pd.get_dummies(features, columns=['type'], drop_first=True)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return features, labels


def sigmoid(z):
    return 1 / (1 + torch.exp(-z))

def logistic_loss_and_gradient(w, X, y):
    # PyTorch 自动求导版
    z = X @ w
    predictions = sigmoid(z)
    loss = -torch.mean(y * torch.log(predictions + 1e-8) + (1 - y) * torch.log(1 - predictions + 1e-8))
    return loss

def train_logistic_regression_adam(X, y, learning_rate=0.14, max_iter=20000):
    # 转换为 PyTorch 张量
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)  # 确保 y 是列向量
    w = torch.empty((X.shape[1], 1), dtype=torch.float32, requires_grad=True)
    w.data.uniform_(-0.01, 0.01)  # 使用 `.data` 修改其数据
    # w = torch.tensor([4.4605281e-02, 4.9298957e-01, 9.1079741e+00, -9.2429676e+00,2.7651975e+00, -2.9988246e+00,2.5074570e+03,-1.3657297e-01, -2.8153747e-02, -1.5586638e-01, -4.1844092e-02],dtype=torch.float32)
    # w=torch.tensor(w,dtype=torch.float32)

    # 使用 Adam 优化器
    optimizer = torch.optim.Adam([w], lr=learning_rate)

    for i in range(max_iter):
        # 清零梯度
        optimizer.zero_grad()
        # 计算损失
        loss = logistic_loss_and_gradient(w, X, y)
        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()
        # 打印当前损失值
        if i % 10 == 0 or i == max_iter - 1:  # 每 10 次迭代打印一次
            print(f"Iteration {i}: Loss = {loss.item()}")
    return w.detach().numpy()  # 返回 NumPy 格式的权重


# 加载和预处理数据
file_path = 'payment_fraud_dataset.csv'
X, y = load_and_preprocess_data(file_path)

# 训练逻辑回归模型

# 训练逻辑回归模型
weights = train_logistic_regression_adam(X, y)
print("最终权重:", weights)

# 预测函数
def predict(X, w):
    z = X @ w
    return 1 / (1 + np.exp(-z)) >= 0.5  # 返回布尔值

# 预测测试集
y_pred = predict(X, weights)
f1 = f1_score(y, y_pred, average='binary')
accuracy = accuracy_score(y, y_pred)
# 评估模型
print(classification_report(y, y_pred))

# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # 训练逻辑回归模型
# weights = train_logistic_regression(X_train, y_train)
# print("true_w",weights)

# # 预测函数
# def predict(X, w):
#     z = X @ w
#     return 1 / (1 + np.exp(-z)) >= 0.5  # 返回布尔值


# # 预测测试集
# y_pred = predict(X_test, weights)
# f1 = f1_score(y_test, y_pred, average='binary')
# accuracy = accuracy_score(y_test, y_pred)
# # 评估模型
# print(classification_report(y_test, y_pred))