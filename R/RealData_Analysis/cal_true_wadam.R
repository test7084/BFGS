library(readr)        # 用于读取 CSV 文件
library(dplyr)        # 用于数据处理
library(caret)        # 用于数据缩放
library(pROC)         # 用于评估模型
library(glmnet)       # 用于逻辑回归
library(purrr)        # 用于函数式编程
library(tidyr)        # 用于数据整理

load_and_preprocess_data <- function(file_path) {
  # 加载和预处理数据集
  data <- read_csv(file_path)
  labels <- data$isFraud
  features <- data %>%
    select(-isFraud, -nameOrig, -nameDest) %>%
    mutate_if(is.character, as.factor) %>%
    mutate(across(everything(), ~ as.numeric(as.factor(.)))) %>%
    dummy_cols(select_columns = "type", remove_first_dummy = TRUE) %>%
    select(-type)  # 去掉原始的 type 列
  features <- scale(features)  # 数据标准化
  return(list(features = features, labels = labels))
}

sigmoid <- function(z) {
  return(1 / (1 + exp(-z)))
}

logistic_loss_and_gradient <- function(w, X, y) {
  z <- X %*% w
  predictions <- sigmoid(z)
  loss <- -mean(y * log(predictions + 1e-8) + (1 - y) * log(1 - predictions + 1e-8))
  return(loss)
}

train_logistic_regression_adam <- function(X, y, learning_rate = 0.14, max_iter = 20000) {
  # 初始化权重
  w <- matrix(runif(ncol(X), -0.01, 0.01), ncol = 1)

  for (i in 1:max_iter) {
    z <- X %*% w
    predictions <- sigmoid(z)
    
    # 计算梯度
    gradient <- t(X) %*% (predictions - y) / length(y)
    
    # 更新权重
    w <- w - learning_rate * gradient
    
    # 打印当前损失值
    if (i %% 10 == 0 || i == max_iter) {
      loss <- logistic_loss_and_gradient(w, X, y)
      cat(sprintf("Iteration %d: Loss = %.4f\n", i, loss))
    }
  }
  return(w)
}

# 加载和预处理数据
file_path <- 'payment_fraud_dataset.csv'
data <- load_and_preprocess_data(file_path)
X <- as.matrix(data$features)
y <- as.matrix(data$labels)

# 训练逻辑回归模型
weights <- train_logistic_regression_adam(X, y)
cat("最终权重:", weights, "\n")

# 预测函数
predict <- function(X, w) {
  z <- X %*% w
  return(sigmoid(z) >= 0.5)  # 返回布尔值
}

# 预测
y_pred <- predict(X, weights)

# 评估模型
confusion_matrix <- table(y, y_pred)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
precision <- confusion_matrix[2, 2] / sum(confusion_matrix[, 2])
recall <- confusion_matrix[2, 2] / sum(confusion_matrix[2, ])
f1 <- 2 * (precision * recall) / (precision + recall)

# 打印评估结果
cat("准确率:", accuracy, "\n")
cat("精确率:", precision, "\n")
cat("召回率:", recall, "\n")
cat("F1 分数:", f1, "\n")