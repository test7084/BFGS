library(MASS)  # 用于线性代数
library(dplyr)  # 数据处理
library(e1071)  # 用于标准化
library(parallel)  # 并行计算
library(readr)  # 读取 CSV 文件
library(purrr)  # 用于函数式编程

# 计算真实协方差矩阵
cal_truevariance <- function(X) {
  X_covariance <- t(X) %*% X
  true_covariance_matrix <- ginv(X_covariance)
  return(true_covariance_matrix)
}

# 加载和预处理数据
load_and_preprocess_data <- function(file_path) {
  data <- read_csv(file_path)

  labels <- data$isFraud  # 目标标签
  features <- data %>% select(-isFraud, -nameOrig, -nameDest)

  # 对类别列 'type' 进行独热编码
  features <- model.matrix(~ type - 1, data = features) %>% as.data.frame()
  
  # 标准化数值特征
  features <- scale(features)

  return(list(features = as.matrix(features), labels = labels))
}

safe_log <- function(x) {
  epsilon <- 1e-8
  return(log(pmax(x, epsilon)))
}

sigmoid <- function(z) {
  return(1 / (1 + exp(-z)))
}

loss_function <- function(w, X, y) {
  z <- X %*% w
  predictions <- sigmoid(z)
  loss <- -mean(y * safe_log(predictions) + (1 - y) * safe_log(1 - predictions))
  return(loss)
}

# 预测函数
predict <- function(X, w) {
  z <- X %*% w
  return(as.numeric(sigmoid(z) >= 0.5))
}

# 逻辑回归损失和梯度计算
logistic_loss_and_gradient <- function(w, X, y) {
  z <- X %*% w
  predictions <- sigmoid(z)

  loss <- -mean(y * safe_log(predictions + 1e-8) + (1 - y) * safe_log(1 - predictions + 1e-8))
  gradient <- t(X) %*% (predictions - y) / length(y)

  return(list(loss = loss, gradient = gradient))
}

# 代理损失和梯度计算
surrogate_loss_and_gradient <- function(theta, theta_t, X, y, global_grad_t) {
  residuals <- X %*% theta - y
  loss_f_k <- 0.5 * mean(residuals^2)
  grad_f_k <- t(X) %*% residuals / length(y)

  # 计算∇f_k(θ_t)
  grad_f_k_t <- logistic_loss_and_gradient(theta_t, X, y)$gradient

  inner_product <- (grad_f_k_t - global_grad_t) %*% theta
  surrogate_loss <- loss_f_k - inner_product

  surrogate_gradient <- grad_f_k - (grad_f_k_t - global_grad_t)

  return(list(surrogate_loss = surrogate_loss, surrogate_gradient = surrogate_gradient))
}

# 客户端训练函数
client_training <- function(X_m, y_m, local_model, global_model, global_grad_t, alpha, num_local_steps) {
  for (i in 1:num_local_steps) {
    indices <- sample(1:nrow(X_m), 10, replace = FALSE)
    X_batch <- X_m[indices, ]
    y_batch <- y_m[indices]
    surrogate_result <- surrogate_loss_and_gradient(local_model, global_model, X_batch, y_batch, global_grad_t)
    local_model <- local_model - alpha * surrogate_result$surrogate_gradient
  }
  return(local_model)
}

# 服务器聚合模型
server_aggregate <- function(local_models) {
  return(rowMeans(do.call(rbind, local_models)))
}

# 联邦训练函数
federated_CSL <- function(X, y, true_w, num_clients, initial_model, true_covariance, max_iter = 500, tol = 1e-6, 
                          batch_size = 10, initial_alpha = 0.2, decay_rate = 0.99, num_local_steps = 10) {
  N <- nrow(X)
  p <- ncol(X)
  alpha <- initial_alpha

  # 初始化每个机器的本地模型
  local_models <- replicate(num_clients, initial_model, simplify = FALSE)
  global_model <- rowMeans(do.call(rbind, local_models))
  
  loss_history <- c()
  accuracy_history <- c()
  f1_history <- c()
  
  # 将数据划分给每台机器
  client_data_size <- floor(N / num_clients)
  data_splits <- split(data.frame(X, y), rep(1:num_clients, each = client_data_size, length.out = N))

  for (iteration in 1:max_iter) {
    local_grads <- list()
    local_losses <- c()

    # 每台机器计算局部损失和梯度
    for (m in 1:num_clients) {
      local_result <- logistic_loss_and_gradient(local_models[[m]], data_splits[[m]][, -ncol(data_splits[[m]])], data_splits[[m]][, ncol(data_splits[[m]])])
      local_losses[m] <- local_result$loss
      local_grads[[m]] <- local_result$gradient
    }

    # 中央处理器计算平均梯度并广播
    global_grad <- Reduce("+", local_grads) / num_clients

    # 更新每台机器的本地模型
    for (m in 1:num_clients) {
      local_models[[m]] <- local_models[[m]] - alpha * local_grads[[m]]
    }

    # 计算全局模型
    global_model <- rowMeans(do.call(rbind, local_models))
    
    # 记录损失
    current_loss <- mean(local_losses)
    loss_history <- c(loss_history, current_loss)

    # 输出当前损失
    cat(sprintf("Iteration %d: Loss = %.4f\n", iteration, current_loss))

    if (abs(current_loss) < tol) {
      cat(sprintf("模型收敛于第 %d 次全局更新\n", iteration))
      break
    }

    alpha <- alpha * decay_rate
  }
  
  return(loss_history)
}

# 主函数
file_path <- 'payment_fraud_dataset.csv'
data <- load_and_preprocess_data(file_path)
X <- data$features
y <- data$labels

num_clients <- 20
initial_alpha <- 1
batch_size <- 10
max_iter <- 300
true_covariance <- cal_truevariance(X)

true_w <- c(5.6783423e-02, 6.3214862e-01, 1.0839984e+01, -1.0987664e+01, 3.6733518e+00, -3.9949934e+00,
            2.7498127e+03, -1.4874005e-01, -2.5315860e-02, -1.6703321e-01, -3.4908421e-02)

initial_model <- rnorm(ncol(X), mean = 0, sd = 0.1)

loss_history <- federated_CSL(X, y, true_w, num_clients, initial_model, true_covariance, max_iter)

# 保存损失历史到CSV文件
write_csv(data.frame(Loss_History = loss_history), 'FedCSL_LogR_payment.csv')