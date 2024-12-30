# 导入所需的库
library(MASS)
library(dplyr)
library(parallel)
library(openxlsx)

generate_logistic_data <- function(N, p, true_w, noise = 1) {
  # 初始化均值和协方差矩阵
  mean <- rep(0, p)
  covariance <- matrix(0.2, nrow = p, ncol = p)
  diag(covariance) <- 1
  
  # 生成特征数据 X
  X <- mvrnorm(n = N, mu = mean, Sigma = covariance)
  
  # 计算线性组合 Z
  z <- X %*% true_w + rnorm(N, sd = noise)
  
  # 使用 sigmoid 函数计算概率
  probability <- 1 / (1 + exp(-z))
  
  # 生成二元的响应变量 y
  y <- rbinom(N, 1, probability)

  # 计算真实协方差矩阵
  X_covariance <- t(X) %*% X
  X_covariance_inv <- solve(X_covariance)
  true_covariance_matrix <- noise^2 * X_covariance_inv
  
  return(list(X = X, y = y, true_covariance_matrix = true_covariance_matrix))
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

logistic_loss_and_gradient <- function(w, X, y) {
  z <- X %*% w
  predictions <- sigmoid(z)
  loss <- -mean(y * safe_log(predictions + 1e-8) + (1 - y) * safe_log(1 - predictions + 1e-8))
  gradient <- t(X) %*% (predictions - y) / length(y)
  return(list(loss = loss, gradient = gradient))
}

surrogate_loss_and_gradient <- function(theta, theta_t, X, y, global_grad_t) {
  residuals <- X %*% theta - y
  loss_f_k <- 0.5 * mean(residuals^2)
  grad_f_k <- t(X) %*% residuals / length(y)

  # 计算∇f_k(θ_t)
  _, grad_f_k_t <- logistic_loss_and_gradient(theta_t, X, y)
  
  inner_product <- (grad_f_k_t - global_grad_t) %*% theta
  surrogate_loss <- loss_f_k - inner_product
  surrogate_gradient <- grad_f_k - (grad_f_k_t - global_grad_t)
  
  return(list(surrogate_loss = surrogate_loss, surrogate_gradient = surrogate_gradient))
}

client_training <- function(X_m, y_m, local_model, global_model, global_grad_t, alpha, num_local_steps) {
  for (i in 1:num_local_steps) {
    indices <- sample(1:nrow(X_m), 10, replace = FALSE)  # 使用固定的batch_size
    X_batch <- X_m[indices, ]
    y_batch <- y_m[indices]
    surrogate_results <- surrogate_loss_and_gradient(local_model, global_model, X_batch, y_batch, global_grad_t)
    local_model <- local_model - alpha * surrogate_results$surrogate_gradient
  }
  return(local_model)
}

server_aggregate <- function(local_models) {
  return(rowMeans(local_models))
}

federated_CSL <- function(X, y, true_w, num_clients, initial_model, true_covariance, max_iter = 500, tol = 1e-6, batch_size = 10,
                          initial_alpha = 0.2, decay_rate = 0.99, num_local_steps = 10) {
  N <- nrow(X)
  p <- ncol(X)
  alpha <- initial_alpha
  
  # 初始化每个机器的本地模型
  local_models <- replicate(num_clients, initial_model, simplify = FALSE)
  global_model <- rowMeans(do.call(rbind, local_models))
  
  l2_norm_history <- c()
  ail_history <- c()
  cp_history <- c()
  
  # 将数据划分给每台机器
  client_data_size <- floor(N / num_clients)
  data_splits <- lapply(0:(num_clients - 1), function(i) {
    list(X = X[(i * client_data_size + 1):(min((i + 1) * client_data_size, N)), , drop = FALSE],
         y = y[(i * client_data_size + 1):(min((i + 1) * client_data_size, N))])
  })
  
  for (iteration in 1:max_iter) {
    results <- mclapply(1:num_clients, function(m) {
      X_m <- data_splits[[m]]$X
      y_m <- data_splits[[m]]$y
      local_model <- local_models[[m]]
      local_loss_grad <- logistic_loss_and_gradient(local_model, X_m, y_m)
      return(local_loss_grad)
    }, mc.cores = num_clients)
    
    local_losses <- sapply(results, function(res) res$loss)
    local_grads <- lapply(results, function(res) res$gradient)
    
    # 中央处理器计算平均梯度
    global_grad <- rowMeans(do.call(rbind, local_grads))
    
    # 更新每个机器的本地模型
    updated_models <- mclapply(1:num_clients, function(m) {
      client_training(data_splits[[m]]$X, data_splits[[m]]$y, local_models[[m]], global_model, global_grad, alpha, num_local_steps)
    }, mc.cores = num_clients)

    # 更新本地模型和全局模型
    local_models <- updated_models
    global_model <- server_aggregate(local_models)

    # 计算损失和L2范数
    current_loss <- mean(local_losses)
    l2_norm <- sqrt(sum((global_model - true_w) ^ 2))
    l2_norm_history <- c(l2_norm_history, l2_norm)
    ail_history <- c(ail_history, NA)  # 这里可以计算 AIL
    cp_history <- c(cp_history, NA)  # 这里可以计算 CP

    print(sprintf("FedCSL %d: Loss = %.4f, L2 Norm = %.4f", iteration, current_loss, l2_norm))

    if (l2_norm < tol) {
      print(sprintf("模型收敛于第 %d 次全局更新", iteration))
      break
    }
    
    alpha <- alpha * decay_rate
  }
  
  return(list(loss_history = l2_norm_history, ail_history = ail_history, cp_history = cp_history))
}

# 主程序
start_time <- Sys.time()

N <- 200000
p <- 100
num_clients <- 20
initial_alpha <- 0.00001
batch_size <- 10
max_iter <- 300
true_w <- runif(p, -0.5, 0.5)
data <- generate_logistic_data(N, p, true_w)
X <- data$X
y <- data$y
true_covariance <- data$true_covariance_matrix
initial_model <- rnorm(p, mean = 0, sd = 0.1)

results <- federated_CSL(X, y, true_w, num_clients, initial_model, true_covariance, max_iter)

# 创建一个数据框来存储这些历史数据
df <- data.frame(
  L2_Norm_History = results$loss_history,
  AIL_History = results$ail_history,
  CP_History = results$cp_history
)

# 将数据框写入Excel文件
write.xlsx(df, "FedCSL_LogR_CI_N200000k20p100.xlsx", rowNames = FALSE)

end_time <- Sys.time()
execution_time <- end_time - start_time
print(sprintf("程序运行时间: %.4f 秒", as.numeric(execution_time)))