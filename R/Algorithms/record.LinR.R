library(openxlsx) # 用于保存 Excel 文件
library(MASS)     # 用于生成多元正态分布
library(parallel) # 用于并行计算
library(autograd) # 用于自动微分

# 生成数据的函数
generate_data <- function(N, p, true_w, noise) {
  mean <- rep(0, p)
  covariance <- matrix(0.2, nrow = p, ncol = p)
  diag(covariance) <- 1
  X <- mvrnorm(n = N, mu = mean, Sigma = covariance)
  z <- X %*% true_w + rnorm(N, sd = noise)
  y <- X %*% true_w + rnorm(N, sd = noise)
  true_covariance_matrix <- noise^2 * solve(t(X) %*% X)
  return(list(X = X, y = y, true_covariance = true_covariance_matrix))
}

# Local SGD 函数（示例）
local_sgd <- function(X, y, true_w, num_clients, initial_model, true_covariance, max_iter) {
  # 示例代码（请根据实际算法进行实现）
  l2_norm_history <- runif(max_iter) # 假设的 L2 范数历史
  ail_history <- runif(max_iter)      # 假设的 AIL 历史
  srr_history <- runif(max_iter)      # 假设的 SRR 历史
  cp_history <- runif(max_iter)       # 假设的 CP 历史
  return(list(l2_norm_history, ail_history, srr_history, cp_history))
}

# Federated SGD 函数（示例）
federated_sgd <- function(X, y, true_w, num_clients, initial_model, true_covariance, max_iter) {
  # 示例代码（请根据实际算法进行实现）
  l2_norm_history <- runif(max_iter)
  ail_history <- runif(max_iter)
  srr_history <- runif(max_iter)
  cp_history <- runif(max_iter)
  return(list(l2_norm_history, ail_history, srr_history, cp_history))
}

# Federated CSL 函数（示例）
federated_CSL <- function(X, y, true_w, num_clients, initial_model, true_covariance, max_iter) {
  # 示例代码（请根据实际算法进行实现）
  l2_norm_history <- runif(max_iter)
  ail_history <- runif(max_iter)
  srr_history <- runif(max_iter)
  cp_history <- runif(max_iter)
  return(list(l2_norm_history, ail_history, srr_history, cp_history))
}

# Federated BFGS 函数（示例）
federated_BFGS <- function(X, y, true_w, num_clients, initial_model, true_covariance, max_iter) {
  # 示例代码（请根据实际算法进行实现）
  l2_norm_history <- runif(max_iter)
  ail_history <- runif(max_iter)
  srr_history <- runif(max_iter)
  cp_history <- runif(max_iter)
  return(list(l2_norm_history, ail_history, srr_history, cp_history))
}

# 主程序
N <- 10000
p <- 5
num_clients <- 10
initial_alpha <- 0.1
batch_size <- 10
max_iter <- 400
noise <- 1

true_ws <- list()
true_w <- runif(p, -0.5, 0.5)
X_data <- generate_data(N, p, true_w, noise)
X <- X_data$X
y <- X_data$y
true_covariance <- X_data$true_covariance

true_ws[[1]] <- true_w
initial_model <- rnorm(p, mean = 0, sd = 0.1)

# 调用不同的算法
results_localsgd <- local_sgd(X, y, true_w, num_clients, initial_model, true_covariance, max_iter)
results_fedsgd <- federated_sgd(X, y, true_w, num_clients, initial_model, true_covariance, max_iter)
results_fedCSL <- federated_CSL(X, y, true_w, num_clients, initial_model, true_covariance, max_iter)
results_fedBFGS <- federated_BFGS(X, y, true_w, num_clients, initial_model, true_covariance, max_iter)

# 创建数据框
data <- data.frame(
  Iteration = 1:max_iter,
  `L2 Norm (Local SGD)` = results_localsgd[[1]],
  `AIL (Local SGD)` = results_localsgd[[2]],
  `SRR (Local SGD)` = results_localsgd[[3]],
  `CP (Local SGD)` = results_localsgd[[4]],
  
  `L2 Norm (Federated SGD)` = results_fedsgd[[1]],
  `AIL (Federated SGD)` = results_fedsgd[[2]],
  `SRR (Federated SGD)` = results_fedsgd[[3]],
  `CP (Federated SGD)` = results_fedsgd[[4]],
  
  `L2 Norm (Federated CSL)` = results_fedCSL[[1]],
  `AIL (Federated CSL)` = results_fedCSL[[2]],
  `SRR (Federated CSL)` = results_fedCSL[[3]],
  `CP (Federated CSL)` = results_fedCSL[[4]],
  
  `L2 Norm (Federated BFGS)` = results_fedBFGS[[1]],
  `AIL (Federated BFGS)` = results_fedBFGS[[2]],
  `SRR (Federated BFGS)` = results_fedBFGS[[3]],
  `CP (Federated BFGS)` = results_fedBFGS[[4]]
)

# 保存为 Excel 文件
write.xlsx(data, 'results_LinR.xlsx', rowNames = FALSE)