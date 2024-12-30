library(MASS)
library(dplyr)
library(magrittr)
library(stats)
library(pracma)
library(openxlsx)

generate_logistic_data <- function(N, p, true_w, noise = 1) {
  mean <- rep(0, p)
  covariance <- matrix(0.2, nrow = p, ncol = p)
  diag(covariance) <- 1
  X <- mvrnorm(N, mu = mean, Sigma = covariance)
  z <- X %*% true_w + rnorm(N, mean = 0, sd = noise)
  probability <- 1 / (1 + exp(-z))
  y <- rbinom(N, 1, probability)

  X_covariance <- t(X) %*% X
  X_covariance_inv <- solve(X_covariance)
  true_covariance_matrix <- noise^2 * X_covariance_inv
  return(list(X = X, y = y, true_covariance_matrix = true_covariance_matrix))
}

loss_function <- function(w, X, y) {
  z <- X %*% w
  predictions <- 1 / (1 + exp(-z))
  loss <- -mean(y * log(predictions) + (1 - y) * log(1 - predictions))
  return(loss)
}

logistic_loss_and_gradient <- function(w, X, y) {
  z <- X %*% w
  predictions <- 1 / (1 + exp(-z))
  loss <- -mean(y * log(predictions) + (1 - y) * log(1 - predictions))
  gradient <- t(X) %*% (predictions - y) / length(y)
  return(list(loss = loss, gradient = gradient))
}

line_search_wolfe <- function(w, direction, X, y, grad, c1 = 0.01, c2 = 0.5, max_iter = 10, alpha = 1.0) {
  for (i in 1:max_iter) {
    w_new <- w + alpha * direction
    results_new <- logistic_loss_and_gradient(w_new, X, y)
    loss_new <- results_new$loss
    loss_current <- loss_function(w, X, y)

    if (loss_new > loss_current + c1 * alpha * sum(grad * direction)) {
      alpha <- alpha * 0.5
    } else if (sum(results_new$gradient * direction) < c2 * sum(grad * direction)) {
      alpha <- alpha * 0.5
    } else {
      break
    }
  }
  return(alpha)
}

batch_update <- function(w, H, X, y, tol = 1e-6) {
  results <- logistic_loss_and_gradient(w, X, y)
  loss <- results$loss
  grad <- results$gradient
  
  if (sqrt(sum(grad^2)) < tol) {
    return(list(w = w, grad = grad, H = H, loss = loss))
  }

  direction <- -H %*% grad
  alpha <- line_search_wolfe(w, direction, X, y, grad)
  w_new <- w + alpha * direction

  s <- matrix(w_new - w, ncol = 1)
  results_new <- logistic_loss_and_gradient(w_new, X, y)
  grad_new <- results_new$gradient
  y_vec <- matrix(grad_new - grad, ncol = 1)

  if (t(s) %*% y_vec > 1e-8) {
    rho <- 1 / as.numeric(t(y_vec) %*% s)
    V <- diag(length(w)) - rho * (s %*% t(y_vec))
    H <- t(V) %*% H %*% V + rho * (s %*% t(s))
  }

  return(list(w = w_new, grad = grad, H = H, loss = loss))
}

client_training <- function(X_m, y_m, H, local_model, batch_size) {
  indices <- sample(1:nrow(X_m), batch_size, replace = FALSE)
  X_batch <- X_m[indices, ]
  y_batch <- y_m[indices]
  results <- batch_update(local_model, H, X_batch, y_batch)
  return(list(w = results$w, grad = results$grad, H = results$H, loss = results$loss))
}

server_aggregate <- function(local_models) {
  return(rowMeans(do.call(rbind, local_models)))
}

calculate_srr <- function(w, Sigma, Sigma_true) {
  numerator <- t(w) %*% Sigma %*% w
  denominator <- t(w) %*% Sigma_true %*% w
  return(sqrt(numerator / denominator))
}

federated_BFGS <- function(X, y, true_w, num_clients, initial_model, true_covariance, max_iter = 500, tol = 1e-6, batch_size = 10, initial_alpha = 0.2, decay_rate = 0.95) {
  N <- nrow(X)
  p <- ncol(X)
  alpha <- initial_alpha
  l2_norm_history <- c()
  ail_history <- c()
  cp_history <- c()
  cp01_history <- rep(0, max_iter)
  srr_history <- c()
  z_975 <- qnorm(0.975)
  coverage_count <- 0

  client_data_size <- N %/% num_clients
  data_splits <- lapply(0:(num_clients - 1), function(i) {
    list(X = X[(i * client_data_size + 1):((i + 1) * client_data_size), ],
         y = y[(i * client_data_size + 1):((i + 1) * client_data_size)])
  })

  w <- rep(1, p) / sqrt(p)
  global_model <- initial_model
  global_hessian <- diag(p)

  for (iteration in 1:max_iter) {
    local_models <- list()
    gradients <- list()
    local_hessians <- list()
    local_losses <- c()

    for (m in 1:num_clients) {
      X_m <- data_splits[[m]]$X
      y_m <- data_splits[[m]]$y
      results <- client_training(X_m, y_m, global_hessian, global_model, batch_size)
      local_models[[m]] <- results$w
      gradients[[m]] <- results$grad
      local_hessians[[m]] <- results$H
      local_losses[m] <- results$loss
    }

    alpha <- alpha * decay_rate
    global_model <- server_aggregate(local_models)
    global_hessian <- Reduce("+", local_hessians) / num_clients

    current_loss <- mean(local_losses)
    l2_norm <- sqrt(sum((global_model - true_w)^2))
    l2_norm_history <- c(l2_norm_history, l2_norm)

    g_outer_sum <- matrix(0, nrow = ncol(gradients[[1]]), ncol = ncol(gradients[[1]]))
    for (gi in gradients) {
      g_outer_sum <- g_outer_sum + (gi %*% t(gi))
    }
    g_outer_avg <- g_outer_sum / num_clients
    est_var <- global_hessian %*% g_outer_avg %*% global_hessian

    number_samples <- num_clients * batch_size
    ci_length <- z_975 * sqrt((t(w) %*% est_var %*% w) / number_samples)
    ci_lower <- t(w) %*% global_model - ci_length
    ci_upper <- t(w) %*% global_model + ci_length

    if (t(w) %*% true_w >= ci_lower && t(w) %*% true_w <= ci_upper) {
      coverage_count <- coverage_count + 1
      cp01_history[iteration] <- 1
    }
    
    ail <- ci_length
    ail_history <- c(ail_history, ail)

    coverage_rate <- coverage_count / iteration
    cp_history <- c(cp_history, coverage_rate)

    cat(sprintf("FedBFGS %d: Loss = %.4f, L2 Norm = %.4f, CP = %.4f, final_CP = %.1f, AIL = %.4f\n",
                iteration, current_loss, l2_norm, coverage_rate, cp01_history[iteration], ail))

    if (l2_norm < tol) {
      cat(sprintf("模型收敛于第 %d 次全局更新\n", iteration))
      break
    }
  }

  return(list(l2_norm_history = l2_norm_history, ail_history = ail_history, cp_history = cp_history))
}

# 主函数
start_time <- Sys.time()

N <- 200000
p <- 100
num_clients <- 20
initial_alpha <- 0.1
batch_size <- 20
max_iter <- 300
true_w <- runif(p, -0.5, 0.5)
data <- generate_logistic_data(N, p, true_w)

X <- data$X
y <- data$y
true_covariance <- data$true_covariance_matrix
initial_model <- rnorm(p, 0, 0.1)

results <- federated_BFGS(X, y, true_w, num_clients, initial_model, true_covariance, max_iter)

# 创建一个DataFrame来存储这些历史数据
df <- data.frame(
  `L2 Norm History` = results$l2_norm_history,
  `AIL History` = results$ail_history,
  `CP History` = results$cp_history
)

# 将DataFrame写入Excel文件
write.xlsx(df, 'FedBFGS_LogR_CI_N200000k20p100.xlsx', rowNames = FALSE)

# 计算指标
last_l2_norm <- tail(results$l2_norm_history, 1)
min_l2_norm <- min(results$l2_norm_history)
avg_l2_norm <- mean(results$l2_norm_history)

last_ail <- tail(results$ail_history, 1)
min_ail <- min(results$ail_history)
avg_ail <- mean(results$ail_history)

last_cp <- tail(results$cp_history, 1)
min_cp <- min(results$cp_history)
avg_cp <- mean(results$cp_history)

cat(sprintf("最后迭代的 L2 Norm: %.4f\n", last_l2_norm))
cat(sprintf("所有迭代中的最小 L2 Norm: %.4f\n", min_l2_norm))
cat(sprintf("所有迭代的平均 L2 Norm: %.4f\n", avg_l2_norm))

cat(sprintf("最后迭代的 AIL: %.4f\n", last_ail))
cat(sprintf("所有迭代中的最小 AIL: %.4f\n", min_ail))
cat(sprintf("所有迭代的平均 AIL: %.4f\n", avg_ail))

cat(sprintf("最后迭代的 CP: %.4f\n", last_cp))
cat(sprintf("所有迭代中的最小 CP: %.4f\n", min_cp))
cat(sprintf("所有迭代的平均 CP: %.4f\n", avg_cp))

end_time <- Sys.time()
execution_time <- end_time - start_time
cat(sprintf("程序运行时间: %.4f 秒\n", execution_time))
cat(sprintf("通信开销: %d\n", (p * p + p) * num_clients))