library(readr)        # 用于读取 CSV 文件
library(dplyr)        # 用于数据处理
library(caret)        # 用于数据标准化
library(purrr)        # 用于函数式编程
library(magrittr)     # 用于管道操作
library(glmnet)       # 用于逻辑回归
library(openxlsx)     # 用于写入 Excel 文件
library(pROC)         # 用于评估模型
library(mclust)       # 用于协方差矩阵计算

load_and_preprocess_data <- function(file_path) {
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

cal_truevariance <- function(X) {
  X_covariance <- t(X) %*% X
  X_covariance_inv <- solve(X_covariance)
  return(X_covariance_inv)
}

safe_log <- function(x) {
  epsilon <- 1e-8
  return(log(pmax(x, epsilon)))
}

sigmoid <- function(z) {
  return(1 / (1 + exp(-z)))
}

loss_function <- function(w, X, y) {
  predictions <- sigmoid(X %*% w)
  loss <- -mean(y * safe_log(predictions) + (1 - y) * safe_log(1 - predictions))
  return(loss)
}

logistic_loss_and_gradient <- function(w, X, y) {
  z <- X %*% w
  predictions <- sigmoid(z)
  loss <- -mean(y * safe_log(predictions) + (1 - y) * safe_log(1 - predictions))
  gradient <- t(X) %*% (predictions - y) / length(y)
  return(list(loss = loss, gradient = gradient, predictions = predictions))
}

predict <- function(X, w) {
  z <- X %*% w
  return(sigmoid(z) >= 0.5)
}

line_search_wolfe <- function(w, direction, X, y, grad, c1 = 0.01, c2 = 0.5, max_iter = 10, alpha = 0.7) {
  for (i in 1:max_iter) {
    w_new <- w + alpha * direction
    loss_new <- logistic_loss_and_gradient(w_new, X, y)$loss
    loss <- logistic_loss_and_gradient(w, X, y)$loss
    grad_new <- logistic_loss_and_gradient(w_new, X, y)$gradient

    if (loss_new > loss + c1 * alpha * sum(grad * direction)) {
      alpha <- alpha * 0.5
    } else if (sum(grad_new * direction) < c2 * sum(grad * direction)) {
      alpha <- alpha * 0.5
    } else {
      break
    }
  }
  return(alpha)
}

batch_update <- function(w, H, X, y, tol = 1e-6) {
  result <- logistic_loss_and_gradient(w, X, y)
  loss <- result$loss
  grad <- result$gradient
  
  if (sqrt(sum(grad^2)) < tol) {
    return(list(w = w, grad = grad, H = H, loss = loss))
  }

  direction <- -H %*% grad
  alpha <- line_search_wolfe(w, direction, X, y, grad)
  w_new <- w + alpha * direction

  s <- w_new - w
  result_new <- logistic_loss_and_gradient(w_new, X, y)
  grad_new <- result_new$gradient
  y_vec <- grad_new - grad
  
  if (crossprod(s, y_vec) > 1e-8) {
    rho <- 1 / crossprod(y_vec, s)
    V <- diag(length(w)) - rho * (s %*% t(y_vec))
    H <- V %*% H %*% t(V) + rho * (s %*% t(s))
  }

  return(list(w = w_new, grad = grad, H = H, loss = loss))
}

client_training <- function(X_batch, y_batch, H, local_model, batch_size) {
  result <- batch_update(local_model, H, X_batch, y_batch)
  return(result)
}

server_aggregate <- function(local_models) {
  return(rowMeans(local_models))
}

federated_BFGS <- function(X, y, true_w, num_clients, initial_model, true_covariance, max_iter = 500, tol = 1e-6, batch_size = 10, initial_alpha = 0.2, decay_rate = 0.95) {
  N <- nrow(X)
  p <- ncol(X)
  
  alpha <- initial_alpha
  l2_norm_history <- c()
  loss_history <- c()
  accuarcy_history <- c()
  f1_history <- c()
  
  client_data_size <- N %/% num_clients
  data_splits <- lapply(1:num_clients, function(i) {
    start_idx <- (i - 1) * client_data_size + 1
    end_idx <- i * client_data_size
    list(X = X[start_idx:end_idx, , drop = FALSE], y = y[start_idx:end_idx])
  })

  w <- rep(1, p) / sqrt(p)
  global_model <- initial_model
  global_hessian <- diag(length(global_model))
  
  for (iteration in 1:max_iter) {
    local_models <- list()
    local_losses <- numeric(num_clients)
    local_f1 <- numeric(num_clients)
    local_accuracy <- numeric(num_clients)

    for (m in 1:num_clients) {
      X_m <- data_splits[[m]]$X
      y_m <- data_splits[[m]]$y
      start_idx <- iteration * batch_size
      end_idx <- start_idx + batch_size - 1
      X_batch <- X_m[start_idx:end_idx, , drop = FALSE]
      y_batch <- y_m[start_idx:end_idx]

      result <- client_training(X_batch, y_batch, global_hessian, global_model, batch_size)
      local_models[[m]] <- result$w
      local_losses[m] <- result$loss
      y_pred <- predict(X_batch, result$w)
      local_f1[m] <- f1_score(y_batch, y_pred, zero_division = 1)
      local_accuracy[m] <- accuracy_score(y_batch, y_pred)
    }

    alpha <- alpha * decay_rate
    global_model <- server_aggregate(do.call(rbind, local_models))
    global_hessian <- rowMeans(do.call(rbind, lapply(local_models, function(m) diag(length(m)))))
    
    current_loss <- mean(local_losses)
    loss_history <- c(loss_history, current_loss)
    l2_norm <- sqrt(sum((global_model - true_w)^2))
    l2_norm_history <- c(l2_norm_history, l2_norm)
    accuracy <- mean(local_accuracy)
    accuarcy_history <- c(accuarcy_history, accuracy)
    f1 <- mean(local_f1)
    f1_history <- c(f1_history, f1)

    cat(sprintf("Iteration %d: Loss = %.4f, L2 Norm = %.4f, Accuracy = %.4f, F1 Score = %.4f\n", iteration, current_loss, l2_norm, accuracy, f1))
    
    if (l2_norm < tol) {
      cat(sprintf("模型收敛于第 %d 次全局更新\n", iteration))
      break
    }
  }
  
  return(list(loss_history = loss_history, l2_norm_history = l2_norm_history, f1_history = f1_history, accuracy_history = accuarcy_history))
}

# 主函数
start_time <- Sys.time()

num_clients <- 20
initial_alpha <- 1
batch_size <- 10
max_iter <- 300
file_path <- 'payment_fraud_dataset.csv'
data <- load_and_preprocess_data(file_path)
X <- data$features
y <- data$labels
N <- nrow(X)
p <- ncol(X)
true_covariance <- cal_truevariance(X)

true_w <- c(5.6783423e-02, 6.3214862e-01, 1.0839984e+01, -1.0987664e+01, 3.6733518e+00, -3.9949934e+00, 2.7498127e+03, -1.4874005e-01, -2.5315860e-02, -1.6703321e-01, -3.4908421e-02)

initial_model <- rnorm(p, 0, 0.1)

results <- federated_BFGS(X, y, true_w, num_clients, initial_model, true_covariance, max_iter)

# 创建一个DataFrame来存储这些历史数据
df <- data.frame(
  Loss_History = results$loss_history,
  L2_Norm_History = results$l2_norm_history,
  F1_Score = results$f1_history,
  Accuracy = results$accuracy_history
)

# 将DataFrame写入Excel文件
write.xlsx(df, 'FedBFGS_LogR_payment.xlsx', row.names = FALSE)

# 计算指标
print(sprintf("最后迭代的 Loss: %.4f", tail(results$loss_history, 1)))
print(sprintf("最后迭代的 L2 Norm: %.4f", tail(results$l2_norm_history, 1)))
print(sprintf("最后迭代的 F1 Score: %.4f", tail(results$f1_history, 1)))
print(sprintf("最后迭代的 Accuracy: %.4f", tail(results$accuracy_history, 1)))

end_time <- Sys.time()
execution_time <- end_time - start_time
print(sprintf("程序运行时间: %.4f 秒", execution_time))
