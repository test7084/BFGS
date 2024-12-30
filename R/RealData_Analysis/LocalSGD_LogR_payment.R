library(readr)        # 用于读取 CSV 文件
library(dplyr)        # 用于数据处理
library(magrittr)     # 用于管道操作
library(caret)        # 用于标准化
library(parallel)     # 用于并行计算
library(pROC)         # 用于评估模型
library(openxlsx)     # 用于写入 Excel 文件
library(MASS)         # 用于计算海森矩阵
library(mclust)       # 用于协方差矩阵计算

# 预测函数
predict <- function(X, w) {
  z <- X %*% w
  return(1 / (1 + exp(-z)) >= 0.5)  # 返回布尔值
}

cal_truevariance <- function(X) {
  X_covariance <- t(X) %*% X
  X_covariance_inv <- solve(X_covariance)
  return(X_covariance_inv)
}

load_and_preprocess_data <- function(file_path) {
  data <- read_csv(file_path)
  labels <- data$isFraud
  
  features <- data %>%
    select(-isFraud, -nameOrig, -nameDest) %>%
    mutate_if(is.character, as.factor) %>%
    mutate(across(everything(), ~ as.numeric(as.factor(.)))) %>%
    dummy_cols(select_columns = "type", remove_first_dummy = TRUE) %>%
    select(-type)
  
  scaler <- preProcess(features, method = c("center", "scale"))
  features <- predict(scaler, features)
  
  return(list(features = features, labels = labels))
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
  loss <- -mean(y * safe_log(predictions) + (1 - y) * safe_log(1 - predictions))
  gradient <- t(X) %*% (predictions - y) / length(y)
  return(list(loss = loss, gradient = gradient))
}

local_sgd <- function(X, y, true_w, num_clients, initial_model, true_covariance, max_iter = 500, tol = 1e-6, batch_size = 10,
                      initial_alpha = 0.9, decay_rate = 0.99) {
  N <- nrow(X)
  p <- ncol(X)
  alpha <- initial_alpha
  local_models <- replicate(num_clients, initial_model, simplify = FALSE)
  global_model <- colMeans(do.call(rbind, local_models))
  
  l2_norm_history <- c()
  loss_history <- c()
  ail_history <- c()
  cp_history <- c()
  cp01_history <- numeric(max_iter)
  srr_history <- c()
  
  z_975 <- qnorm(0.975)
  coverage_count <- 0
  accuarcy_history <- c()
  f1_history <- c()
  
  machine_data_size <- N %/% num_clients
  data_splits <- lapply(1:num_clients, function(i) {
    start_idx <- (i - 1) * machine_data_size + 1
    end_idx <- i * machine_data_size
    list(X = X[start_idx:end_idx, , drop = FALSE], y = y[start_idx:end_idx])
  })
  
  w <- rep(1, p) / sqrt(p)
  
  for (iteration in 1:max_iter) {
    results <- mclapply(1:num_clients, function(m) {
      X_m <- data_splits[[m]]$X
      y_m <- data_splits[[m]]$y
      
      start_idx <- (iteration * batch_size - 1) %% nrow(X_m) + 1
      end_idx <- start_idx + batch_size - 1
      
      if (end_idx > nrow(X_m)) {
        end_idx <- nrow(X_m)
      }
      
      if (start_idx > end_idx) {
        X_batch <- rbind(X_m[start_idx:nrow(X_m), ], X_m[1:end_idx, ])
        y_batch <- c(y_m[start_idx:nrow(X_m)], y_m[1:end_idx])
      } else {
        X_batch <- X_m[start_idx:end_idx, ]
        y_batch <- y_m[start_idx:end_idx]
      }
      
      result <- logistic_loss_and_gradient(local_models[[m]], X_batch, y_batch)
      localf1 <- f1_score(y_batch, predict(X_batch, local_models[[m]]), zero_division = 1)
      localacc <- accuracy_score(y_batch, predict(X_batch, local_models[[m]]))
      return(c(result$loss, result$gradient, localf1, localacc))
    }, mc.cores = num_clients)
    
    local_losses <- sapply(results, function(res) res[1])
    gradients <- sapply(results, function(res) res[2:(p + 1)])
    localf1 <- sapply(results, function(res) res[p + 2])
    localacc <- sapply(results, function(res) res[p + 3])
    
    for (m in 1:num_clients) {
      local_models[[m]] <- local_models[[m]] - alpha * gradients[, m]
    }
    
    alpha <- alpha * decay_rate
    global_model <- colMeans(do.call(rbind, local_models))
    
    current_loss <- mean(local_losses)
    loss_history <- c(loss_history, current_loss)
    
    l2_norm <- sqrt(sum((global_model - true_w)^2))
    l2_norm_history <- c(l2_norm_history, l2_norm)
    
    accuracy <- mean(localacc)
    accuarcy_history <- c(accuarcy_history, accuracy)
    
    f1 <- mean(localf1)
    f1_history <- c(f1_history, f1)
    
    print(sprintf("Iteration %d: Loss = %.4f, L2 Norm = %.4f, Accuracy = %.4f, F1 Score = %.4f", 
                  iteration, current_loss, l2_norm, accuracy, f1))
    
    if (l2_norm < tol) {
      print(sprintf("模型收敛于第 %d 次全局更新", iteration))
      break
    }
  }
  
  return(list(loss_history = loss_history, l2_norm_history = l2_norm_history, f1_history = f1_history, accuracy_history = accuarcy_history))
}

# 主函数
start_time <- Sys.time()

num_clients <- 20
initial_alpha <- 0.8
batch_size <- 10
max_iter <- 300
file_path <- 'payment_fraud_dataset.csv'
data <- load_and_preprocess_data(file_path)
X <- data$features
y <- data$labels

true_covariance <- cal_truevariance(X)

true_w <- c(5.6783423e-02, 6.3214862e-01, 1.0839984e+01, -1.0987664e+01, 
            3.6733518e+00, -3.9949934e+00, 2.7498127e+03, -1.4874005e-01, 
            -2.5315860e-02, -1.6703321e-01, -3.4908421e-02)

initial_model <- rnorm(ncol(X), mean = 0, sd = 0.1)

results <- local_sgd(X, y, true_w, num_clients, initial_model, true_covariance, max_iter)

# 创建一个DataFrame来存储这些历史数据
df <- data.frame(
  Loss_History = results$loss_history,
  L2_Norm_History = results$l2_norm_history,
  F1_Score = results$f1_history,
  Accuracy = results$accuracy_history
)

# 将DataFrame写入Excel文件
write.xlsx(df, 'LocalSGD_LogR_payment.xlsx', row.names = FALSE)

# 计算指标
print(sprintf("最后迭代的 Loss: %.4f", tail(results$loss_history, 1)))
print(sprintf("最后迭代的 L2 Norm: %.4f", tail(results$l2_norm_history, 1)))
print(sprintf("最后迭代的 F1 Score: %.4f", tail(results$f1_history, 1)))
print(sprintf("最后迭代的 Accuracy: %.4f", tail(results$accuracy_history, 1)))

end_time <- Sys.time()
execution_time <- end_time - start_time
print(sprintf("程序运行时间: %.4f 秒", execution_time))
print(sprintf("通信开销： %d", (ncol(X) * ncol(X) + ncol(X)) * num_clients))