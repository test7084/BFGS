library(data.table)
library(caret)
library(MASS)
library(parallel)
library(dplyr)

load_and_preprocess_data <- function(file_path) {
  # 读取数据
  data <- fread(file_path)
  
  # 分离标签列
  labels <- data$isFraud
  
  # 移除非数值列 'nameOrig', 'nameDest'
  features <- data[, !c("isFraud", "nameOrig", "nameDest"), with = FALSE]
  
  # 对类别列 'type' 进行独热编码
  features <- dummyVars("~ .", data = features, fullRank = TRUE)
  features <- predict(features, newdata = features)
  
  # 标准化数值特征
  features <- scale(features)
  
  return(list(features = features, labels = labels))
}

cal_truevariance <- function(X) {
  # 计算真实协方差矩阵
  X_covariance <- t(X) %*% X
  true_covariance_matrix <- solve(X_covariance)
  return(true_covariance_matrix)
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

predict <- function(X, w) {
  z <- X %*% w
  return(as.numeric(sigmoid(z) >= 0.5))
}

local_sgd <- function(X, y, local_model, batch_size, alpha) {
  result <- logistic_loss_and_gradient(local_model, X, y)
  loss <- result$loss
  grad <- result$gradient
  local_model <- local_model - alpha * grad
  return(list(local_model = local_model, gradient = grad, loss = loss))
}

server_aggregate <- function(local_models) {
  return(rowMeans(do.call(rbind, local_models)))
}

federated_sgd <- function(X, y, true_w, num_clients, initial_model, true_covariance, max_iter = 500, tol = 1e-6, batch_size = 10,
                          initial_alpha = 0.9, decay_rate = 0.99) {
  N <- nrow(X)
  p <- ncol(X)
  alpha <- initial_alpha
  local_models <- replicate(num_clients, initial_model, simplify = FALSE)
  global_model <- rowMeans(do.call(rbind, local_models))
  l2_norm_history <- c()
  accuarcy_history <- c()
  f1_history <- c()
  
  machine_data_size <- floor(N / num_clients)
  data_splits <- lapply(0:(num_clients - 1), function(i) {
    start_idx <- i * machine_data_size + 1
    end_idx <- ifelse(i == num_clients - 1, N, (i + 1) * machine_data_size)
    list(X = X[start_idx:end_idx, , drop = FALSE], y = y[start_idx:end_idx])
  })
  
  for (iteration in 1:max_iter) {
    results <- mclapply(1:num_clients, function(m) {
      X_m <- data_splits[[m]]$X
      y_m <- data_splits[[m]]$y
      updated_model <- local_models[[m]]
      updated_result <- local_sgd(X_m, y_m, updated_model, batch_size, alpha)
      return(updated_result)
    }, mc.cores = num_clients)
    
    for (m in 1:num_clients) {
      local_models[[m]] <- results[[m]]$local_model
    }
    
    global_model <- server_aggregate(local_models)
    
    # 计算损失、L2 范数等指标
    current_loss <- mean(sapply(results, function(res) res$loss))
    l2_norm <- sqrt(sum((global_model - true_w) ^ 2))
    l2_norm_history <- c(l2_norm_history, l2_norm)
    
    y_pred <- predict(X, global_model)
    accuracy <- mean(y_pred == y)
    accuarcy_history <- c(accuarcy_history, accuracy)
    
    # F1 score
    f1 <- F1_Score(y, y_pred)
    f1_history <- c(f1_history, f1)

    print(sprintf("FedSGD %d: Loss = %.4f, L2 Norm = %.4f, Accuracy = %.4f, F1 Score = %.4f", 
                  iteration, current_loss, l2_norm, accuracy, f1))
    
    if (l2_norm < tol) {
      print(sprintf("模型收敛于第 %d 次全局更新", iteration))
      break
    }
    
    alpha <- alpha * decay_rate
  }
  
  return(list(l2_norm_history = l2_norm_history, f1_history = f1_history, accuarcy_history = accuarcy_history))
}

# 主程序
start_time <- Sys.time()

num_clients <- 20
initial_alpha <- 1
batch_size <- 10
max_iter <- 300
file_path <- 'payment_fraud_dataset.csv'
data <- load_and_preprocess_data(file_path)
X <- data$features
y <- data$labels
true_covariance <- cal_truevariance(X)

true_w <- c(5.6783423e-02, 6.3214862e-01, 1.0839984e+01, -1.0987664e+01, 3.6733518e+00, -3.9949934e+00,
            2.7498127e+03, -1.4874005e-01, -2.5315860e-02, -1.6703321e-01, -3.4908421e-02)

initial_model <- rnorm(ncol(X), mean = 0, sd = 0.1)

results <- federated_sgd(X, y, true_w, num_clients, initial_model, true_covariance, max_iter)

end_time <- Sys.time()
execution_time <- end_time - start_time
print(sprintf("程序运行时间: %.4f 秒", execution_time))

# 创建一个数据框来存储这些历史数据
df <- data.frame(
  L2_Norm_History = results$l2_norm_history,
  F1_Score = results$f1_history,
  Accuracy = results$accuarcy_history
)

# 将数据框写入Excel文件
write.xlsx(df, "FedSGD_LogR_payment.xlsx", row.names = FALSE)