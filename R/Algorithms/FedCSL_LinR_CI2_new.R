# 导入必要的库
library(MASS)
library(dplyr)
library(magrittr)
library(parallel)
library(openxlsx)

# 数据生成函数
generate_data <- function(N, p, true_w, noise = 1) {
    mean <- rep(0, p)
    covariance <- matrix(0.2, nrow = p, ncol = p)
    diag(covariance) <- 1
    X <- mvrnorm(N, mu = mean, Sigma = covariance)
    ei <- rnorm(N, mean = 0, sd = noise)
    y <- X %*% true_w + ei
    
    X_covariance <- t(X) %*% X
    X_covariance_inv <- solve(X_covariance)
    true_covariance_matrix <- noise^2 * X_covariance_inv
    return(list(X = X, y = y, true_covariance_matrix = true_covariance_matrix))
}

# 损失函数
loss_function <- function(w, X, y) {
    residuals <- X %*% w - y
    return(0.5 * sum(residuals^2) / length(y))
}

# 损失和梯度计算
loss_and_gradient <- function(w, X, y) {
    residuals <- X %*% w - y
    loss <- 0.5 * mean(residuals^2)
    gradient <- t(X) %*% residuals / length(y)
    return(list(loss = loss, gradient = gradient))
}

# 计算 SRR
calculate_srr <- function(w, Sigma, Sigma_true) {
    numerator <- t(w) %*% Sigma %*% w
    denominator <- t(w) %*% Sigma_true %*% w
    return(sqrt(numerator / denominator))
}

# 代理损失和梯度计算
surrogate_loss_and_gradient <- function(theta, local_grad, X, y, global_grad_t) {
    residuals <- X %*% theta - y
    loss_f_k <- 0.5 * mean(residuals^2)
    grad_f_k <- t(X) %*% residuals / length(y)
    
    inner_product <- (local_grad - global_grad_t) %*% theta
    surrogate_loss <- loss_f_k - inner_product
    surrogate_gradient <- grad_f_k - (local_grad - global_grad_t)
    
    return(list(surrogate_loss = surrogate_loss, surrogate_gradient = surrogate_gradient))
}

# 客户端训练函数
client_training <- function(X_m, y_m, local_model, global_model, global_grad_t, alpha, num_local_steps) {
    for (i in 1:num_local_steps) {
        indices <- sample(1:nrow(X_m), 10, replace = FALSE)
        X_batch <- X_m[indices, ]
        y_batch <- y_m[indices]
        surrogate_results <- surrogate_loss_and_gradient(local_model, global_grad_t, X_batch, y_batch, global_grad_t)
        local_model <- local_model - alpha * surrogate_results$surrogate_gradient
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
    
    local_models <- replicate(num_clients, initial_model, simplify = FALSE)
    global_model <- rowMeans(do.call(rbind, local_models))
    
    l2_norm_history <- c()
    ail_history <- c()
    cp_history <- c()
    cp01_history <- rep(0, max_iter)
    loss_history <- c()
    z_975 <- qnorm(0.975)
    coverage_count <- 0
    
    client_data_size <- N %/% num_clients
    data_splits <- lapply(0:(num_clients - 1), function(i) {
        list(X = X[(i * client_data_size + 1):((i + 1) * client_data_size), ],
             y = y[(i * client_data_size + 1):((i + 1) * client_data_size)])
    })
    
    w <- rep(1, p) / sqrt(p)
    
    for (iteration in 1:max_iter) {
        local_grads <- vector("list", num_clients)
        local_losses <- numeric(num_clients)
        
        for (m in 1:num_clients) {
            loss_grad <- loss_and_gradient(local_models[[m]], data_splits[[m]]$X, data_splits[[m]]$y)
            local_losses[m] <- loss_grad$loss
            local_grads[[m]] <- loss_grad$gradient
        }
        
        global_grad <- Reduce("+", local_grads) / num_clients
        
        updated_models <- vector("list", num_clients)
        for (m in 1:num_clients) {
            updated_models[[m]] <- client_training(data_splits[[m]]$X, data_splits[[m]]$y, local_models[[m]], global_model, global_grad, alpha, num_local_steps)
        }
        
        local_models <- updated_models
        global_model <- server_aggregate(local_models)
        
        current_loss <- mean(local_losses)
        l2_norm <- sqrt(sum((global_model - true_w)^2))
        l2_norm_history <- c(l2_norm_history, l2_norm)
        loss_history <- c(loss_history, current_loss)
        
        g_outer_sum <- matrix(0, nrow = p, ncol = p)
        for (gi in local_grads) {
            g_outer_sum <- g_outer_sum + (gi %*% t(gi))
        }
        
        g_outer_avg <- g_outer_sum / num_clients
        aveT_H_global_inv <- solve(matrix(1, nrow = p, ncol = p))  # 假设 H 为单位矩阵
        est_var <- aveT_H_global_inv %*% g_outer_avg %*% aveT_H_global_inv
        
        number_samples <- num_clients * batch_size
        ci_length <- z_975 * sqrt((t(w) %*% est_var %*% w) / number_samples)
        ci_lower <- t(w) %*% global_model - ci_length
        ci_upper <- t(w) %*% global_model + ci_length
        
        if ((t(w) %*% true_w >= ci_lower) && (t(w) %*% true_w <= ci_upper)) {
            coverage_count <- coverage_count + 1
            cp01_history[iteration] <- 1
        }
        
        ail <- ci_length
        ail_history <- c(ail_history, ail)
        coverage_rate <- coverage_count / iteration
        cp_history <- c(cp_history, coverage_rate)
        
        cat(sprintf("FedCSL %d: Loss = %.4f, L2 Norm = %.4f, CP = %.4f, final_CP = %.1f\n",
                    iteration, current_loss, l2_norm, coverage_rate, cp01_history[iteration]))

        if (l2_norm < tol) {
            cat(sprintf("模型收敛于第 %d 次全局更新\n", iteration))
            break
        }
    }
    
    return(list(loss_history = loss_history, l2_norm_history = l2_norm_history, ail_history = ail_history, cp_history = cp_history))
}

# 主函数
start_time <- Sys.time()

N <- 200000
p <- 10
num_clients <- 20
initial_alpha <- 1
batch_size <- 10
max_iter <- 300
true_w <- runif(p, -0.5, 0.5)
data <- generate_data(N, p, true_w)
X <- data$X
y <- data$y
true_covariance <- data$true_covariance_matrix
initial_model <- rnorm(p, 0, 0.1)

results <- federated_CSL(X, y, true_w, num_clients, initial_model, true_covariance, max_iter)

# 计算指标
last_loss <- tail(results$loss_history, 1)
min_loss <- min(results$loss_history)
avg_loss <- mean(results$loss_history)

last_l2_norm <- tail(results$l2_norm_history, 1)
min_l2_norm <- min(results$l2_norm_history)
avg_l2_norm <- mean(results$l2_norm_history)

last_ail <- tail(results$ail_history, 1)
min_ail <- min(results$ail_history)
avg_ail <- mean(results$ail_history)

last_cp <- tail(results$cp_history, 1)
min_cp <- min(results$cp_history)
avg_cp <- mean(results$cp_history)

cat(sprintf("最后迭代的 loss: %.4f\n", last_loss))
cat(sprintf("所有迭代中的最小 loss: %.4f\n", min_loss))
cat(sprintf("所有迭代的平均 loss: %.4f\n", avg_loss))

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