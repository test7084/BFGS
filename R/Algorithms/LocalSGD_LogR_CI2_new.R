library(MASS)
library(parallel)
library(ggplot2)

generate_logistic_data <- function(N, p, true_w, noise = 1) {
  mean <- rep(0, p)
  covariance <- matrix(0.2, nrow = p, ncol = p)
  diag(covariance) <- 1
  X <- mvrnorm(n = N, mu = mean, Sigma = covariance)
  z <- X %*% true_w + rnorm(N, sd = noise)
  probability <- 1 / (1 + exp(-z))
  y <- rbinom(N, 1, probability)
  
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
  loss <- -mean(y * safe_log(predictions) + (1 - y) * safe_log(1 - predictions))
  gradient <- t(X) %*% (predictions - y) / length(y)
  return(list(loss = loss, gradient = gradient))
}

local_sgd <- function(X, y, true_w, num_clients, initial_model, max_iter = 500, tol = 1e-6, 
                      batch_size = 10, initial_alpha = 0.9, decay_rate = 0.99) {
  N <- nrow(X)
  p <- ncol(X)
  alpha <- initial_alpha
  local_models <- replicate(num_clients, initial_model, simplify = FALSE)
  global_model <- colMeans(do.call(rbind, local_models))
  
  l2_norm_history <- c()
  
  machine_data_size <- N %/% num_clients
  data_splits <- split(X, rep(1:num_clients, each = machine_data_size))
  y_splits <- split(y, rep(1:num_clients, each = machine_data_size))
  
  for (iteration in 1:max_iter) {
    results <- mclapply(1:num_clients, function(m) {
      X_batch <- data_splits[[m]][(1:batch_size), , drop = FALSE]
      y_batch <- y_splits[[m]][1:batch_size]
      return(logistic_loss_and_gradient(local_models[[m]], X_batch, y_batch))
    }, mc.cores = num_clients)
    
    local_losses <- sapply(results, function(res) res$loss)
    gradients <- lapply(results, function(res) res$gradient)
    
    for (m in 1:num_clients) {
      local_models[[m]] <- local_models[[m]] - alpha * gradients[[m]]
    }
    
    alpha <- alpha * decay_rate
    global_model <- colMeans(do.call(rbind, local_models))
    current_loss <- mean(local_losses)
    l2_norm <- sqrt(sum((global_model - true_w)^2))
    l2_norm_history <- c(l2_norm_history, l2_norm)
    
    cat(sprintf("Iteration %d: Loss = %.4f, L2 Norm = %.4f\n", iteration, current_loss, l2_norm))
    
    if (l2_norm < tol) {
      cat(sprintf("Converged at iteration %d\n", iteration))
      break
    }
  }
  
  return(l2_norm_history)
}

# Main execution
set.seed(123)
N <- 200000
p <- 100
num_clients <- 20
initial_alpha <- 0.5
batch_size <- 10
max_iter <- 300
true_w <- runif(p, -0.5, 0.5)

data <- generate_logistic_data(N, p, true_w)
X <- data$X
y <- data$y
initial_model <- rnorm(p, mean = 0, sd = 0.1)

l2_norm_history <- local_sgd(X, y, true_w, num_clients, initial_model, max_iter)

# 输出结果
df <- data.frame(
  Iteration = 1:length(l2_norm_history),
  L2_Norm = l2_norm_history
)

write.csv(df, "LocalSGD_LogR_CI_N200000k20p100.csv", row.names = FALSE)

# 计算指标
last_l2_norm <- tail(l2_norm_history, 1)
min_l2_norm <- min(l2_norm_history)
avg_l2_norm <- mean(l2_norm_history)

cat(sprintf("最后迭代的 L2 Norm: %.4f\n", last_l2_norm))
cat(sprintf("所有迭代中的最小 L2 Norm: %.4f\n", min_l2_norm))
cat(sprintf("所有迭代的平均 L2 Norm: %.4f\n", avg_l2_norm))