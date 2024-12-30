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

local_sgd <- function(X, y, local_model, batch_size, alpha) {
  indices <- sample(1:nrow(X), batch_size, replace = FALSE)
  X_batch <- X[indices, ]
  y_batch <- y[indices]
  res <- logistic_loss_and_gradient(local_model, X_batch, y_batch)
  local_model <- local_model - alpha * res$gradient
  return(list(model = local_model, gradient = res$gradient, loss = res$loss))
}

server_aggregate <- function(local_models) {
  return(rowMeans(do.call(rbind, local_models)))
}

federated_sgd <- function(X, y, true_w, num_clients, initial_model, true_covariance, max_iter = 500, tol = 1e-6,
                          batch_size = 10, initial_alpha = 0.9, decay_rate = 0.99) {
  N <- nrow(X)
  p <- ncol(X)
  alpha <- initial_alpha
  local_models <- replicate(num_clients, initial_model, simplify = FALSE)
  global_model <- rowMeans(do.call(rbind, local_models))
  l2_norm_history <- c()
  
  data_splits <- split(X, rep(1:num_clients, each = N / num_clients))
  y_splits <- split(y, rep(1:num_clients, each = N / num_clients))
  
  for (iteration in 1:max_iter) {
    results <- mclapply(1:num_clients, function(m) {
      local_result <- local_sgd(data_splits[[m]], y_splits[[m]], local_models[[m]], batch_size, alpha)
      local_models[[m]] <- local_result$model
      return(local_result)
    }, mc.cores = num_clients)
    
    alpha <- alpha * decay_rate
    global_model <- server_aggregate(lapply(results, function(res) res$model))
    
    current_loss <- mean(sapply(results, function(res) res$loss))
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
true_w <- runif(p, -0.5, 0.5)
data <- generate_logistic_data(N, p, true_w)
X <- data$X
y <- data$y
initial_model <- rnorm(p, mean = 0, sd = 0.1)

l2_norm_history <- federated_sgd(X, y, true_w, num_clients, initial_model, data$true_covariance_matrix)