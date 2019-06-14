set.seed(1)
n <- 100
Sigma <- 9*matrix(c(1.0, 0.5, 0.5, 1.0), 2, 2)
dat <- MASS::mvrnorm(n = 100, c(69, 69), Sigma) %>%
  data.frame() %>% setNames(c("x", "y"))

set.seed(1)
rmses <- replicate(100, {
              test_index <- createDataPartition(dat$y, times = 1, p = 0.5, list = FALSE)
              train_set <- dat %>% slice(-test_index)
              test_set <- dat %>% slice(test_index)
              fit <- lm(y ~ x, data = train_set)
              y_hat <- predict(fit, test_set)
              sqrt(mean((y_hat - test_set$y)^2))
})

# larger data sets
build_data_set <- function(n) {
    Sigma <- 9*matrix(c(1.0, 0.5, 0.5, 1.0), 2, 2)
    dat <- MASS::mvrnorm(n = n, c(69, 69), Sigma) %>%
      data.frame() %>% setNames(c("x", "y"))
    return (dat)
}

apply_linear_models <- function(n) {
  dat <- build_data_set(n)
  rmses <- replicate(100, {
    test_index <- createDataPartition(dat$y, times = 1, p = 0.5, list = FALSE)
    train_set <- dat %>% slice(-test_index)
    test_set <- dat %>% slice(test_index)
    fit <- lm(y ~ x, data = train_set)
    y_hat <- predict(fit, test_set)
    sqrt(mean((y_hat - test_set$y)^2))
  })
  print(n)
  print(mean(rmses))
  print(sd(rmses))
}

set.seed(1)
Ns <- c(100, 500, 1000, 5000, 10000)
sapply(Ns, apply_linear_models)


# repeat Q1 but with larger correlation between X and Y
set.seed(1)
n <- 100
Sigma <- 9*matrix(c(1.0, 0.95, 0.95, 1.0), 2, 2)
dat <- MASS::mvrnorm(n = 100, c(69, 69), Sigma) %>%
  data.frame() %>% setNames(c("x", "y"))

set.seed(1)
rmses <- replicate(100, {
  test_index <- createDataPartition(dat$y, times = 1, p = 0.5, list = FALSE)
  train_set <- dat %>% slice(-test_index)
  test_set <- dat %>% slice(test_index)
  fit <- lm(y ~ x, data = train_set)
  y_hat <- predict(fit, test_set)
  sqrt(mean((y_hat - test_set$y)^2))
})


#Q6 and Q7
set.seed(1)
Sigma <- matrix(c(1.0, 0.75, 0.75, 0.75, 1.0, 0.25, 0.75, 0.25, 1.0), 3, 3)
dat <- MASS::mvrnorm(n = 100, c(0, 0, 0), Sigma) %>%
  data.frame() %>% setNames(c("y", "x1", "x2"))

set.seed(1)
test_index <- createDataPartition(dat$y, times = 1, p = 0.5, list = FALSE)
train_data <- dat %>% slice(-test_index)
test_data <- dat %>% slice(test_index)

x1model <- lm(y ~ x1, data = train_data)
x1preds <- predict(x1model, test_data)
x1rmse <- sqrt(mean((x1preds - test_data$y)^2))
x2model <- lm(y ~ x2, data = train_data)
x2preds <- predict(x2model, test_data)
x2rmse <- sqrt(mean((x2preds - test_data$y)^2))
x1x2model <- lm(y ~ x1 + x2, data = train_data)
x1x2preds <- predict(x1x2model, test_data)
x1x2rmse <- sqrt(mean((x1x2preds - test_data$y)^2))

# Q8
set.seed(1)
Sigma <- matrix(c(1.0, 0.75, 0.75, 0.75, 1.0, 0.95, 0.75, 0.95, 1.0), 3, 3)
dat <- MASS::mvrnorm(n = 100, c(0, 0, 0), Sigma) %>%
  data.frame() %>% setNames(c("y", "x1", "x2"))

set.seed(1)
test_index <- createDataPartition(dat$y, times = 1, p = 0.5, list = FALSE)
train_data <- dat %>% slice(-test_index)
test_data <- dat %>% slice(test_index)

x1model <- lm(y ~ x1, data = train_data)
x1preds <- predict(x1model, test_data)
x1rmse <- sqrt(mean((x1preds - test_data$y)^2))
x2model <- lm(y ~ x2, data = train_data)
x2preds <- predict(x2model, test_data)
x2rmse <- sqrt(mean((x2preds - test_data$y)^2))
x1x2model <- lm(y ~ x1 + x2, data = train_data)
x1x2preds <- predict(x1x2model, test_data)
x1x2rmse <- sqrt(mean((x1x2preds - test_data$y)^2))