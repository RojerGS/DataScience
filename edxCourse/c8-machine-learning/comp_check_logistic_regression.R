library(ggplot)
library(tidyverse)


make_data <- function(n = 1000, p = 0.5, 
                      mu_0 = 0, mu_1 = 2, 
                      sigma_0 = 1,  sigma_1 = 1){
  
      y <- rbinom(n, 1, p)
      f_0 <- rnorm(n, mu_0, sigma_0)
      f_1 <- rnorm(n, mu_1, sigma_1)
      x <- ifelse(y == 1, f_1, f_0)
      
      test_index <- createDataPartition(y, times = 1, p = 0.5, list = FALSE)
      
      list(train = data.frame(x = x, y = as.factor(y)) %>% slice(-test_index),
           test = data.frame(x = x, y = as.factor(y)) %>% slice(test_index))
}
dat <- make_data()

dat$train %>% ggplot(aes(x, color = y)) + geom_density()

set.seed(1)
mu_1 <- seq(0, 3, len=25)
res <- sapply(mu_1, function (mu1) {
    dat <- make_data(mu_1 = mu1)
    train_data <- dat$train
    test_data <- dat$test
    fit <- glm(y ~ x, data = train_data, family = "binomial")
    p_hat <- predict(fit, test_data, type = "response")
    y_hat <- ifelse(p_hat > 0.5, 1, 0)
    mean(y_hat == test_data$y)
})
qplot(mu_1,  res)
