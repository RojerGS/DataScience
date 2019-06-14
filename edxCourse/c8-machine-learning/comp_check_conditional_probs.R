p_disease <- 0.02
p_healthy <- 1 - p_disease
p_pos_disease <- 0.85
p_neg_healthy <- 0.9
p_pos_healthy <- 1 - p_neg_healthy

p_pos <- p_pos_disease*p_disease + p_pos_healthy*p_healthy

p_disease_pos <- p_pos_disease*(p_disease/p_pos)

# sample this population with R
set.seed(1)
disease <- sample(c(0,1), size=1e6, replace=TRUE, prob=c(0.98,0.02))
test <- rep(NA, 1e6)
test[disease==0] <- sample(c(0,1), size=sum(disease==0), replace=TRUE, prob=c(0.90,0.10))
test[disease==1] <- sample(c(0,1), size=sum(disease==1), replace=TRUE, prob=c(0.15, 0.85))

# probability of positive test is
mean(test)
# probability of having disease | test is negative
neg_test <- which(test == 0)
mean(disease[neg_test])
# probability of having disease | test is positive
pos_test <- which(test == 1)
mean(disease[pos_test])
# if the test is positive, what is the relative risk of having the disease?
# the hint says:
# First calculate the probability of having the disease given a positive test, then normalize it against the disease prevalence.
prevalence = mean(disease)
prob_disease_positive_test <- mean(disease[which(test == 1)])
prob_disease_positive_test/prevalence

library(dslabs)
data("heights")
heights %>%
  mutate(height = round(height)) %>%
  group_by(height) %>%
  summarize(p = mean(sex == "Male")) %>%
  qplot(height, p, data = .)

# instead of plotting for every inch, plot for the 10% quantiles
ps <- seq(from=0, to=1, by=0.1)
heights %>%
    mutate(g = cut(height, quantile(height, ps), include.lowest = TRUE)) %>%
    group_by(g) %>%
    summarize(p = mean(sex == "Male"), height = mean(height)) %>%
    qplot(height, p, data = .)

library(MASS)
Sigma <- 9*matrix(c(1,0.5,0.5,1), 2, 2)
dat <- MASS::mvrnorm(n = 10000, c(69, 69), Sigma) %>%
  data.frame() %>% setNames(c("x", "y"))
plot(dat)
ps <- seq(from=0, to=1, by=0.1)
dat %>%
  mutate(g = cut(x, quantile(x, ps), include.lowest = TRUE)) %>%
  group_by(g) %>%
  summarize(y = mean(y), x = mean(x)) %>%
  qplot(x, y, data =.)