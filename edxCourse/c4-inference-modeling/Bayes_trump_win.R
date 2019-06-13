# lets calculate the probability that trump will win in Florida
# using Bayesian models and such

library(dslabs)
library(dplyr)
data("polls_us_election_2016")

polls <- polls_us_election_2016 %>% filter(enddate >= "2016-11-04" & state=="Florida") %>%
  mutate(spread = rawpoll_clinton/100 - rawpoll_trump/100)

results <- polls %>% summarize(avg = mean(spread), se = sd(spread)/n())

# assume the spread follows N(mu, tau); then Y|spread ~ N(spread, sigma)
mu <- 0
taus <- seq(0.005, 0.05, len=100)
Y <- results$avg
sigma <- results$se

p_calc <- function(tau, sig=sigma) {
  # takes sigma, tau and calculates the probability of trump winning
  B <- sig^2 / (sig^2 + tau^2)
  E <- B*mu + (1-B)*Y
  se <- 1/sqrt(1/sig^2 + 1/tau^2)
  pnorm(0, mean=E, sd=se)
}

ps <- sapply(taus, p_calc)
plot(taus, ps)
# the prob of trump winning Florida decreases when the variability increases,
# i.e. when the variability of the spread increases
