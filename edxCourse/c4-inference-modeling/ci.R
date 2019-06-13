library(dslabs)
library(dplyr)
library(ggplot2)
data("polls_us_election_2016")

polls <- filter(polls_us_election_2016, enddate >= "2016-10-31" & state == "U.S.")
nrow(polls)

N <- polls$samplesize[1]
X_hat <- polls$rawpoll_clinton[1]/100
se_hat <- sqrt(X_hat*(1-X_hat)/N)

q <- qnorm(0.975)
# 95% confidence interval for the Clinton proportion
ci <- c(X_hat - se_hat*q, X_hat + se_hat*q)

pollster_results <- mutate(polls, X_hat = rawpoll_clinton/100,
                                  se_hat = sqrt(X_hat*(1-X_hat)/samplesize),
                                  lower = X_hat-se_hat*q,
                                  upper = X_hat+se_hat*q) %>%
                    select("pollster", "enddate", "X_hat", "se_hat", "lower", "upper")

hit <- pollster_results$lower <= 0.482 & 0.482 <= pollster_results$upper
avg_hit <- summarize(pollster_results, mean=mean(hit))

# now working with the spread
pollster_results <- mutate(polls, d_hat = (rawpoll_clinton-rawpoll_trump)/100,
                                  X_hat = (d_hat+1)/2,
                                  se_hat = 2*sqrt(X_hat*(1-X_hat))/samplesize,
                                  lower = d_hat-se_hat*q,
                                  upper = d_hat+se_hat*q) %>%
                    select("pollster", "enddate", "d_hat", "lower", "upper")

hit <- pollster_results$lower <= 0.021 & pollster_results$upper >= 0.021
avg_hit <- summarize(pollster_results, mean=mean(hit))

data <- mutate(pollster_results, errors = d_hat - 0.021)
data %>%
  group_by(pollster) %>%
  filter(n() >= 5) %>%
  ggplot(aes(x = pollster, y = errors)) +
  geom_point() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))