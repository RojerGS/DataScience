library(dslabs)
library(dplyr)
data("polls_us_election_2016")

polls <- polls_us_election_2016 %>% filter(pollster %in%
                                            c("The Times-Picayune/Lucid",
                                              "Rasmussen Reports/Pulse Opinion Research")
                                          ) %>% 
              mutate(spread = rawpoll_clinton/100 - rawpoll_trump/100)

res <- polls %>% group_by(pollster) %>% summarize(avg = mean(spread),
                                                  sd = sd(spread),
                                                  n = n())

estimate <- max(res$avg) - min(res$avg)
se_hat <- sqrt(res$sd[1]^2/res$n[1] + res$sd[2]^2/res$n[2])

# 0 is not in the CI so it does look like there is some "house effect"
ci <- c(estimate - se_hat*qnorm(0.975), estimate + se_hat*qnorm(0.975))