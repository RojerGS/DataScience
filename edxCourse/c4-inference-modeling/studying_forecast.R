library(dslabs)
library(dplyr)
data("polls_us_election_2016")

polls <- polls_us_election_2016 %>%
          filter(state != "U.S." & enddate >= "2016-10-31") %>% 
          mutate(spread = rawpoll_clinton/100 - rawpoll_trump/100)

# build 95% CI for the spread
cis <- polls %>%
  mutate(X_hat = (spread+1)/2, se = 2*sqrt(X_hat*(1-X_hat)/samplesize),
          lower = spread - qnorm(0.975)*se, upper = spread + qnorm(0.975)*se) %>%
  select(state, startdate, enddate, pollster, grade, spread, lower, upper)

# add the real spread
add <- results_us_election_2016 %>% mutate(actual_spread = clinton/100 - trump/100) %>% select(state, actual_spread)
cis <- cis %>% mutate(state = as.character(state)) %>% left_join(add, by = "state")

# find the proportion of intervals that contain the spread
p_hits <- cis %>% mutate(hit = lower <= actual_spread & actual_spread <= upper) %>%
  summarize(proportion_hits = sum(hit)/n())

# now we stratify by pollster
p_hits <- cis %>% mutate(hit = lower <= actual_spread & actual_spread <= upper) %>%
  group_by(pollster) %>%
  filter(n() >= 5) %>%
  summarize(proportion_hits = mean(hit), n = n(), grade = grade[1]) %>%
  arrange(desc(proportion_hits))

# now by state and with a plot
library(ggplot2)
p_hits <- cis %>% mutate(hit = lower <= actual_spread & actual_spread <= upper) %>%
  group_by(state) %>%
  filter(n() >= 5) %>%
  summarize(proportion_hits = mean(hit), n = n()) %>%
  arrange(desc(proportion_hits))
p_hits %>% ggplot(aes(x=state, y=proportion_hits)) + geom_bar(stat="identity") + coord_flip()

# lets see who predicted this well
errors <- cis %>% mutate(error = spread-actual_spread, hit = spread*actual_spread > 0)
p_hits <- errors %>% group_by(state) %>%
  filter(n() >= 5) %>%
  summarize(proportion_hits = mean(hit), n = n()) %>%
  arrange(desc(proportion_hits))

p_hits %>% ggplot(aes(x=state, y=proportion_hits)) + geom_bar(stat="identity") + coord_flip()

hist(errors$error)
median(errors$error)
# this median represents the (random) bias that occurs every election
# is the bias state-independent?
errors %>% filter(grade %in% c("A+", "A", "A-", "B+")) %>% group_by(state) %>%
  filter(n() >= 5) %>% arrange(error) %>%
  ggplot(aes(x=state, y=error)) + geom_point() + geom_boxplot()