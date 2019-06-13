library(ggplot2)
library(dplyr)

N <- 1:1000
x <- rnorm(1000)
d <- data.frame(n=N, vals=x)

d %>% ggplot(aes(x=n, y=vals)) + geom_point()