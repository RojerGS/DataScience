library(dslabs)
library(tidyverse)

dat <- mnist$train$images

props <- sapply(dat, function(digit) {
  sum(as.vector(dat))
})