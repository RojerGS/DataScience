avg <- function(vec) {
  s <- sum(vec)
  n <- length(vec)
  s/n
}

library(dslabs)

avg(na_example)

identical(avg(na_example), mean(na_example))

avg <- function(vec, arithmetic=TRUE) {
  if (arithmetic) {
    sum(vec)/length(vec)
  } else {
    prod(vec)^(1/length(vec))
  }
}