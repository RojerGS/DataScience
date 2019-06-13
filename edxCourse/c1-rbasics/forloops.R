sum_first_powers <- function(n, power=1) {
  sum <- 0
  for (i in 1:n) {
    sum <- sum + i^power
  }
  sum
}

closed_formula <- function(n) {
  n*(n+1)/2
}

m <- 25
vec <- c(length=m)
for (n in 1:m) {
  vec[n] <- sum_first_powers(n)
}

ns = 1:25
plot(ns, vec)
lines(ns, ns*(ns+1)/2)