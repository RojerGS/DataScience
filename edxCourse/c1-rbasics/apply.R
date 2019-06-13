sum_first <- function(n) {
  s <- 0
  for (i in 1:n) {
    s <- s + i
  }
  s
}

vec <- sapply(1:25, sum_first)

sum_first_powers <- function(n, power=1) {
  s <- 0
  for (i in 1:n) {
    s <- s + i^power
  }
  s
}

# pass optional arguments to the function sum_first_powers
vec2 <- sapply(1:25, sum_first_powers, 1)

identical(vec, vec2)