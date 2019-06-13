library(dslabs)

sum(is.na(na_example))
na_example <- ifelse(is.na(na_example), 0, na_example)
sum(is.na(na_example))

any(na_example > 5)
all(na_example >= 0)