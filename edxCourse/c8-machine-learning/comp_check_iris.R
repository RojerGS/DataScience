library(caret)
library(tidyverse)
data(iris)
iris <- iris[-which(iris$Species=='setosa'),]
y <- iris$Species

set.seed(2)
test_index <- createDataPartition(y, times = 1, p = 0.5, list=FALSE)
test <- iris[test_index,]
train <- iris[-test_index,]

# iterate over all features to find best cutoff for each feature
# in the training data
accs <- map_dbl(c(1,2,3,4), function(idx) {
          name <- names(train)[idx]
          lb <- min(train[idx])
          ub <- max(train[idx])
          sequence <- seq(from=lb, to=ub, by=0.1)
          # iterate over all possible cutoffs
          cutoff_accs <- map_dbl(sequence, function(c) {
                predictions = ifelse(train[idx] >= c, "virginica", "versicolor") %>% factor(levels=levels(train$Species))
                mean(predictions == train$Species)
          })
          max_acc <- max(cutoff_accs)
          max_cutoff <- sequence[which.max(cutoff_accs)]
          print(name)
          print(max_cutoff)
          max_acc
})

# how does this fare in the test data?
predictions <- ifelse(test$Petal.Length >= 4.8, "virginica", "versicolor") %>% factor(levels=levels(test$Species))
mean(predictions == test$Species)

# repeat search, but for test data
accs <- map_dbl(c(1,2,3,4), function(idx) {
  name <- names(train)[idx]
  lb <- min(test[idx])
  ub <- max(test[idx])
  sequence <- seq(from=lb, to=ub, by=0.1)
  # iterate over all possible cutoffs
  cutoff_accs <- map_dbl(sequence, function(c) {
    predictions = ifelse(test[idx] >= c, "virginica", "versicolor") %>% factor(levels=levels(test$Species))
    mean(predictions == train$Species)
  })
  max_acc <- max(cutoff_accs)
  max_cutoff <- sequence[which.max(cutoff_accs)]
  print(name)
  print(max_cutoff)
  max_acc
})

plot(iris,pch=21,bg=iris$Species)

# optimized petal length cutoff to 4.8 and petal width cutoff to 1.6 separately
# create compound rule
length_cutoff <- 4.8
width_cutoff <- 1.6
predictions <- ifelse((test$Petal.Length >= length_cutoff) | (test$Petal.Width >= width_cutoff),
                      "virginica",
                      "versicolor") %>% factor(levels=levels(test$Species))
mean(predictions == test$Species)