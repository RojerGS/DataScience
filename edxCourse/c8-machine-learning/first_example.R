library(dslabs)
data(heights)

library(caret)
library(tidyverse)

# outcomes and predictors
y <- heights$sex
x <- heights$height

# for reproducibility:
set.seed(2)
train_index <- createDataPartition(y, times = 1, p = 0.5, list=FALSE)

train_set <- heights[train_index, ]
test_set <- heights[-train_index, ]

# first algorithm, just guess the sex
# ensure the outcomes are factors
guessed <- sample(c("Male", "Female"), length(test_set[,1]), replace = TRUE) %>% factor(levels=levels(heights$sex))
# find the accuracy
guessed_accuracy <- sum(guessed == test_set$sex)/length(guessed)

# notice that we can do better than guessing; in fact, males are typically taller:
train_set %>% group_by(sex) %>% summarize(mean(height), sd(height))
# female has mean 65 and male has mean 69.3
# say it is male if it is above 67.15
predicted <- ifelse(test_set$height >= 67.15, "Male", "Female") %>% factor(levels=levels(heights$sex))
predicted_accuracy <- sum(predicted == test_set$sex)/length(predicted) # almost 75% accuracy

# find the best cutoff on the training data
cutoffs <- seq(60, 70, 0.2)
accs <- map_dbl(cutoffs, function(c) {
  pred <- ifelse(train_set$height >= c, "Male", "Female") %>% factor(levels=levels(heights$sex))
  mean(pred == train_set$sex)
})
plot(cutoffs, accs)
print("Best accuracy is ")
print(max(accs))
# store the best cutoff
best_cutoff <- cutoffs[which.max(accs)]
# check the final accuracy with this cutoff
accuracy <- mean(ifelse(test_set$height >= best_cutoff, "Male", "Female") %>% factor(levels=levels(heights$sex)) == test_set$sex)
print("Final accuracy is ")
print(accuracy)
