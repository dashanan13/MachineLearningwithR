#Apply knn to the iris dataset to classify to the three types of iris – setosa, versicolor, virginica. 
#How does the accuracy compare with that of knn?
setwd("/home/devops/MachineLearningwithR/Day2")

library(datasets)
head(iris)

iris_data <- iris
heart_data <- read.csv("heart.csv")

#randomize the data set
#set seed for predictability
set.seed(50)
iris_data_rand = iris_data[order(runif(nrow(iris_data))),]
iris_data_rand

str(iris_data_rand) #structure of data set
table(iris_data_rand$Species) #count of each species
summary(iris_data_rand[c(1:4)])   #check min , max etc for each numerical feature

#Normalize functions
normalize1 <- function(x) {
  return((x - min(x))/(max(x) - min(x)))
}
#z-score standardize
normalize2 <- function(x) {
  return((x - mean(x))/(sd(x)))
}

iris_data_normalized <- as.data.frame(lapply(iris_data_rand[,c(1,2,3,4)], normalize2))
summary(iris_data_normalized)

(nrow(iris_data_rand)) #number of rows

# KNN Modelling
iris_data_rand_train <- iris_data_normalized[1:120, ] #train set
iris_data_rand_test <- iris_data_normalized[121:150, ] #test set

iris_train_target_lbel <- iris_data_rand[1:120, 5] #train labels
iris_test_target_label <- iris_data_rand[121:150, 5] #test labels

require(class)

knn_predection_m1 <- knn(train = iris_data_rand_train, test = iris_data_rand_test, cl = iris_train_target_lbel, k = 13)
length(knn_predection_m1)

table(iris_test_target_label, knn_predection_m1)
