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

wbcd_train <- wbcd_n[1:469, ]
wbcd_test <- wbcd_n[470:569, ]