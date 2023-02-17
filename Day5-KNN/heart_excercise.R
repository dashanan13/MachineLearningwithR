#Apply knn to the heart dataset to predict heart problem.
#A description of the dataset is provided. From the data folder, use the processed Cleveland dataset.
#The last column #14 is the outcome – angiographic disease status: 0 (no disease) or 1, 2, 3, 4 (yes).
#Look for the optimum k.

#How does the accuracy compare with that of knn?
setwd("/home/devops/MachineLearningwithR/Day2")

heart_data <- read.csv("heart.csv")

(nrow(heart_data)) #number of rows
(ncol(heart_data)) #number of column
#randomize the data set
#set seed for predictability
set.seed(50)
heart_rand = heart_data[order(runif(nrow(heart_data))),]
heart_rand

str(heart_rand) #structure of data set
View(heart_rand)

#summary(heart_rand)   #check min , max etc for each numerical feature

#Normalize functions
normalize1 <- function(x) {
  return((x - min(x))/(max(x) - min(x)))
}
#z-score standardize
normalize2 <- function(x) {
  return((x - mean(x))/(sd(x)))
}

heart_rand_normalized <- as.data.frame(lapply(heart_rand[,c(1:13)], normalize2))
#summary(heart_rand_normalized)
(nrow(heart_data))*0.8 #number of rows

# KNN Modelling
heart_rand_norm_train <- heart_rand_normalized[1:236, ] #train set
heart_rand_norm_test <- heart_rand_normalized[237:296, ] #test set

heart_rand_train_target_label <- heart_rand[1:236, 14] #train labels
heart_rand_test_target_label <- heart_rand[237:296, 14] #test labels

#require(class)

knn_predection_m1 <- knn(train = heart_rand_norm_train, test = heart_rand_norm_test, cl = heart_rand_train_target_label, k = 15)
#length(knn_predection_m1)
table(heart_rand_test_target_label, knn_predection_m1)
# Create the cross tabulation of predicted vs. actual
#CrossTable(x = heart_rand_test_target_label, y = knn_predection_m1, prop.chisq = FALSE)   

knn_predection_m1 <- knn(train = heart_rand_norm_train, test = heart_rand_norm_test, cl = heart_rand_train_target_label, k = 17)
#length(knn_predection_m1)
table(heart_rand_test_target_label, knn_predection_m1)

knn_predection_m1 <- knn(train = heart_rand_norm_train, test = heart_rand_norm_test, cl = heart_rand_train_target_label, k = 19)
#length(knn_predection_m1)
table(heart_rand_test_target_label, knn_predection_m1)


