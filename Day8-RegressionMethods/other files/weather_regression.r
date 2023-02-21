
# Clean the environment and image, if applicable
dev.off()
rm(list = ls())

# Before starting, download the weather dataset and store as weather1.txt: 
# https://archive.ics.uci.edu/ml/datasets/SML2010  (NEW_DATA.zip)

# un-comment command below if Desktop is your working directory
# setwd("~/Desktop")

#Import csv
weather <- read.table("~/Desktop/Weather1.txt", stringsAsFactors = FALSE)
View(weather)

#  ***** Linear Regression ***** 
# estimate beta manually
b <- cov(weather$V5, weather$V24) / var(weather$V5)
b

# estimate alpha manually
a <- mean(weather$V24) - b * mean(weather$V5)
a

# calculate the correlation of launch data
r <- cov(weather$V5, weather$V24) /
  (sd(weather$V5) * sd(weather$V24))
r
cor(weather$V5, weather$V24)

# computing the slope using correlation
r * (sd(weather$V24) / sd(weather$V5))

# confirming the regression line using the lm function (not in text)
model <- lm(V24 ~ V5, data = weather)
model
summary(model)

#  ***** Multilinear Regression ***** 
# creating a simple multiple regression function
reg <- function(y, x) {
  x <- as.matrix(x)
  x <- cbind(Intercept = 1, x)
  b <- solve(t(x) %*% x) %*% t(x) %*% y
  colnames(b) <- "estimate"
  print(b)
}

# examine the launch data
str(weather)

# test regression model with simple linear regression
reg(y = weather$V24, x = weather[24])

# use regression model with multiple regression
reg(y = weather$V24, x = weather[5:8])

# confirming the multiple regression result using the lm function (not in text)
model <- lm(V24 ~ V5 + V4 + V12, data = weather)
model


# ***** Regression tree and Model tree  *****
# reload data if necessary, here we have done it above already
# weather <- read.table("~/Desktop/Weather1.txt", stringsAsFactors = FALSE)
View(weather)
# examine the weather data
str(weather)
weather$V24 <- as.integer(weather$V24)
# the distribution of quality ratings
hist(weather$V24)

# remove columns with characters
weather$V1 <- NULL
weather$V2 <- NULL
weather$V12<- NULL
weather$V19 <- NULL
weather$V18 <- NULL
weather$V20 <- NULL
weather$V21 <- NULL

str(weather)
#summary
summary(weather)

length(weather$V24)

weather_train <- weather[1:1098, ]
weather_test <- weather[1099:1373, ]

##  Training a model on the data ----
# regression tree using rpart
library(rpart)
m.rpart <- rpart(V24 ~ ., data = weather_train)

# get basic information about the tree
m.rpart

# get more detailed information about the tree
summary(m.rpart)

# use the rpart.plot package to create a visualization
# install.packages("rpart.plot")
library(rpart.plot)

# a basic decision tree diagram
rpart.plot(m.rpart, digits = 4)

# a few adjustments to the diagram
rpart.plot(m.rpart, digits = 5, fallen.leaves = TRUE, type = 4, extra = 101)

##  Evaluate model performance ----

# generate predictions for the testing dataset
p.rpart <- predict(m.rpart, weather_test)

# compare the distribution of predicted values vs. actual values
summary(p.rpart)
summary(weather_test$V24)

# compare the correlation
cor(p.rpart, weather_test$V24)

# function to calculate the mean absolute error
MAE <- function(actual, predicted) {
  mean(abs(actual - predicted))  
}

# mean absolute error between predicted and actual values
MAE(p.rpart, weather_test$V24)

# mean absolute error between actual values and mean value
mean(weather_train$V24) # The result is 4.28
MAE(4.28, weather_test$V24)


## **** Improving model performance - Model Tree ****

# train a Cubist Model Tree
library(lattice)
library(Cubist)

m.cubist <- cubist(x = weather_train[-17], y = weather_train$V24)

# display basic information about the model tree
m.cubist

# display the tree itself
summary(m.cubist)

# generate predictions for the model
p.cubist <- predict(m.cubist, weather_test)

# summary statistics about the predictions
summary(p.cubist)

# correlation between the predicted and true values
cor(p.cubist, weather_test$V24)

# mean absolute error of predicted and true values
# (uses a custom function defined above)
MAE(weather_test$V24, p.cubist)

