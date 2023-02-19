library(datasets)
data("iris")

library('C50')

summary(iris) 
boxplot(iris[-5], main = 'Boxplot of Iris data by attributes') 
pairs(iris[,-5], main="iris Data", pch=21, bg = c("black", "red", "blue")[unclass(iris$Classification)]) 


irisTree <- C5.0(iris[,-5], iris[,5]) 

summary(irisTree) # view the model components  

plot(irisTree, main = 'Iris decision tree') # view the model graphically


plot(irisTree, main = 'Iris decision tree')


# build a rules set  
irisRules <- C5.0(iris[,-5], iris[,5], rules = TRUE) 
summary(irisRules) # view the ruleset 

