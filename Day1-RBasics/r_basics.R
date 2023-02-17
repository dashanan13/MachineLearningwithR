#Variable assignment in R
assignment1 = "it can be done with a ="
assignment2 <- "it can also be done with a <-"

#Data structures in R
#=====================================================================
#---------------------------------------------------------------------
#Vectors: ORDERED set of similar type of elements, only text or only numbers (!!! ORDER BEGINS at 1 not 0!!!)
#Logical Vectors
logical_vector <- c(TRUE,FALSE,FALSE,TRUE,TRUE,FALSE,FALSE,TRUE)
logical_vector[2:5]
#Numeric Vectors
numeical_vector <- c(15, 86, 99.999, 69, 455, 86, NA, NULL)  # NA = missing value, NULL = absence of any value (nothing is or can be there)
numeical_vector[c(TRUE,TRUE,TRUE,FALSE,FALSE,TRUE,TRUE,TRUE)] #selecting value based on Boolean place value or index number, false will not render
#Integer Vectors
integral_vector <- c(15L, 86L, 99L, 15L, 86L, 8, NA, NULL)
#String Vectors
string_vectors <- c("mohit", "sumit", "darsh", "vishnu", "mohit", "sumit", "darsh", "vishnu")
string_vectors[-2] #neglecting 2nd value

#---------------------------------------------------------------------
#Factors: Optimized to store category labels
#c() function gives the vales and the levels argument gives the acceptable values
factors_bloodtypes <- factor(c("O", "AB", "A"), levels = c("A", "B", "AB", "O"))
factors_bloodtypes[1]
#---------------------------------------------------------------------
#Lists: ORDERED set of similar or dissimilar elements (!!! ORDER BEGINS at 1 not 0!!!)
list_example <- list(
  fullname="mohit sharma",
  temperature=98.7,
  age=35,
  gender="male",
  canDrive=TRUE
)

list_example$fullname
list_example$gender
#---------------------------------------------------------------------
#Matrix: 2 dimension table
matrix_row_col_specified <- matrix(c(6:30),nrow=5,ncol=5)
matrix_row_specified <- matrix(c('a', 'b', 'c', 'd', 'e', 'f'), nrow = 2)
matrix_col_specified <- matrix(c('a', 'b', 'c', 'd'), ncol = 2)

matrix_row_col_specified[2,3]
matrix_col_specified[1,] #first row
matrix_row_col_specified[,2] #second column
matrix_row_col_specified[1:2,2:3] #select sub matrix
#---------------------------------------------------------------------
#Array: 2 or more than 2 dimension layer   ***** 
array_example <- array(c(1:9), dim <- c(3,3,4,2)) # 3x3 matrix stored in a 4x2 matrix, each element of 4x2 matrix is a 3x3 matrix
array_example
#---------------------------------------------------------------------
#Data Frame: sort of a 2 dimensional array
# A data frame can be understood as a list of vectors or factors, each having exactly the same number of values
subject_name <- c("mohit", "dashanan", "vishnu", "raghav")
temperature <- c(98.6, 104, 96.1, 98.4)
flu_status <- c(TRUE, FALSE, FALSE, FALSE)
gender <- c("male", "male", "male", "male")
blood  <- factor(c("O+", "AB", "A", "B"), levels = c("A", "B", "AB", "O+"))

dataframe_example <- data.frame(subject_name, temperature, flu_status, gender, blood, stringsAsFactors = FALSE)

dataframe_example
dataframe_example$subject_name
dataframe_example[c("subject_name", "temperature")]  #calling out with column name
dataframe_example[c(1,3),c(2,4)] #calling via rows, column sequences
dataframe_example[,1] #calling all rows of column one
dataframe_example[1,] #calling all columns of row one
#---------------------------------------------------------------------
#=====================================================================
#Important Functions
#View data in a visual and searchable
View(dataframe_example)
#Get the class of a variable
class(string_vectors)
#Get working directory
getwd()
#Set working directory
setwd("./MachineLearningwithR/Day1")
#=====================================================================
#Operators
#---------------------------------------------------------------------

#=====================================================================
#Working with Data
#---------------------------------------------------------------------
#Save Data
save(dataframe_example, array_example, list_example, matrix_row_col_specified, file = "mydata.RData")

#Load Data
load("./MachineLearningwithR/Day1/mydata.RData")
#read csv
csv_data <- read.csv("usedcars.csv", stringsAsFactors = FALSE)
View(csv_data)
#save csv
write.csv(csv_data, file = "saved_csv_data.csv")
#---------------------------------------------------------------------
## saving, loading, and removing R data structures
#show all data structures in memory
ls()
#remove the named object
rm(assignment1, assignment2)
#remove all objects
rm(list=ls())
#---------------------------------------------------------------------
#Exploring and understanding data

##data exploration example using used car data
usedcars <- read.csv("usedcars.csv", stringsAsFactors = FALSE)
#get structure of used car data
str(usedcars)

##Exploring numeric variables -----

#summarize numeric variables
summary(usedcars$year)
summary(usedcars[c("price", "mileage")])

#calculate the mean income
(36000 + 44000 + 56000) / 3
mean(c(36000, 44000, 56000))

# the median income
median(c(36000, 44000, 56000))

# the min/max of used car prices
range(usedcars$price)

# the difference of the range
diff(range(usedcars$price))

# IQR for used car prices
IQR(usedcars$price)

# use quantile to calculate five-number summary
quantile(usedcars$price)

# the 99th percentile
quantile(usedcars$price, probs = c(0.01, 0.99))

# quintiles
quantile(usedcars$price, seq(from = 0, to = 1, by = 0.20))

# boxplot of used car prices and mileage
boxplot(usedcars$price, main="Boxplot of Used Car Prices",
        ylab="Price ($)")

boxplot(usedcars$mileage, main="Boxplot of Used Car Mileage",
        ylab="Odometer (mi.)")

# histograms of used car prices and mileage
hist(usedcars$price, main = "Histogram of Used Car Prices",
     xlab = "Price ($)")

hist(usedcars$mileage, main = "Histogram of Used Car Mileage",
     xlab = "Odometer (mi.)")

# variance and standard deviation of the used car data
var(usedcars$price)
sd(usedcars$price)
var(usedcars$mileage)
sd(usedcars$mileage)

## Exploring numeric variables -----

# one-way tables for the used car data
table(usedcars$year)
table(usedcars$model)
table(usedcars$color)

# compute table proportions
model_table <- table(usedcars$model)
prop.table(model_table)

# round the data
color_table <- table(usedcars$color)
color_pct <- prop.table(color_table) * 100
round(color_pct, digits = 1)

## Exploring relationships between variables -----

# scatterplot of price vs. mileage
plot(x = usedcars$mileage, y = usedcars$price,
     main = "Scatterplot of Price vs. Mileage",
     xlab = "Used Car Odometer (mi.)",
     ylab = "Used Car Price ($)")

# new variable indicating conservative colors
usedcars$conservative <-  usedcars$color %in% c("Black", "Gray", "Silver", "White")

# checking our variable
table(usedcars$conservative)

install.packages("gmodels")

# Crosstab of conservative by model
library(gmodels)
CrossTable(x = usedcars$model, y = usedcars$conservative)

