## Example: Identifying Poisonous Mushrooms ----
## Step 2: Exploring and preparing the data ---- 

mushrooms <- read.csv("~/Desktop/mushrooms.csv", stringsAsFactors = TRUE)

# examine the structure of the data frame
str(mushrooms)

# drop the veil_type feature
mushrooms$veil_type <- NULL

# examine the class distribution
table(mushrooms$type)


library(C50)

# check that the dataset is reordened
head( mushrooms[1], 16 )

# create dataset to test and training
X <- mushrooms[,2:22]
y <- mushrooms[,1]

# calculate split 1/3 to test. This number is where the dataset must split the values
split <- length(mushrooms$type) - round(length(mushrooms$type)/3) 

trainInputs <- X[1:split,]
trainOutput <- y[1:split]
testInputs <- X[ (split + 1):length(mushrooms$type),]
testOutput <- y[(split + 1):length(mushrooms$type)]


#apply C5.0
mushrooms_model <- C5.0(trainInputs, trainOutput)
mushrooms_model
summary(mushrooms_model)

mushrooms_pred <- predict(mushrooms_model, testInputs)
#accuracy of the model
sum(mushrooms_pred == testOutput ) / length(mushrooms_pred)

type<- mushrooms$type

library(gmodels)
CrossTable(mushrooms_pred, type,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE)

summary(mushrooms_pred)
summary(mushrooms_model)
