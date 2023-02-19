
# Clean the environment and image
dev.off()
rm(list = ls())
#### Part 1: Decision Trees -------------------

## Understanding Decision Trees ----
# calculate entropy of a two-class segment
-0.60 * log2(0.60) - 0.40 * log2(0.40)

curve(-x * log2(x) - (1 - x) * log2(1 - x),
      col = "red", xlab = "x", ylab = "Entropy", lwd = 4)

## Example: Identifying Risky Bank Loans ----
## Step 2: Exploring and preparing the data ----
credit <- read.csv("~/Desktop/credit.csv", stringsAsFactors = FALSE)
str(credit)



library(dplyr)
credit$months_loan_duration<- NULL
credit$amount<-NULL
credit$percent_of_income<-NULL
credit$years_at_residence<-NULL
credit$age<-NULL
credit$existing_loans_count<-NULL
credit$dependents<-NULL
head(credit)
head(credit)
credit <- mutate_at(credit, vars(checking_balance, credit_history, purpose, savings_balance, employment_duration, other_credit, housing, job, phone, default), as.factor)
credit
str(credit)

table(credit$default)

## Step 3: Training a model on the data ----
# build the simplest decision tree
#install.packages("C50")
library(rJava)
library(OneR)


# train OneR() on the data
credit_1R <- OneR(default ~ ., data = credit)
length(credit$default)
length(credit$credit_history)
# display simple facts about the tree
credit_1R

credit_1R_pred <- predict(credit_1R, credit)
table(actual = credit$default, predicted = credit_1R_pred)
# Applying Ripper
library(RWeka)
credit_JRip <- JRip(default ~ ., data = credit)
credit_JRip
summary(credit_JRip)

# Clear plots
dev.off()  # But only if there IS a plot

# Clear console
cat("\014")  # ctrl+L


