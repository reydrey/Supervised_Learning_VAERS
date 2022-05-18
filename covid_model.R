# loading packages 
library(tidyverse)
library(janitor)
library(Hmisc)
library(outliers)
library(FactoMineR)
library(factoextra)
library(corrplot)
library(fastDummies)
library(tidytext)
library(dplyr)
library(caret)
library(rpart)
library(rpart.plot)
library(MLmetrics)
library(e1071)
library(ROCR)

#import data 
df <- read.csv('2021VAERSDATA_CLEANED.csv')
set.seed(123)
df$AE <- as.factor(df$AE)
# first classification method - NAIVE BAYES 
# splitting into training and testing sets
inTrain <- createDataPartition(y = df$AE, p = 0.8, list = FALSE)
# assigning a training set 
training <- df[inTrain,]
# assigning a testing set (any rows selected for training removed from testing set)
testing <- df[-inTrain,]

fitControl <- trainControl(method = "cv", number = 3)
tuneControl <- data.frame(fL=1, usekernel = TRUE, adjust=1)
bayes <- train(AE ~ ., data = training, method = "nb", trControl=fitControl, tuneGrid=tuneControl)
#confusion matrix for training data
pred <- predict(bayes, newdata = training)
confusionMatrix(pred, training$AE)
#confusion matrix for testing data
pred <- predict(bayes, newdata = testing)
confusionMatrix(pred, training$AE)
confusionMatrix(pred, testing$AE)

varImp(bayes)





