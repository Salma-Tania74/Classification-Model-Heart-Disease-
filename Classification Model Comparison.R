
data<-only_cats

head(data)

str(data)
is.na(data)

#random forest

data$Presenceofheartdisease<-as.factor(data$Presenceofheartdisease)
str(data)


data$Gender<-as.factor(data$Gender)
data$CurrentSmoking<-as.factor(data$CurrentSmoking)
data$agegroup<-as.factor(data$agegroup)
data$bmigroup<-as.factor(data$bmigroup)
data$sbplevel<-as.factor(data$sbplevel)
data$dbplevel<-as.factor(data$dbplevel)
data$hrgroup<-as.factor(data$hrgroup)
data$Hblevel<-as.factor(data$Hblevel)
data$platelets<-as.factor(data$platelets)
data$wbc<-as.factor(data$wbc)
data$rbcgrp<-as.factor(data$rbcgrp)
data$Neutrophils<-as.factor(data$Neutrophils)
data$Creatinine<-as.factor(data$Creatinine)
data$ldllevel<-as.factor(data$ldllevel)
data$hdllevel<-as.factor(data$hdllevel)
data$cholesterol<-as.factor(data$cholesterol)
data$bloodglucose<-as.factor(data$bloodglucose)
data$sodium<- as.factor(data$sodium)
data$potassium<-as.factor(data$potassium)
data$chloride<- as.factor(data$chloride)

str(data)
is.na(data)



#random forest model



install.packages("caret")
library(caret)
library(ggplot2)
library(lattice)
install.packages("randomForest")
library(randomForest)
library(datasets)

str(data)

dim(data)


set.seed(123)
train_index <- createDataPartition(data$Presenceofheartdisease, p = 0.8, list = FALSE, times = 1)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]


rf <- randomForest(Presenceofheartdisease~., data=train_data, proximity=TRUE) 
print(rf)


p1 <- predict(rf, train_data)
cm <- confusionMatrix(p1, train_data$Presenceofheartdisease)
cm

p2 <- predict(rf, test_data)
confusion_matrix<- confusionMatrix(p2, test_data$Presenceofheartdisease)
confusion_matrix



cm_table <- confusion_matrix$table
cm_table


accuracy <- confusion$overall["Accuracy"]
cat("Accuracy:", accuracy, "\n")



plot(rf)
install.packages("pROC")
library(pROC)

prf <- predict(rf, newdata = test_data)


true_labels <- test_data$Presenceofheartdisease

str(true_labels)
str(prf)

prf <- as.numeric(prf)
unique(prf)
length(true_labels) == length(prf)

complete_data <- na.omit(data.frame(response = true_labels, predictor = prf))
roc_obj <- roc(response = true_labels, predictor = prf)
plot(roc_obj, main = "ROC Curve")
auc_value <- auc(roc_obj)

legend("bottomright", 
       legend = paste("AUC =", round(auc_value, 2)), 
       col = "black", bg = "white")







#support vector machine

install.packages("e1071")
library(e1071)

library(caret)
library(ggplot2)
library(lattice)
library(DescTools)
library(e1071)

Abstract(data)


set.seed(123)
index <- sample(1:nrow(data), round(0.8*nrow(data)) ,replace = F)
data_train <- data[index,]
data_test <- data[-index,]




set.seed(123)

svm_model<- 
  svm(Presenceofheartdisease ~ ., 
      data = data_train, 
      type = "C-classification", 
      kernel = "linear",
      scale = FALSE)


svm_model
summary(svm_model)

test_pred <- predict(svm_model, newdata = data_test)
test_pred

CF<- confusionMatrix(table(test_pred, data_test$Presenceofheartdisease))
CF
library(e1071)
library(ROCR)
library(e1071)
library(pROC)




p <- predict(svm_model, newdata = data_test)


true_labels <- data_test$Presenceofheartdisease

str(true_labels)
str(p)

p <- as.numeric(p)
unique(p)
length(true_labels) == length(p)

complete_data <- na.omit(data.frame(response = true_labels, predictor = p))
roc_obj <- roc(response = true_labels, predictor = p)
plot(roc_obj, main = "ROC Curve")
auc_value <- auc(roc_obj)

legend("bottomright", 
       legend = paste("AUC =", round(auc_value, 2)), 
       col = "black", bg = "white")




#KNN


library(caret)
library(pROC)
library(mlbench)
library(ggplot2)
library(lattice)



set.seed(123)
index <- sample(1:nrow(data), round(0.8*nrow(data)) ,replace = F)
data_train <- data[index,]
data_test <- data[-index,]

trainControl <- trainControl(method="repeatedcv", 
                             number=10, 
                             repeats=3)
metric <- "Accuracy"


set.seed(5)
fit.knn <- train(Presenceofheartdisease~., 
                 data=data_train, 
                 method="knn",
                 metric=metric,
                 trControl=trainControl)

knn.k1 <- fit.knn$dataset 
print(fit.knn)

plot(fit.knn)

set.seed(7)
p1 <- predict(fit.knn, data_train)
cm_TRAIN <- confusionMatrix(p1, data_train$Presenceofheartdisease)
cm_TRAIN


p2 <- predict(fit.knn, data_test)
cm_test <- confusionMatrix(p2, data_test$Presenceofheartdisease)
cm_test


install.packages("pROC")
library(pROC)

pknn <- predict(fit.knn, newdata = test_data)


true_labels <- test_data$Presenceofheartdisease

str(true_labels)
str(pknn)

pknn <- as.numeric(pknn)
unique(pknn)
length(true_labels) == length(pknn)

complete_data <- na.omit(data.frame(response = true_labels, predictor = pknn))
roc_obj <- roc(response = true_labels, predictor = pknn)
plot(roc_obj, main = "ROC Curve")
auc_value <- auc(roc_obj)

legend("bottomright", 
       legend = paste("AUC =", round(auc_value, 2)), 
       col = "black", bg = "white")

#Naive Bayes model

install.packages(naivebayes)
library(naivebayes)
library(dplyr)
library(ggplot2)
library(psych)

install.packages("e1071")
library(caret)
library(DescTools)
library(e1071)

Abstract(data)

set.seed(123)
index <- sample(1:nrow(data), round(0.8*nrow(data)) ,replace = F)
data_train <- data[index,]
data_test <- data[-index,]

nb_model <- naiveBayes(Presenceofheartdisease ~ ., data = data_train)
nb_model



p1 <- predict(nb_model, data_train)
p1

cm_train <- confusionMatrix(p1, data_train$Presenceofheartdisease)
cm_train 


p2 <- predict(nb_model, data_test)
p2

cm_test <- confusionMatrix(p2, data_test$Presenceofheartdisease)
cm_test

install.packages("pROC")
library(pROC)

pnb <- predict(nb_model, newdata = data_test)


true_labels <- data_test$Presenceofheartdisease

str(true_labels)
str(pnb)

pnb <- as.numeric(pnb)
unique(pnb)
length(true_labels) == length(pnb)

complete_data <- na.omit(data.frame(response = true_labels, predictor = pnb))
roc_obj <- roc(response = true_labels, predictor = pnb)
plot(roc_obj, main = "ROC Curve")
auc_value <- auc(roc_obj)

legend("bottomright", 
       legend = paste("AUC =", round(auc_value, 2)), 
       col = "blue", bg = "white")




#Desision TREE MODEL


#decision tree
data<-only_cats

head(data)

str(data)
is.na(data)

#random forest

data$Presenceofheartdisease<-as.factor(data$Presenceofheartdisease)
str(data)


data$Gender<-as.factor(data$Gender)
data$CurrentSmoking<-as.factor(data$CurrentSmoking)
data$agegroup<-as.factor(data$agegroup)
data$bmigroup<-as.factor(data$bmigroup)
data$sbplevel<-as.factor(data$sbplevel)
data$dbplevel<-as.factor(data$dbplevel)
data$hrgroup<-as.factor(data$hrgroup)
data$Hblevel<-as.factor(data$Hblevel)
data$platelets<-as.factor(data$platelets)
data$wbc<-as.factor(data$wbc)
data$rbcgrp<-as.factor(data$rbcgrp)
data$Neutrophils<-as.factor(data$Neutrophils)
data$Creatinine<-as.factor(data$Creatinine)
data$ldllevel<-as.factor(data$ldllevel)
data$hdllevel<-as.factor(data$hdllevel)
data$cholesterol<-as.factor(data$cholesterol)
data$bloodglucose<-as.factor(data$bloodglucose)
data$sodium<- as.factor(data$sodium)
data$potassium<-as.factor(data$potassium)
data$chloride<- as.factor(data$chloride)

str(data)
is.na(data)

install.packages("rpart")
library(rpart)
install.packages(c("rpart", "rpart.plot"))
library(caret)
library(DescTools)
library(rpart)
library(rpart.plot)
install.packages("caret")
install.packages("yardstick")
library(caret)
library(yardstick)
library(ggplot2)
library( lattice)


set.seed(123)
train_index <- createDataPartition(data$Presenceofheartdisease, p = 0.8, list = FALSE)
data_train <- data[train_index, ]
data_test <- data[-train_index, ]


tree_model <- rpart(Presenceofheartdisease ~ ., data = data , method = "class")
tree_model

tree_model$variable.importance

prp(x = tree_model, extra =2) 
base.trpreds <- predict(object = tree_model, 
                        newdata = data_train, 
                        type = "class") 

base.trpreds

levels(base.trpreds)
levels(data_train$Presenceofheartdisease)

data_train$Presenceofheartdisease <- factor(data_train$Presenceofheartdisease, 
                                            levels = levels(base.trpreds))

DT_train_conf <- confusionMatrix(data = base.trpreds, # predictions
                                 reference = data_train$Presenceofheartdisease, # actual
                                 positive = "No",
                                 mode = "everything")
DT_train_conf


base.trpreds <- predict(object = tree_model, 
                        newdata = data_test, 
                        type = "class") 

base.trpreds
levels(base.trpreds)
levels(data_test$Presenceofheartdisease)
data_test$Presenceofheartdisease <- factor(data_test$Presenceofheartdisease, levels = levels(base.trpreds))

DT_test_conf <- confusionMatrix(data = base.trpreds, # predictions
                                reference = data_test$Presenceofheartdisease, # actual
                                positive = "No",
                                mode = "everything")
DT_test_conf

library(rpart)
library(pROC)

predictions <- predict(tree_model, newdata = data_test, type = "prob")[, "No"]
roc_obj <- roc(response = data_test$Presenceofheartdisease, 
               predictor = predictions, 
               levels = c("No", "Yes"),
               direction = ">")


plot(roc_obj, main = "ROC Curve")
auc_value <- auc(roc_obj)

legend("bottomright", 
       legend = paste("AUC =", round(auc_value, 2)), 
       col = "blue", bg = "white")




#glm model 

library(caret)
library(ggplot2)
library(lattice)


set.seed(123)
index <- sample(1:nrow(data), round(0.8*nrow(data)) ,replace = F)
data_train <- data[index,]
data_test <- data[-index,]


glm_model<- glm(Presenceofheartdisease ~ Gender+
                  CurrentSmoking+
                  agegroup+
                  bmigroup+
                  sbplevel+
                  dbplevel+
                  hrgroup+
                  Hblevel+
                  platelets+
                  wbc+
                  rbcgrp+
                  Neutrophils+
                  Creatinine+
                  ldllevel+
                  hdllevel+
                  cholesterol+
                  bloodglucose+
                  sodium+
                  potassium+chloride,
                data = data_train, family = binomial(link = "logit"))

glm_model
summary(glm_model)


levels(data_test$sodium) <- as.factor(data_train$sodium)

probabilities <- predict(glm_model, newdata = data_test, type = "response")
predictions <- ifelse(probabilities > 0.5, "Yes", "No")
predictions

predictions_factor <- factor(predictions, 
                             levels = levels(data_test$Presenceofheartdisease))

conf_matrix <- confusionMatrix(predictions_factor, data_test$Presenceofheartdisease)
print(conf_matrix)

accuracy <- conf_matrix$overall["Accuracy"]
cat("Accuracy:", accuracy, "\n")





install.packages("pROC")
library(pROC)

pglm <- predict(glm_model, newdata =data_test)


true_labels <- data_test$Presenceofheartdisease

str(true_labels)
str(ptree)

pglm <- as.numeric(pglm)
unique(pglm)
length(true_labels) == length(pglm)

complete_data <- na.omit(data.frame(response = true_labels, predictor = pglm))
roc_obj <- roc(response = true_labels, predictor = pglm)
plot(roc_obj, main = "ROC Curve")
auc_value <- auc(roc_obj)

legend("bottomright", 
       legend = paste("AUC =", round(auc_value, 2)), 
       col = "blue", bg = "white")

