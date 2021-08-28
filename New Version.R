#read the csv
df <- read.csv("Telco-Customer-Churn.csv", header = TRUE,
               stringsAsFactors = TRUE)
#delete the first column ID
df$customerID <- NULL
#load necessary libraries
library(caret)
library(pROC)
library(reshape)
library(rpart)
library(rpart.plot)

#1 Data exploration----
#check column name
t(t(names(df)))
#check summary of all variables
summary(df)

#do visualizations on some columns
barplot(data.frame(table(df[1]))$Freq, names.arg = data.frame(table(df[2]))$Var1, 
        xlab = "Gender", ylab = "Frequency", main = "Bar Chart of gender")
barplot(data.frame(table(df[2]))$Freq, names.arg = data.frame(table(df[3]))$Var1, 
        xlab = "Senior citizen", ylab = "Frequency", main = "Bar Chart of Senior Citizen")
hist(df$tenure, xlab = "tenure", main = "Histogram of Tenure")
barplot(data.frame(table(df$Contract))$Freq, names.arg = data.frame(table(df[15]))$Var1, 
        xlab = "Contract", ylab = "Frequency", main = "Bar Chart of Contract")
boxplot(df$MonthlyCharges, main = "Box Plot of Monthly Charge")
barplot(data.frame(table(df[20]))$Freq, names.arg = data.frame(table(df[20]))$Var1, 
        xlab = "Churn", ylab = "Frequency", main = "Bar Chart of Churn")

##Piviot tables----
library(reshape)
pv <- cast(melt(df, id = c("gender", "Churn"), measure = "Churn"),gender ~ Churn,margins=TRUE)
pv$Yes <- round(pv$Yes/pv$`(all)`,3)
pv$No <- round(pv$No/pv$`(all)`,3)
pv$`(all)` <- pv$`(all)`/pv$`(all)`
print(pv)

pv <- cast(melt(df, id = c("gender", "Churn"), measure = "Churn"),gender ~ Churn,margins=TRUE)
pv$Yes <- round(pv$Yes/pv$`(all)`,3)
pv$No <- round(pv$No/pv$`(all)`,3)
pv$`(all)` <- pv$`(all)`/pv$`(all)`
print(pv)

pv <- cast(melt(df, id = c("Contract", "Churn"), measure = "Churn"),Contract ~ Churn,margins=TRUE)
pv$Yes <- round(pv$Yes/pv$`(all)`,3)
pv$No <- round(pv$No/pv$`(all)`,3)
pv$`(all)` <- pv$`(all)`/pv$`(all)`
print(pv)
# we can delete this variable after we're done to keep it clean
rm(pv)

##Conditional Mean----
data.for.plot <- aggregate(df$tenure, by = list(df$Churn), FUN = mean)
names(data.for.plot) <- c("Churn", "Meantenure")
barplot(data.for.plot$Meantenure, names.arg = data.for.plot$Churn,
        xlab = "Churn", ylab = "Mean tenure", main = "Conditional Mean of Tenure")

# we can delete this variable after we're done to keep it clean
rm(data.for.plot)

#2 Data preprocessing----
#make missing value 0 because they're new users
df$TotalCharges[is.na(df$TotalCharges)] <- 0
summary(df$TotalCharges)

#4 Partition the data----
#use 10 fold cross validation
"folds <- 10
row.index.list <- list()
row.index <- 1:dim(df)[1]
set.seed(1)
for(fold in 1:folds){
  sample.size <- length(row.index) / (folds + 1 - fold)
  sampled.row.index <- sample(row.index, sample.size)
  row.index.list[[fold]] <- sampled.row.index
  row.index <- setdiff(row.index, sampled.row.index)}"
install.packages("e1071")
library(caret)
library(e1071)
set.seed(1234)

#create index matrix
index <- createDataPartition(df$Churn, p=0.8, list=FALSE, times=1)

#create train and test
train <- df[index,]
test <- df[-index,]

#relabel
#train$Churn[train$Churn==1] <- "Yes"
#train$Churn[train$Churn==1] <- "No" 
#test$Churn[test$Churn==1] <- "Yes"
#test$Churn[test$Churn==1] <- "No" 

#5 Model building----
##Logistic regression----
#configure numebr of folds
fold <- trainControl(method="cv",number=10,
                     savePredictions = "all",
                     classProbs=TRUE)
set.seed(1234)

#Logisitic regression
logistic <- train(Churn ~SeniorCitizen+Dependents+tenure+MultipleLines+
                    InternetService+TechSupport+StreamingTV+
                    StreamingMovies+Contract+PaperlessBilling+
                    PaymentMethod+MonthlyCharges+TotalCharges 
                  ,data = train, method="glm",family = binomial,
                  trControl = fold)
print(logistic)

#variable importance
varImp(logistic)

#apply model to test set which it hasn't seen
logisticpred <- predict(logistic, newdata=test)
#logisticpred <- ifelse(logisticprob >= 0.5, "Yes", "No")

#confusion matrix
confusionMatrix(data=logisticpred, test$Churn, positive = 'Yes')

#6 Handle imbalanced data----
#oversampling
install.packages("ROSE")
library(ROSE)

oversample <- ovun.sample(Churn~., data=train, method = "over", N =8280)$data

logistic2 <- train(Churn ~SeniorCitizen+Dependents+tenure+MultipleLines+
                    InternetService+TechSupport+StreamingTV+
                    StreamingMovies+Contract+PaperlessBilling+
                    PaymentMethod+MonthlyCharges+TotalCharges 
                  ,data = oversample, method="glm",family = binomial,
                  trControl = fold)
logisticpred2 <- predict(logistic2, newdata=test)
confusionMatrix(data=logisticpred2, test$Churn, positive = 'Yes')

#undersampling
undersample <- ovun.sample(Churn~., data=train, method = "under", N =2992)$data

logistic3 <- train(Species ~SeniorCitizen+Dependents+tenure+MultipleLines+
                     InternetService+TechSupport+StreamingTV+
                     StreamingMovies+Contract+PaperlessBilling+
                     PaymentMethod+MonthlyCharges+TotalCharges 
                   ,data = undersample, method='nb',family = binomial,
                   trControl = fold)
logisticpred3 <- predict(logistic3, newdata=test)
confusionMatrix(data=logisticpred3, test$Churn, positive = 'Yes')

