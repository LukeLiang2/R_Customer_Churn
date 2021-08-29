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
##Logistic regression----(Didn't handle imbalance)
#configure numebr of folds
trainControl <- trainControl(method="LOOCV")
set.seed(1234)

#Logisitic regression
logistic <- train(Churn ~SeniorCitizen+Dependents+tenure+MultipleLines+
                    InternetService+TechSupport+StreamingTV+
                    StreamingMovies+Contract+PaperlessBilling+
                    PaymentMethod+MonthlyCharges+TotalCharges 
                  ,data = train, method="glm",family = binomial,
                  trControl = trainControl)
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
                  trControl = trainControl)
logisticpred2 <- predict(logistic2, newdata=test)
confusionMatrix(data=logisticpred2, test$Churn, positive = 'Yes')

#undersampling
undersample <- ovun.sample(Churn~., data=train, method = "under", N =2992)$data

logistic3 <- train(Churn ~SeniorCitizen+Dependents+tenure+MultipleLines+
                     InternetService+TechSupport+StreamingTV+
                     StreamingMovies+Contract+PaperlessBilling+
                     PaymentMethod+MonthlyCharges+TotalCharges 
                   ,data = undersample, method="glm",family = binomial,
                   trControl = trainControl)
logisticpred3 <- predict(logistic3, newdata=test)
confusionMatrix(data=logisticpred3, test$Churn, positive = 'Yes')

#KNN
set.seed(7)
trainControl <- trainControl(method="LOOCV")
knn <- train(Churn ~SeniorCitizen+Dependents+tenure+MultipleLines+
               InternetService+TechSupport+StreamingTV+
               StreamingMovies+Contract+PaperlessBilling+
               PaymentMethod+MonthlyCharges+TotalCharges, data=undersample,
             method="knn",metric="Accuracy", trControl=trainControl)
print(knn)
predictions <- predict(knn, test)
confusionMatrix(predictions, test$Churn, positive = 'Yes')

#7 Model deployment----
logisticfinal <- train(Churn ~SeniorCitizen+Dependents+tenure+MultipleLines+
                     InternetService+TechSupport+StreamingTV+
                     StreamingMovies+Contract+PaperlessBilling+
                     PaymentMethod+MonthlyCharges+TotalCharges 
                   ,data = undersample, method="glm",family = binomial
                   )
#save the model
saveRDS(logisticfinal, "model.rds")
#load the model
model <- readRDS("model.rds")

library(shiny)
library(data.table)
library(caret)
ui <- fluidPage(
  
  # Page header
  headerPanel('Logistic Regression Model(Churn)'),
  
  # Input values
  sidebarPanel(
    HTML("<h3>Input parameters</h3>"),
    
    selectInput("SeniorCitizen", label = "SeniorCitizen:", 
                choices = list("Yes" = 1, "No" = 0), 
                selected = "Yes"),
    selectInput("Dependents", label = "Dependents:", 
                choices = list("Yes" = "Yes", "No" = "No"), 
                selected = "Yes"),            
    sliderInput("tenure", "Tenure:",
                min = 0, max = 100,
                value = 30),
    selectInput("MultipleLines", label = "MultipleLines:", 
                choices = list("No" = "No", 
                               "No Phone service" = "No phone service",
                               "Yes" = "Yes"),
                selected = "No"), 
    selectInput("InternetService", label = "InternetService:", 
                choices = list("DSL" = "DSL", "Fiber optic" = "Fiber optic",
                               "No"="No"), 
                selected = "Fiber optic"),
    selectInput("TechSupport", label = "TechSupport:", 
                choices = list("No" = "No", 
                               "No internet" = "No internet service",
                               "Yes" = "Yes"),
                selected = "Yes"), 
    selectInput("StreamingTV", label = "StreamingTVt:", 
                choices = list("No" = "No", 
                               "No internet" = "No internet service",
                               "Yes" = "Yes"),
                selected = "No"),
    selectInput("StreamingMovies", label = "StreamingMovies:", 
                choices = list("No" = "No", 
                               "No internet" = "No internet service",
                               "Yes" = "Yes"),
                selected = "No"),
    selectInput("Contract", label = "Contract:", 
                choices = list("Monthly" = "Month-to-month", 
                               "1 yr" = "One year",
                               "2 yrs" = "Two year"),
                selected = "Monthly"),
    selectInput("PaperlessBilling", label = "PaperlessBilling:", 
                choices = list("No" = "No", 
                               "Yes" = "Yes"),
                selected = "Yes"),
    selectInput("PaymentMethod", label = "PaymentMethod:", 
                choices = list("Bank Transfer" = "Bank transfer (automatic)", 
                               "Credit Card" = "Credit card (automatic)",
                               "E-Check" = "Electronic check",
                               "Mail" = "Mailed check"),
                selected = "E-Check"),
    numericInput("MonthlyCharges", "MonthlyCharges:",
                 value=65),
    numericInput("TotalCharges", "TotalCharges:",
                 value=1400),
    
    actionButton("submitbutton", "Predict", class = "btn btn-primary")
  ),
  mainPanel(
    tags$label(h3('Status/Output')), # Status/Output Text Box
    verbatimTextOutput('contents'),
    tableOutput('tabledata') # Prediction results table
  )
)

####################################
# Server                           #
####################################

server <- function(input, output, session) {
  
  # Input Data
  datasetInput <- reactive({  
    
    # outlook,temperature,humidity,windy,play
    df <- data.frame(
      Name = c("SeniorCitizen",
               "Dependents",
               "tenure",
               "MultipleLines",
               "InternetService",
               "TechSupport",
               "StreamingTV",
               "StreamingMovies",
               "Contract",
               "PaperlessBilling",
               "PaymentMethod",
               "MonthlyCharges",
               "TotalCharges"),
      
      Value = as.character(c(input$SeniorCitizen,
                             input$Dependents,
                             input$tenure,
                             input$MultipleLines,
                             input$InternetService,
                             input$TechSupport,
                             input$StreamingTV,
                             input$StreamingMovies,
                             input$Contract,
                             input$PaperlessBilling,
                             input$PaymentMethod,
                             input$MonthlyCharges,
                             input$TotalCharges
      )),
      stringsAsFactors = FALSE)
    
    Churn <- "Churn"
    df <- rbind(df, Churn)
    input <- transpose(df)
    write.table(input,"input.csv", sep=",", quote = FALSE, row.names = FALSE, col.names = FALSE)
    
    test <- read.csv(paste("input", ".csv", sep=""), header = TRUE)
    
    #test$outlook <- factor(test$outlook, levels = c("overcast", "rainy", "sunny"))
    
    
    Output <- data.frame(Prediction=predict(model,test), round(predict(model,test,type="prob"), 3))
    print(Output)
    
  })
  
  # Status/Output Text Box
  output$contents <- renderPrint({
    if (input$submitbutton>0) { 
      isolate("Calculation complete.") 
    } else {
      return("Server is ready for calculation.")
    }
  })
  
  # Prediction results table
  output$tabledata <- renderTable({
    if (input$submitbutton>0) { 
      isolate(datasetInput()) 
    } 
  })
  
}

####################################
# Create the shiny app             #
####################################
shinyApp(ui = ui, server = server)




