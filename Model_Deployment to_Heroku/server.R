model <- readRDS("model.rds")

library(shiny)
library(data.table)
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
      stringsAsFactors = TRUE)
    
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