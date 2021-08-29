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