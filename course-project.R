## #############################################################################
## Coursera
## Practical Machine Learning
## Johns Hopkins University
##
## Course Project

# Load necessary libraries
library(caret)
library(randomForest)
library(gbm)
library(e1071)
library(corrplot)

## #############################################################################
## Load and Explore

# Set seed for reproducibility
set.seed(123)

# Load the data
training_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testing_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

training <- read.csv(training_url, na.strings = c("NA", "", "#DIV/0!"))
testing <- read.csv(testing_url, na.strings = c("NA", "", "#DIV/0!"))

# Initial exploration
dim(training)
str(training[, 1:10])
table(training$classe)

# Check for missing values
missing_values <- colSums(is.na(training))/nrow(training)
hist(missing_values, main = "Histogram of Missing Value Proportions", xlab = "Proportion")

## #############################################################################
## Preprocess

# Remove identification and timestamp variables (columns 1-7)
training <- training[, -c(1:7)]
testing <- testing[, -c(1:7)]

# Remove columns with more than 95% missing values
columns_to_keep <- which(colSums(is.na(training))/nrow(training) < 0.95)
training <- training[, columns_to_keep]
testing <- testing[, columns_to_keep]

# Check for near-zero variance predictors
nzv <- nearZeroVar(training)
if(length(nzv) > 0) {
  training <- training[, -nzv]
  testing <- testing[, -nzv]
}

# Split the training data for validation
inTrain <- createDataPartition(training$classe, p = 0.7, list = FALSE)
train_set <- training[inTrain, ]
validation_set <- training[-inTrain, ]

## #############################################################################
## Build and evaluate models

# Train a Random Forest model with cross-validation
control <- trainControl(method = "cv", number = 5, verboseIter = FALSE)
model_rf <- train(classe ~ ., data = train_set, method = "rf", 
                  trControl = control, ntree = 200)

# Train a Gradient Boosting model
model_gbm <- train(classe ~ ., data = train_set, method = "gbm", 
                   trControl = control, verbose = FALSE)

# Evaluate models on validation set
pred_rf <- predict(model_rf, validation_set)
conf_matrix_rf <- confusionMatrix(pred_rf, factor(validation_set$classe))
accuracy_rf <- conf_matrix_rf$overall["Accuracy"]

pred_gbm <- predict(model_gbm, validation_set)
conf_matrix_gbm <- confusionMatrix(pred_gbm, factor(validation_set$classe))
accuracy_gbm <- conf_matrix_gbm$overall["Accuracy"]

# Compare accuracies
results <- data.frame(
  Model = c("Random Forest", "Gradient Boosting"),
  Accuracy = c(accuracy_rf, accuracy_gbm)
)
print(results)

# Check variable importance for the best model (assuming RF is better)
importance <- varImp(model_rf)
plot(importance, top = 20)

# Final model (choose the better model)
final_model <- if(accuracy_rf > accuracy_gbm) model_rf else model_gbm

# Final prediction on test set
final_predictions <- predict(final_model, testing)

results_df <- data.frame(
  Problem_ID = 1:20,
  Prediction = as.character(final_predictions
  )
)

# Display as a numbered list
for(i in 1:20) {
  cat(paste0(i, ". ", results_df$Prediction[i], "\n"))
}