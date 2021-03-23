rm(list = ls())

#### Importing required packages ####

#### Installing required libraries ####
x <- c('dplyr', 'ggplot2', 'plotly', 'corrplot','caret','caTools', 'glmnet', 'pROC',
       'ROSE', 'e1071', 'MLmetrics','randomForest')
# 'ROCit'
lapply(x, require, character.only = TRUE)
rm(x)

# Set working directory
setwd('D:\\Data Science\\Edwisor\\Projects\\Customer Transaction Prediction')

#### Importing Datasets ####
df_test <- read.csv('test.csv', header = TRUE)
df_train <- read.csv('train.csv', header = TRUE)

# Dimension
dim(df_test)
dim(df_train)

# Summary
summary(df_test)
summary(df_train)

# Structure
str(df_test)
str(df_train)

# Percentage of different classes in the target variable
table(df_train$target)/nrow(df_train)*100

# Plotting Imbalance
ggplot(data = df_train, aes(target)) + theme_bw() + geom_bar(stat = 'count', fill = 'orange', col = 'black')

# # As we can see all except CUST ID are integer or numeric, let us drop it
# df_train <- df_train[-1]

# As we can see the dataset has class imbalance which if not fixed will effect the 
# performance of the model

# Creating new dataframe
df = subset(df_train, select = -c(ID_code, target))
test = subset(df_test, select = -ID_code)

#### EDA ####
# Missing Values
sum(is.na(df))
sum(is.na(test))
# There are no missing values in the test dataset

# Outlier Analysis
# Train Data
for (i in 2:200) {
  boxplot(df[i], main = colnames(df[i]))
}

# Test Data
for (i in 2:200) {
  boxplot(test[i], main = colnames(test[i]))
}

# Almost all variables have outliers.
# Replace outliers with nan in Train Data
for(i in colnames(df)){
  val = df[,i][df[,i] %in% boxplot.stats(df[,i])$out]
  df[,i][df[,i] %in% val] = NA
}

# Imputing mean in na
for(i in colnames(test)){
  df[is.na(df[,i]), i] <- mean(df[,i], na.rm = TRUE)
}

# Replace outliers with nan in Test Data
for(i in colnames(test)){
  val = test[,i][test[,i] %in% boxplot.stats(test[,i])$out]
  test[,i][test[,i] %in% val] = NA
}

# Imputing mean in na
for(i in colnames(test)){
  test[is.na(test[,i]), i] <- mean(test[,i], na.rm = TRUE)
}

# Correlation Matrix
correlation_matrix = View(cor(df))

# From the correlation matrix it is clear that there is low correlation between 
# variables to a point where they are not correlated at all.

#### Visualizing Data ####
# Histogram to visualize the distribution of data
par(mfrow = c(2,1))
for (i in 2:200) {
  hist(df[,i], main = colnames(df[i]), col = 'magenta2')
}

# From the histogram we can see that most of the data is normally distributed.

# Almost all of the data is uniformly distributed but the values for each variable varies
# at different ranges. So in order to make it better for the ML model the data needs to be scaled.

#### Scaling both datasets ####
train_std <- scale(df)
test_std <- scale(test)

# merge train_std and df_train in df
df <- as.data.frame(cbind(df_train$target, train_std))
colnames(df)[colnames(df) == 'V1'] <- 'target'

# As we know the data is heavily imbalanced so we are going to use a technique called
# oversampling in order to get rid of the class imbalance
set.seed(123)
df_treated <- ovun.sample(target ~ .,data = df, method = 'over', N = 359804)$data
table(df_treated$target)

#### Splitting Dataset ####
# Splitting Imbalanced Dataset
set.seed(123)
split <- sample.split(df$target, SplitRatio = 0.7)
train_cl <- subset(df, split == TRUE)
test_cl <- subset(df, split == FALSE)

# Splitting Balanced Dataset
set.seed(123)
split_bal <- sample.split(df_treated$target, SplitRatio = 0.7)
train_bal <- subset(df_treated, split_bal == TRUE)
test_bal <- subset(df_treated, split_bal ==FALSE)

#### MODELS ####
#### Logistic Regression - Imbalanced Data ####
set.seed(123)
model_lm <- glm(target ~ ., train_cl, family = 'binomial')
model_lm

# Model Performance on test data
set.seed(42)
prob_lm <- predict(model_lm, test_cl)

# convert probality to class according to threshold
pred_lm <- ifelse(prob_lm > 0.5, 1, 0)

# Confusion Matrix
cm_lm <- table(pred_lm, test_cl$target)
cm_lm

# ROC Curve and AUC
par(mfrow = c(1,1))
set.seed(22)
roc.curve(test_cl$target, pred_lm, col = 'blue')

# Classification Scores
confusionMatrix(cm_lm)
Accuracy(pred_lm, test_cl$target)

# Precision, Recall and F1-Score

cat('Precison:',Precision(test_cl$target, pred_lm))
cat('Recall', Recall(test_cl$target, pred_lm))
cat('F1 score:',F1_Score(test_cl$target, pred_lm))

#### Logistic Regression - Balanced Data(Oversampling) ####
set.seed(123)
model_lm_bal <- glm(target ~ ., train_bal, family = 'binomial')
model_lm_bal

# Model Performance on Test Data
set.seed(42)
prob_lm_bal <- predict(model_lm_bal, test_bal)

# convert probality to class according to threshold
pred_lm_bal <- ifelse(prob_lm_bal > 0.5, 1, 0)

# Confusion Matrix
cm_lm_bal <- table(pred_lm_bal, test_bal$target)
cm_lm_bal

# ROC Curve and AUC
par(mfrow = c(1,1))
set.seed(22)
roc.curve(test_bal$target, pred_lm_bal, col = 'blue')

# Classification Scores
confusionMatrix(cm_lm_bal)
Accuracy(pred_lm_bal, test_bal$target)

# Precision, Recall and F1-Score

cat('Precison:',Precision(test_bal$target, pred_lm_bal))
cat('Recall', Recall(test_bal$target, pred_lm_bal))
cat('F1 score:',F1_Score(test_bal$target, pred_lm_bal))

# The classification scores and the ROC score clearly show that the model is working poorly.
# Let us try another model with imbalanced data.

#### Random Forest - Imbalanced Data ####
set.seed(42)
model_rfc <- randomForest(as.factor(train_cl$target) ~ ., train_cl, ntree=10)

# Model Performance
set.seed(123)
pred_rfc <- predict(model_rfc, test_cl)

# Confusion Matrix
cm_rfc <- table(pred_rfc, test_cl$target)
cm_rfc

# ROC Score and AUC
par(mfrow = c(1,1))
set.seed(22)
roc.curve(test_cl$target, pred_rfc, col = 'blue')

# Classsification Scores
confusionMatrix(cm_rfc)
Accuracy(pred_rfc, test_cl$target)

# Precision, Recall and F1-Score

cat('Precison:',Precision(test_cl$target, pred_rfc))
cat('Recall', Recall(test_cl$target, pred_rfc))
cat('F1 score:',F1_Score(test_cl$target, pred_rfc))

#### Random Forestr Classifier - Balanced Data(Oversampling) ####
set.seed(101)
model_rfc_bal <- randomForest(as.factor(train_bal$target) ~ ., train_bal, ntree = 10)
model_rfc_bal

# Model Performance
set.seed(21)
pred_rfc_bal <- predict(model_rfc_bal, test_bal)

# Confusion matrix
cm_rfc_bal <- table(pred_rfc_bal, test_bal$target)
cm_rfc_bal

# ROC Score and AUC
par(mfrow = c(1,1))
set.seed(22)
roc.curve(test_bal$target, pred_rfc_bal, col = 'blue')

# Classification Scores
confusionMatrix(cm_rfc_bal)
Accuracy(pred_rfc_bal, pred_rfc_bal)

# Precision, Recall and F1-Score
cat('Precison:',Precision(test_bal$target, pred_rfc_bal))
cat('Recall', Recall(test_bal$target, pred_rfc_bal))
cat('F1 score:',F1_Score(test_bal$target, pred_rfc_bal))

#### Naive Bayes - Imbalanced Data ####
set.seed(21)
model_nb <- naiveBayes(as.factor(target) ~ ., data = train_cl)

# Model performance
set.seed(123)
pred_nb <- predict(model_nb, test_cl)

# Confusion Matrix
cm_nb <- table(pred_nb, test_cl$target)
cm_nb

# ROC Score and AUC
par(mfrow = c(1,1))
set.seed(22)
roc.curve(test_cl$target, pred_nb, col = 'blue')

# Classification Scores
confusionMatrix(cm_nb)
Accuracy(pred_nb, test_cl$target)

# Precision, Recall and F1-Score

cat('Precison:',Precision(test_cl$target, pred_nb))
cat('Recall', Recall(test_cl$target, pred_nb))
cat('F1 score:',F1_Score(test_cl$target, pred_nb))

#### Naive Bayes - Balanced Data(Oversampling) ####
set.seed(21)
model_nb_bal <- naiveBayes(as.factor(target) ~ .,data = train_bal)

# Model Performance on Test Data
set.seed(123)
pred_nb_bal <- predict(model_nb_bal, test_bal)

# Confusion Matrix
cm_nb_bal <- table(pred_nb_bal, test_bal$target)
cm_nb_bal

# ROC Curve and AUC
par(mfrow = c(1,1))
set.seed(22)
roc.curve(test_bal$target, pred_nb_bal, col = 'blue')

# Classification Scores
confusionMatrix(cm_nb_bal)
Accuracy(pred_nb_bal, test_bal$target)

# Precision, Recall and F1-Score
cat('Precison:',Precision(test_bal$target, pred_nb_bal))
cat('Recall', Recall(test_bal$target, pred_nb_bal))
cat('F1 score:',F1_Score(test_bal$target, pred_nb_bal))

# Predicting Naive Bayes on test data
pred_test <- predict(model_nb_bal, test_std)


#### Final CSV ####
df_predicted <- data.frame(ID_code = df_test$ID_code, target = pred_test)
write.csv(df_predicted, 'Predicted Data.csv', row.names = F)