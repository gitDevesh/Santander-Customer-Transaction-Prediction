import os
import pandas as pd
import numpy as np
import requests

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)

os.chdir(r'D:\Data Science\Edwisor\Projects\Customer Transaction Prediction')
os.getcwd()

df_test = pd.read_csv('test.csv')
df_train = pd.read_csv('train.csv')

df_train

df_test

df_train.describe()

df_train.info()

df_test.info()

# creating new dataframes
df = df_train.drop(columns=['ID_code', 'target'], axis=1)
test = df_test.drop(columns='ID_code', axis=1)

# EDA

# Missing Value Analysis
df.isnull().sum().value_counts()

# There are no missing values in the dataset

# There are no missing values in the datasets

# Outlier Analysis
plt.figure(figsize=(30, 400))
for i in range(1, 199):
    i += 1
    plt.subplot(67, 3, i)
    plt.boxplot(df[df.columns[i]])
plt.show()

# Replacing outliers with nan
for i in df.columns:
    # print(i)
    q75, q25 = np.percentile(df.loc[:, i], [75, 25])
    iqr = q75 - q25

    min = q25 - (1.5 * iqr)
    max = q75 + (1.5 * iqr)

    df.loc[df[i] < min, i] = np.nan
    df.loc[df[i] > max, i] = np.nan

# Imputing values in nan
df = df.loc[:i].fillna(df.loc[:i].mean())

# Replacing outliers with nan in test dataset
for i in test.columns:
    # print(i)
    q75, q25 = np.percentile(test.loc[:, i], [75, 25])
    iqr = q75 - q25

    min = q25 - (1.5 * iqr)
    max = q75 + (1.5 * iqr)

    test.loc[test[i] < min, i] = np.nan
    test.loc[test[i] > max, i] = np.nan

# Imputing values in nan
test = test.loc[:i].fillna(test.loc[:i].mean())

# Let's check if there is a class imbalance in the dataset
plt.figure(figsize=(6, 6))
sns.countplot(df_train['target'])

# Percentage of class imbalance
df_train['target'].value_counts() / len(df_train) * 100

# As we can see on the above plot and the percentage of two values in target variable there is a big class imbalance in the
# dataset resulting in the model being biased and giving wrong predictions.

# CHecking the correlation of variables in datasets
corr = df.corr()
corr

# There is low correlation between variables to a point where they are not correlated at all.

# Visualizing the data

plt.figure(figsize=(30, 150))
for i in range(1, 199):
    plt.subplot(67, 3, i + 1)
    sns.histplot(df[df.columns[i]], kde_kws={'bw': 0.05, 'lw': 2}, color='red')
plt.tight_layout

# As we can see the almost all of the data us uniformaly distributed but the values for each variable varies at different 
# ranges. So in order to make it better for the ML model the data needs to be scaled.

scaler = StandardScaler()

train_std = pd.DataFrame(scaler.fit_transform(df))
test_std = pd.DataFrame(scaler.fit_transform(test))

train_std

test_std

# Splitting Dataset

X = train_std
y = df_train['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# We know there is class imbalance problem which will make the model biased. Let us create a logistic model without treating
# the imblance and a model with the imbalance problem treated to see how the model performs on both of them.

# Let us treat the imbalance in the dataset first. There are several ways to treat the class imbalance.
# We are going to oversample the minority using SMOTE.

# Treating Imbalanced Splitted Data using SMOTE

sm = SMOTE(random_state=21)

X_train_sm, y_train_sm = sm.fit_sample(X_train, y_train)
X_test_sm, y_test_sm = sm.fit_sample(X_test, y_test)
print(X_train_sm.shape)
print(X_test_sm.shape)
print(y_train_sm.shape)
print(y_test_sm.shape)

# Creating Model
# 1) Logistic Regression with Class mbalance

model_lm = LogisticRegression(max_iter=100, random_state=21).fit(X_train, y_train)

pred_lm = model_lm.predict(X_test)
pred_lm

score_lm = accuracy_score(y_test, pred_lm)
print('The accuracy score is: ', score_lm)

# Accuracy is not the best metric to evaluate the models' perfromance so we will introduce some new metrics
# The metrics for evaluation being ROC Score, Precision and Recall

# Confusion Matrix
cm_lm = confusion_matrix(y_test, pred_lm)
cm_lm = pd.crosstab(y_test, pred_lm)
cm_lm

# ROC Score
roc_score_lm = roc_auc_score(y_test, pred_lm)
print('The roc score is ', roc_score_lm)

plt.figure()
fpr_lm, rec_lm, thresh_lm = roc_curve(y_test, pred_lm)
plt.plot(fpr_lm, rec_lm, label='Area under ROC curve = %0.3f)' % roc_score_lm)
plt.plot([0, 1], [0, 1], 'r--')
plt.legend()
plt.title('ROC Curve')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate(Recall)')
plt.show()
print('The ROC score is ', roc_score_lm)

print(classification_report(y_test, pred_lm))

# The area under the ROC curve is 0.6262279562629977 and the f1-score for customers that will do transaction is also very
# low compared to those who will not do the transaction which shows that the model will not perform well on imbalanced data.

# Logistic Regression: Class Imbalance Treated

model_lm_sm = LogisticRegression(max_iter=100, random_state=21).fit(X_train_sm, y_train_sm)

pred_lm_sm = model_lm_sm.predict(X_test_sm)
pred_lm_sm

score_lm_sm = accuracy_score(y_test_sm, pred_lm_sm)
print('The accuracy score is: ', score_lm_sm)

# Confusion Matrix
cm_lm_sm = confusion_matrix(y_test_sm, pred_lm_sm)
cm_lm_sm = pd.crosstab(y_test_sm, pred_lm_sm)
cm_lm_sm

roc_score_lm_sm = roc_auc_score(y_test_sm, pred_lm_sm)

plt.figure()
fpr_lm_sm, rec_lm_sm, thresh_lm_sm = roc_curve(y_test_sm, pred_lm_sm)
plt.plot(fpr_lm_sm, rec_lm_sm, label='Area under ROC curve: %0.3f' % roc_score_lm_sm)
plt.legend()
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate(Recall)')
plt.title('ROC Curve')
plt.show()
print('The ROC score is: ', roc_score_lm_sm)

print(classification_report(y_test_sm, pred_lm_sm))

# As we can see using SMOTE and over sampling the minority class the ROC score and the f1-score improved. Now f1-score for 
# both the customers who will do transaction and those who will not is high.

# 2) Random Forest Classifier with class imbalance

model_rfc = RandomForestClassifier(random_state=21, n_estimators=10).fit(X_train, y_train)

pred_rfc = model_rfc.predict(X_test)

score_rfc = accuracy_score(y_test, pred_rfc)
print('The accuracy score is: ', score_rfc)

# Confusion Matrix
cm_rfc = confusion_matrix(y_test, pred_rfc)
cm_rfc = pd.crosstab(y_test, pred_rfc)
cm_rfc

roc_score_rfc = roc_auc_score(y_test, pred_rfc)
print('The ROC score is: ', roc_score_rfc)

plt.figure()
fpr_rfc, rec_rfc, thresh_rfc = roc_curve(y_test, pred_rfc)
plt.plot(fpr_rfc, rec_rfc, label='Area under ROC curve: %0.3f' % roc_score_rfc)
plt.legend()
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Postive Rate(Recall)')
plt.title('ROC Curve')
plt.show

print(classification_report(y_test, pred_rfc))

# The ROC score and curve clearly shows the model is preforming poorly. It is completely biased towards the majority class.
# Let us use the same algorithm in the balanced dataset.

# Random Forest Classifier : Class Imbalance Treated

model_rfc_sm = RandomForestClassifier(random_state=21, n_estimators=30).fit(X_train_sm, y_train_sm)

pred_rfc_sm = model_rfc_sm.predict(X_test_sm)

score_rfc_sm = accuracy_score(y_test_sm, pred_rfc_sm)
print('The accuracy score is: ', score_rfc_sm)

# Confusion Matrix
cm_rfc_sm = confusion_matrix(y_test_sm, pred_rfc_sm)
cm_rfc_sm = pd.crosstab(y_test_sm, pred_rfc_sm)
cm_rfc_sm

roc_score_rfc_sm = roc_auc_score(y_test_sm, pred_rfc_sm)
print('The ROC score is: ', roc_score_rfc_sm)

plt.figure()
fpr_rfc_smt, rec_rfc_smt, thresh_rfc_smt = roc_curve(y_test_sm, pred_rfc_sm)
plt.plot(fpr_rfc_smt, rec_rfc_smt, label='Area under ROC curve: %0.3f' % roc_score_rfc_sm)
plt.legend()
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate(Recall)')
plt.title('ROC Cruve')
plt.show()

print(classification_report(y_test_sm, pred_rfc_sm))

# Even though the ROC score of balanced data is much better than imbalanced data, recall and f1-score are still not satisfactory
# Between the two models the better model for this particular problem is Logistic Regression.

# Let us try one more model to compare its performance with logistic Regression model.

# 3) Naive Bayes with class imbalance

gnb = GaussianNB()

model_gnb = gnb.fit(X_train, y_train)

pred_gnb = model_gnb.predict(X_test)
pred_gnb

score_gnb = accuracy_score(y_test, pred_gnb)
print('The accuracy score is: ', score_gnb)

# Confusion Matrix
cm_gnb = confusion_matrix(y_test, pred_gnb)
cm_gnb = pd.crosstab(y_test, pred_gnb)
cm_gnb

roc_score_gnb = roc_auc_score(y_test, pred_gnb)
print('The ROC score is %0.6f' % roc_score_gnb)

plt.figure()
fpr_gnb, rec_gnb, thresh_gnb = roc_curve(y_test, pred_gnb)
plt.plot(fpr_gnb, rec_gnb, label='Area under ROC curve: %0.3f' % roc_score_gnb)
plt.legend()
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('False Positive Rate', fontsize=11)
plt.ylabel('True Positive Rate(Recall)', fontsize=11)
plt.title('ROC Curve')
plt.show()

print(classification_report(y_test, pred_gnb))

# Naive Bayes : Class Imbalance Treated

model_gnb_sm = gnb.fit(X_train_sm, y_train_sm)

pred_gnb_sm = gnb.predict(X_test_sm)
pred_gnb_sm

score_gnb_sm = accuracy_score(y_test_sm, pred_gnb_sm)
print('The accuracy score is %0.5f' % score_gnb_sm)

# Confusion Matrix
cm_gnb_sm = confusion_matrix(y_test_sm, pred_gnb_sm)
cm_gnb_sm = pd.crosstab(y_test_sm, pred_gnb_sm)
cm_gnb_sm

roc_score_gnb_sm = roc_auc_score(y_test_sm, pred_gnb_sm)
print('The ROC score is %0.3f' % roc_score_gnb_sm)

plt.figure()
fpr_gnb_sm, rec_gnb_sm, thresh_gnb_sm = roc_curve(y_test_sm, pred_gnb_sm)
plt.plot(fpr_gnb_sm, rec_gnb_sm, label='Area under ROC curve: %0.3f' % roc_score_gnb_sm)
plt.legend()
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('False Postive Rate', fontsize=11)
plt.ylabel('True Positive Rate', fontsize=11)
plt.title('ROC Curve')
plt.show()

print(classification_report(y_test_sm, pred_gnb_sm))

# Predicting target on test data using Naive Bayes

test_pred = model_gnb_sm.predict(test_std)

df_predicted = pd.DataFrame({'ID Code': df_test['ID_code'].values})
df_predicted['target'] = test_pred
df_predicted.to_csv('Predicted Data.csv', index=False)

df_predicted.sum().value_counts()
