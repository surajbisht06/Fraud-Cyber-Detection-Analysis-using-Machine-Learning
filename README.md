# <p align="center"> Fraud Detection System using Machine Learning </p>
# <p align="center">![image](https://github.com/surajbisht06/Fraud-Cyber-Detection-Analysis-using-Machine-Learning/assets/158066824/1ac6cfb9-aa92-4271-9199-4eaa45f6fe9d)
</p>

## Overview

This project aims to develop a machine learning-based system for detecting fraud in cyber security transactions. The system utilizes 
various supervised and unsupervised learning techniques to analyze transactional data and identify potentially fraudulent activities.
By employing advanced algorithms, the system can adapt to evolving fraud patterns and provide accurate predictions in real-time.

**Tools:-** Excel,Python

[Datasets Used](https://docs.google.com/spreadsheets/d/1Yp_rcOS2TbVn-wHUIsCeCzkeDP7MIPLP/edit?usp=sharing&ouid=102868121048017441192&rtpof=true&sd=true )

[Python Script (Code)](cyber_security.ipynb)

[Ppt presentation](sql_prjct.pptx)

### Features 

- Data preprocessing: Clean and prepare the transactional data for analysis.
  
- Supervised learning: Train classification models to classify transactions as fraudulent or legitimate.
  
- Model evaluation: Assess the performance of the models using relevant metrics such as precision, recall, and F1-score.


## Requirements

- Python 3

- Libraries: NumPy, pandas, Sklearn, etc.

- Jupyter Lab

## Balancing an unbalanced dataset:
```py
#So, we can do Undersampling technique to balance the datasets otherwise As you can see, this model is only predicting 0, which means it’s completely ignoring the minority class in favor of the majority class.
df_majority = sample[sample.Attack == 0]
df_minority = sample[sample.Attack == 1]
df_majority_undersample = df_majority.sample(replace = False, n = 144503, random_state = 123)#random_state it's won't shuffle if we run this multiple time
b_sample = pd.concat([df_majority_undersample, df_minority])
print(b_sample.Attack.value_counts())
b_sample.shape
```
```py
fig = plt.figure(figsize = (8,5))
b_sample.Attack.value_counts().plot(kind='bar', color= ['blue','green'], alpha = 0.9, rot=0)
plt.title('Distribution of data based on the Binary attacks of our balanced dataset')
plt.show()
```
###### Result: 

![image](https://github.com/surajbisht06/Fraud-Cyber-Detection-Analysis-using-Machine-Learning/assets/158066824/9cc6de4e-76a6-4a37-9c68-194cbfda1573)

## Model evaluation:
#### Decision Tree Classifier Model
```py
ds=DecisionTreeClassifier(max_depth=3)
ds.fit(x_train,y_train)
train_pred=ds.predict(x_train)
test_pred=ds.predict(x_test)
print(accuracy_score(train_pred,y_train))
print(accuracy_score(test_pred,y_test))
```
```py
#creating list for train test accuracy
train_test = ['Train','test']
aucc = [dt_aucc,dt_test] 
plt.figure(figsize=(8, 4))
# Plot the bar graph
bars = plt.bar(train_test, aucc, color=['green', 'skyblue'])
#Add Labels and title 
plt.xlabel('Decision Tree')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison for train test')
#Show the plot
plt.show()
```

###### Result:

![image](https://github.com/surajbisht06/Fraud-Cyber-Detection-Analysis-using-Machine-Learning/assets/158066824/2f6a76a8-83fe-4b2c-9c51-fc084e5a6b71)

#### Building A Decision Tree Classifier plot 

```py
#8. building model using  decision trees classifier 
import matplotlib.pyplot as plt
from  sklearn.tree import plot_tree
plt.figure(figsize=(20,10))
plot_tree(ds,feature_names=x.columns.tolist(),class_names=["0","1"],filled=True)
plt.show()
```

###### Result:

![image](https://github.com/surajbisht06/Fraud-Cyber-Detection-Analysis-using-Machine-Learning/assets/158066824/af1bb3a0-fa36-4461-9732-274d3e52e155)


### Random Forest classifier Model
```py
rfr=RandomForestClassifier(n_estimators=9,max_depth=5,random_state=42)
rfr.fit(x_train,y_train)
test_pred_rf=rfr.predict(x_test)
train_pred_rf=rfr.predict(x_train)
print(accuracy_score(train_pred_rf,y_train))
print(accuracy_score(test_pred_rf,y_test))
test_aucc=accuracy_score(test_pred_rf,y_test)
rf_aucc=accuracy_score(train_pred_rf,y_train)
```
```py
#creating list for train test accuracy
train_test = ['Train','test']
aucc = [rf_aucc,dt_test] 
plt.figure(figsize=(8, 4))
# Plot the bar graph
bars = plt.bar(train_test, aucc, color=['green', 'skyblue'])
#Add Labels and title 
plt.xlabel('Random Forest')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison for train test accuracy')
#Show the plot
plt.show()
```

###### Result:
![image](https://github.com/surajbisht06/Fraud-Cyber-Detection-Analysis-using-Machine-Learning/assets/158066824/de60e4b2-ce5b-4023-825a-b289802253f0)



### Logistic Regression Model

```py
model = LogisticRegression()
model.fit(x_train, y_train)
train_pred_logi=model.predict(x_train)
test_pred_logi = model.predict(x_test)
print(accuracy_score(train_pred_logi,y_train))
lr_aucc=accuracy_score(train_pred_logi,y_train)
print(accuracy_score(test_pred_logi,y_test))
print(accuracy_score(test_pred_logi,y_test))
lr_test=accuracy_score(test_pred_logi,y_test)
```
```py
#creating list for train test accuracy
train_test = ['Train','Test']
aucc = [lr_aucc,lr_test] 
plt.figure(figsize=(8, 4))
# Plot the bar graph
bars = plt.bar(train_test, aucc, color=['green', 'skyblue'])
#Add Labels and title 
plt.xlabel('Logistic Regrassion')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison for Train Test')
#Show the plot
plt.show()
```

###### Result:
![image](https://github.com/surajbisht06/Fraud-Cyber-Detection-Analysis-using-Machine-Learning/assets/158066824/51f3f17a-deb0-4681-80c2-f7c5f5ac011e)



### K-Nearest Neighbour Model 
 
```py
“K-Nearest Neighbour
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
knn_model = KNeighborsClassifier(n_neighbors=5)
# Fit the model to your training data
knn_model.fit(x_train_scaled, y_train)
# Predict labels for test data
test_pred_knn = knn_model.predict(x_test_scaled)
train_pred_knn = knn_model.predict(x_train_scaled)
print(accuracy_score(train_pred_knn,y_train))
knn_aucc=accuracy_score(train_pred_knn,y_train)
print(accuracy_score(test_pred_knn,y_test))
knn_test=accuracy_score(test_pred_knn,y_test)
```
```py
#creating list for train test accuracy
train_test = ['Train','Test']
aucc = [knn_aucc,knn_test] 
plt.figure(figsize=(8, 4))
# Plot the bar graph
bars = plt.bar(train_test, aucc, color=['green', 'skyblue'])
#Add Labels and title 
plt.xlabel('KNN')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison for Train Test')
#Show the plot
plt.show()
```

###### Result:

![image](https://github.com/surajbisht06/Fraud-Cyber-Detection-Analysis-using-Machine-Learning/assets/158066824/4b6e0f2a-2c10-4e53-9dc8-e3c9a9a41b55)


### Naive Bayes Model

```py
gnb = GaussianNB()
# fit the model
gnb.fit(x_train, y_train)
y_train_pred = gnb.predict(x_train)
y_train_pred = pd.Series(y_train_pred)
Model_data_train = pd.DataFrame(y_train)
Model_data_train.shape
Model_data_train['y_pred'] = y_train_pred
print('model accuracy-->{0:0.3f}'.format(accuracy_score(y_train,y_train_pred)))
naive_aucc=accuracy_score(y_train,y_train_pred)
# Data validation on x_test
test_pred_naive=gnb.predict(x_test)
print(accuracy_score(test_pred_naive,y_test))
naive_test=accuracy_score(test_pred_naive,y_test)

from sklearn.metrics import confusion_matrix

data_table = confusion_matrix(y_train, y_train_pred)

print('Confusion matrix\n\n', data_table)

print('\nTrue Positives(TP) = ', data_table[0,0])

print('\nTrue Negatives(TN) = ', data_table[1,1])

print('\nFalse Positives(FP) = ', data_table[0,1])

print('\nFalse Negatives(FN) = ', data_table[1,0])
data_table.shape
```

###### Result:
##### Confusion Matrix:

![image](https://github.com/surajbisht06/Fraud-Cyber-Detection-Analysis-using-Machine-Learning/assets/158066824/5ca15d8f-0cd3-4d5b-bfba-451ebd569c2e)


##### Ploting Confusion Matrix Using Heat Map
```py
matrix = pd.DataFrame(data=data_table, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(matrix, annot=True, fmt='d', cmap='YlGnBu')
```
###### Result:

![image](https://github.com/AhamedSahil/CYBER-SECURITY-/assets/164605797/ed76c0a6-2220-401a-ac77-9bb3451f0347)

```py
#creating list for train test accuracy
train_test = ['Train','Test']
aucc = [naive_aucc,naive_test] 
plt.figure(figsize=(8, 4))
# Plot the bar graph
bars = plt.bar(train_test, aucc, color=['green', 'skyblue'])
#Add Labels and title 
plt.xlabel('Naive Bayes')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison for Train Test Accuracy')
#Show the plot
plt.show()
```
###### Result:

![image](https://github.com/AhamedSahil/CYBER-SECURITY-/assets/164605797/37214d65-7a7e-414f-99e8-559f9711e0cf)

### Ensamble Learning bagging Model

```py
from sklearn.ensemble import BaggingClassifier, BaggingRegressor,RandomForestClassifier,RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
pargrid_ada = {'n_estimators': [5,10,15,20,25,30,35,40]}
gscv_bagging = GridSearchCV(estimator=BaggingClassifier(), 
                        param_grid=pargrid_ada, 
                        cv=5,
                        verbose=True, n_jobs=-1, scoring='roc_auc')
gscv_results = gscv_bagging.fit(x_train, y_train)
gscv_results.best_params_
gscv_results.best_score_
ensm_aucc=metrics.roc_auc_score(y_train, pd.DataFrame(gscv_results.predict_proba(x_train))[1])
print(metrics.roc_auc_score(y_test, pd.DataFrame(gscv_results.predict_proba(x_test))[1]))
ensm_test=metrics.roc_auc_score(y_test, pd.DataFrame(gscv_results.predict_proba(x_test))[1])

#creating list for train test accuracy
train_test = ['Train','Test']
aucc = [ensm_aucc,ensm_test] 
plt.figure(figsize=(8, 4))
# Plot the bar graph
bars = plt.bar(train_test, aucc, color=['green', 'skyblue'])
#Add Labels and title 
plt.xlabel('Ensamble Learning Bagging')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison for Train Test')
#Show the plot
plt.show()

```
###### Result:
![image](https://github.com/surajbisht06/Fraud-Cyber-Detection-Analysis-using-Machine-Learning/assets/158066824/10c70994-7401-4352-afed-eeb5e37ccf85)

![image](https://github.com/AhamedSahil/CYBER-SECURITY-/assets/164605797/d6d7f150-c207-41cb-828f-7420452ff6bd)

##### Compariosion of All The Accuracy of Each Model

```py
#Create a bar graph for knn, decision tree, random forest, and logistic regression
models = ['KNN', 'Decision Tree', 'Random Forest', 'Naive Bayes','Ensemble'] 
accuracy_values = [knn_aucc, dt_aucc, rf_aucc, naive_aucc,ensm_aucc] 
plt.figure(figsize=(13, 5))
# Plot the bar graph
bars = plt.bar(models, accuracy_values, color=['blue', 'green', 'red', 'orange', 'skyblue'])
#Add accuracy values on top of each bar
plt.bar_label(bars, labels=[f"{acc:.2f}" for acc in accuracy_values])
#Add Labels and title 
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison for Different Models')
#Show the plot
plt.show()
```

###### Result:

![image](https://github.com/surajbisht06/Fraud-Cyber-Detection-Analysis-using-Machine-Learning/assets/158066824/8c83b4fe-6500-4b05-905b-debbc3c0a98f)

