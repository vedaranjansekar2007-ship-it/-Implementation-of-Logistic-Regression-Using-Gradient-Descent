# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import libraries and load the placement dataset.
2. Remove unnecessary columns and encode categorical values into numbers.
3. Split the data into input features (x) and target output (y).
4. Initialize random weights (theta) and define the sigmoid function.
5. Train the logistic regression model using gradient descent by updating weights repeatedly.
6. Predict placement results and calculate model accuracy.

## Program:
```

Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: VEDARANJAN S
RegisterNumber: 212225220119

import pandas as pd
import numpy as np
data=pd.read_csv(r"C:\Users\acer\Downloads\Placement_Data (2).csv")
data.head()
data1=data.copy()
data1.head()
data1=data1.drop(['sl_no','salary'],axis=1)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
x=data1.iloc[:, : -1]
y=data1["status"]
theta=np.random.randn(x.shape[1])
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,x,y):
    h=sigmoid(x.dot(theta))
    return -pd.sum(y*np.log(h)+(1-y)*np.log(1-h)) 
def gradient_descent(theta,x,y,alpha,num_iterations):
    m=len(y)
    for i in range (num_iterations):
        h=sigmoid(x.dot(theta))
        gradient=x.T.dot(h-y)/m
        theta-=alpha*gradient
    return theta
theta=gradient_descent(theta,x,y,alpha=0.01,num_iterations=1000)
def predict(theta,x):
    h=sigmoid(x.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred
y_pred=predict(theta,x)
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
print("Predicted:\n",y_pred)
print("Actual:\n",y.values)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print("Prdicted Result:",y_prednew) 

```

## Output:

<img width="1410" height="536" alt="image" src="https://github.com/user-attachments/assets/9f2e653c-49cf-4c38-97ef-94d3203007de" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

