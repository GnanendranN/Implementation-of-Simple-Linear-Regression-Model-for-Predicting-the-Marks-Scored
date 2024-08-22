# Implementation of Simple Linear Regression Model for Predicting the Marks Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries. 
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Gnanendran N
RegisterNumber: 212223240037
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```
## Output:
#### DATA SET VALUES
![image](https://github.com/user-attachments/assets/d0f18cba-9b29-42bc-8a55-b2b017be939c)
#### HEAD AND TAIL VALUES
![image](https://github.com/user-attachments/assets/e9f4d3b1-6aec-4fa5-bcc2-5be7c63ba3b7)
#### X AND Y VALUES
![image](https://github.com/user-attachments/assets/677faa7e-b2b1-4bab-bc9e-79b512dda759)
#### Predication values of X and Y
![image](https://github.com/user-attachments/assets/3d914290-f9c5-45e0-8917-458368366476)
#### MSE,MAE and RMSE
![image](https://github.com/user-attachments/assets/f9e1220b-afa1-4436-bda6-a7495335ad39)
#### Training Set
![image](https://github.com/user-attachments/assets/23c452d2-8250-48fc-9196-e3bad5361bdf)
#### Testing Set
![image](https://github.com/user-attachments/assets/c363903d-e7b3-41c5-b7d8-8c0602ef481a)
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
