# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Upload the file to your cell.
2.Type the required program.
3.Print the program.
4.End the program. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: 212222240008
RegisterNumber: AMURTHA VAAHINI.KN
*/
RegisterNumber: 212222240082
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error , mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()

df.tail()

X = df.iloc[:,:-1].values
X

y = df.iloc[:,-1].values
y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)

y_pred

y_test

plt.scatter(X_train,y_train,color="green")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,y_test,color="grey")
plt.plot(X_test,regressor.predict(X_test),color="purple")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
*/
*/
```

## Output:
![Screenshot 2023-04-05 141815](https://user-images.githubusercontent.com/118679102/230143236-d87d026a-2539-444d-a3b2-19faa68838ce.png)
![Screenshot 2023-04-05 215008](https://user-images.githubusercontent.com/118679102/230143441-0a32852c-11e8-4469-bb2b-9e00a9bfa620.png)
![Screenshot 2023-04-05 215024](https://user-images.githubusercontent.com/118679102/230143649-fb4c2d6b-ca05-4069-bfc5-b9064e6001bd.png)
![Screenshot 2023-04-05 215041](https://user-images.githubusercontent.com/118679102/230143813-dca8bc19-9105-413c-93c3-3b8a8844f447.png)
![Screenshot 2023-04-05 215055](https://user-images.githubusercontent.com/118679102/230144647-af77f2db-c5ab-4c6c-87ef-4e67b41380f7.png)
![Screenshot 2023-04-05 215107](https://user-images.githubusercontent.com/118679102/230144781-10dd3a8e-256d-46ef-b83d-06027abd53c1.png)
![Screenshot 2023-04-05 215120](https://user-images.githubusercontent.com/118679102/230144870-c28e2764-4047-479a-acd0-33193c3c8e1e.png)
![Screenshot 2023-04-05 215138](https://user-images.githubusercontent.com/118679102/230144952-737f6ef1-d6f6-4550-a2c4-ebed18292f24.png)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
