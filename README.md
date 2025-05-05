# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. **Load Data**: Read "salary.csv" dataset
2. **Preprocess**:
   - Check dataset info and null values
   - Encode "Position" column using `LabelEncoder`
3. **Feature Selection**:
   - Features (`x`): Position (encoded) and Level
   - Target (`y`): Salary
4. **Split Data**: 60% training, 40% testing (note reversed variables)
5. **Train Model**: Decision Tree Regressor
6. **Evaluate**:
   - Calculate Mean Squared Error (MSE)
   - Calculate R² Score
7. **Predict**: Sample prediction for Level 6 Position 5

## Program:

```py
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: 
RegisterNumber:  
*/
```

```py
#Name: S Rajath
#Reg No: 212224240127
import pandas as pd

data = pd.read_csv("salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()

x = data[["Position", "Level"]]
y = data[["Salary"]]

from sklearn.model_selection import train_test_split
X_test, X_train, Y_test, Y_train = train_test_split(x, y, test_size = 0.4, random_state = 4)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(X_train, Y_train)
Y_pred = dt.predict(X_test)

from sklearn import metrics
mse = metrics.mean_squared_error(Y_test, Y_pred)
mse

r2 = metrics.r2_score(Y_test, Y_pred)
r2

dt.predict([[5, 6]])
```

## Output:

### Dataset Preview
![image](https://github.com/user-attachments/assets/4d199b1c-3482-41e2-bb0a-24946776c421)


### df.info()
![image](https://github.com/user-attachments/assets/9c489899-a260-44b9-8bb9-d07561c34031)

### Value of df.isnull().sum()
![image](https://github.com/user-attachments/assets/9aeb1acf-47c0-4435-80de-c3c7b2c70c84)

### Data after encoding calculating Mean Squared Error
![image](https://github.com/user-attachments/assets/7d087205-a3f1-4838-83d0-c926718bf5d1)

### MSE
![image](https://github.com/user-attachments/assets/d4182442-d32d-4fd0-9532-5b088fc6f0c9)

### R2
![image](https://github.com/user-attachments/assets/d561c002-b78a-4fab-aa02-115d5eb2f018)

### Model prediction with [5,6] as input
![image](https://github.com/user-attachments/assets/9046c0d1-137c-4d84-b7e9-add19dafd297)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
