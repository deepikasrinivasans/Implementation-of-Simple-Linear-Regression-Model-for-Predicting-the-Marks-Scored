# EX-02 Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

### AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas, numpy and sklearn.
2. Calculate the values for the training data set
3. Calculate the values for the test data set.
4. Plot the graph for both the data sets and calculate for MAE, MSE and RMSE.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: DEEPIKA S
RegisterNumber: 212222230028
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("/content/student_scores.csv")
df.head()
df.tail()
X,Y=df.iloc[:,:-1].values, df.iloc[:,1].values
print(X)
print(Y)
from sklearn.model_selection import train_test_split as tts
Xtrain,Xtest,Ytrain,Ytest=tts(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression as lr
reg=lr()
reg.fit(Xtrain,Ytrain)
Ypred=reg.predict(Xtest)
print(Ypred)
plt.scatter(Xtrain,Ytrain,color="orange")
plt.plot(Xtrain,reg.predict(Xtrain),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(Xtest,Ytest,color="blue")
plt.plot(Xtest,reg.predict(Xtest),color="green")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
print("MSE : ",mean_squared_error(Ytest,Ypred))
print("MAE : ",mean_absolute_error(Ytest,Ypred))
print("RMSE : ",np.sqrt(mse))
```
### Output:
<table>
<tr>
<td width=45%>

**df.head()** <br>
![m11](https://github.com/deepikasrinivasans/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393935/d8f284a1-ef94-4b77-ba1e-c2199280bf60)
**df.tail** <br>
![m112](https://github.com/deepikasrinivasans/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393935/4d64cf01-67c6-45a2-8a0f-46846e69be3b)
</td>
<td>

**X and Y values split** <br>
![m113](https://github.com/deepikasrinivasans/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393935/35c5d4ff-084b-44ed-a0c4-da4d6d8c3b26)
</td>
</tr>
</td>table>
<br>
<br>

![m14](https://github.com/deepikasrinivasans/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393935/c298cb99-6053-4870-abec-9f654cf3b366)
<br>
<br>
**Training and Testing set**<br>
![m15](https://github.com/deepikasrinivasans/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393935/28096130-1c01-4e5b-b47e-935fe6408e1e)
![m16](https://github.com/deepikasrinivasans/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393935/e304a41c-081a-4b1b-865a-7d3fc266cc3f)
<br>
**Values of MSE,MAE and RMSE**<br>
![m17](https://github.com/deepikasrinivasans/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393935/65f8ae10-281e-4d42-ae9e-2164fa501155)
### Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
