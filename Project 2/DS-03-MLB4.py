# MAJOR PROJECT "DS-03-MLB4".
# Predicting the price of ford cars based on the previous data.

from random import random
from cv2 import rotate
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Major Project/Csv_Datasets/ford.csv') # Loading the csv file for reading purpose


# E.D.A from here On : -
# Missing values 
print('\nThe Missing Value data is as following >>')
print(df.isnull().sum()) # We used isnull() and sum to get total number of missing values in respective columns.

# No Cleaning required as there are no missing values.

# Since we need to predict the Price, we don't need some extra data columns.
df = df.drop(['mpg'],axis=1) # Since mpg is same as mileage

# Creating Data cleaner function
def cleaning_outlier(data,x):
    Q1 = data[x].quantile(0.25)
    Q2 = data[x].quantile(0.75)
    IQR = Q2 - Q1
    data = data[~((data[x] < (Q1 - 1.5*IQR)) | (data[x] > (Q2 - 1.5*IQR)))]
    return data

df = cleaning_outlier(df,'price') # Cleaning price from outliers ( Detached values )

# Removing Duplicated Data, If Any.
df.drop_duplicates(inplace=True)

# Visualization of Data based on car model.
# Car Model column visualisation
sns.catplot(x='model',y='price',data=df,height=5,aspect=2).set(title='Car Model VS Price Graph')
plt.xticks(rotation=90)
plt.show()

# Year and Price column Visualisation
sns.catplot(x='year',y='price',data=df,height=5,aspect=2).set(title='Year VS Price Graph')
plt.xticks(rotation=90)
plt.show()

# Engine type and Price column
sns.catplot(x='engineSize',y='price',data=df,height=5,aspect=2).set(title='Engine type VS Price Graph')
plt.xticks(rotation=90)
plt.show()

# Looking for a correlation between the column data
plt.figure(figsize=(17,15))
correlation_mask = np.triu(df.corr())
h_map = sns.heatmap(df.corr(),
    mask = correlation_mask, annot=True,
    cmap='Oranges').set(title='CORRELATION Graph')
plt.yticks(rotation=360)
plt.show()


#----------------------------------- Label Encoding on The Dataset -----------------------------------------

print("\nLabel Encoding on The Dataset")

model = pd.get_dummies(df['model'], drop_first=True)
engine_fuel_type = pd.get_dummies(df['fuelType'], drop_first=True)
transmission_type = pd.get_dummies(df['transmission'], drop_first=True)

df = df.drop(['model',
              'fuelType',
              'transmission'], axis=1)

df = pd.concat([model,
                engine_fuel_type,
                transmission_type,
                df], axis=1)

# Dividing into Input and Output or Separating between Features and Lables
# Feature is input or descriptive attribute
# Lablel is what we predict or Output.

X = df.drop('price',axis=1) # Only feature
Y = df['price'].astype(int) # Only Lable

# Scalling of Data
print(df.info()) # To get information about the data.
print('\nThe following Dataset contains (Rows , Columns) >> ',df.shape) # Number of Rows and Columns in our Data.
print('\nThe Unique Data is: ')
print(df.nunique()) # Printing information on Unique Values.

# Training, Testing Data
from sklearn.model_selection import train_test_split

X_train_full, X_test, Y_train_full, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

X_train, X_val, Y_train, Y_val = train_test_split(X_train_full, Y_train_full, test_size=0.2, random_state=42)


# Building a Machine Learning Model
# We will check some of them here and according to there prediction
# we wil choose the best.

#-------------------------------------------- LINEAR REGRESSION -----------------------------------------
print('-------------------LINEAR REGRESSION Results------------------')
from sklearn.linear_model import LinearRegression
model_linreg = LinearRegression()
model_linreg = model_linreg.fit(X_train, Y_train)
Y_pred_linreg = model_linreg.predict(X_test)

# Evaluating
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Mean Squared Error (MSE)
mse = mean_squared_error(Y_test, Y_pred_linreg)
print('\nMean squared error dari Testing Set:', round(mse))

# Mean Absolute Error (MAE)
mae = mean_absolute_error(Y_test, Y_pred_linreg)
print('Mean absolute error dari Testing Set:', round(mae))

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print('Root Mean Squared Error dari Testing Set:', round(rmse))


#------------------------------------------------ DECISION TREE Methode --------------------------------
print('-------------------DECISION TREE Results------------------')
from sklearn.tree import DecisionTreeRegressor
model_dtr = DecisionTreeRegressor(random_state=42)
model_dtr = model_dtr.fit(X_train, Y_train)
Y_pred_dtr = model_dtr.predict(X_test)

# Evaluating
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Mean Squared Error
mse = mean_squared_error(Y_test, Y_pred_dtr)
print('\nMean squared error dari Testing Set:', round(mse))

# Mean Absolute Error
mae = mean_absolute_error(Y_test, Y_pred_dtr)
print('Mean absolute error dari Testing Set:', round(mae))

# Root Mean Squared Error
rmse = np.sqrt(mse)
print('Root Mean Squared Error dari Testing Set:', round(rmse))


#------------------------------------------- RANDOM FOREST -------------------------------------------
# Building a Machine Learning Model RandomForest
print('-------------------RANDOM FOREST Results------------------')
from sklearn.ensemble import RandomForestRegressor
model_rfr = RandomForestRegressor(random_state=42)
model_rfr = model_rfr.fit(X_train, Y_train)
Y_pred_rfr = model_rfr.predict(X_test)

# Evaluating The Machine Learning Model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Mean Squared Error
mse = mean_squared_error(Y_test, Y_pred_rfr)
print('\nMean squared error dari Testing Set:', round(mse))

# Mean Absolute Error
mae = mean_absolute_error(Y_test, Y_pred_rfr)
print('Mean absolute error dari Testing Set:', round(mae))

# Root Mean Squared Error
rmse = np.sqrt(mse)
print('Root Mean Squared Error dari Testing Set:', round(rmse))


# ------------------------------------------------LASSO-------------------------------------------------
# Build a Machine Learning Model Lasso
print('-------------------LASSO Results------------------')
from sklearn.linear_model import Lasso
model_lasso = Lasso(random_state=42)
model_lasso = model_lasso.fit(X_train, Y_train)
Y_pred_lasso = model_lasso.predict(X_test)

#Evaluating The Machine Learning Model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

#Mean Squared Error
mse = mean_squared_error(Y_test, Y_pred_lasso)
print('\nMean squared error dari Testing Set:', round(mse))

#Mean Absolute Error
mae = mean_absolute_error(Y_test, Y_pred_lasso)
print('Mean absolute error dari Testing Set:', round(mae))

#Root Mean Squared Error
rmse = np.sqrt(mse)
print('Root Mean Squared Error dari Testing Set:', round(rmse))


#---------------------------------------------------- RIDGE --------------------------------------------
# Building a Machine Learning Model Ridge
print('-------------------RIDGE Results------------------')
from sklearn.linear_model import Ridge
model_ridge = Ridge(random_state=42)
model_ridge = model_ridge.fit(X_train, Y_train)
Y_pred_ridge = model_ridge.predict(X_test)

#Evaluating The Machine Learning Model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

#Mean Squared Error
mse = mean_squared_error(Y_test, Y_pred_ridge)
print('\nMean squared error dari Testing Set:', round(mse))

#Mean Absolute Error
mae = mean_absolute_error(Y_test, Y_pred_ridge)
print('Mean absolute error dari Testing Set:', round(mae))

#Root Mean Squared Error
rmse = np.sqrt(mse)
print('Root Mean Squared Error dari Testing Set:', round(rmse))

# -------------------------------------------- Evaluating all the Models-----------------------------------
print('From the results We concluded that the Machine Learning model using Linear Regression has the smallest RMSE')
print('RMSE >> Root Mean Square Error')

# ----------------------------------------Model Fitting or Visualisation of the Model----------------------

for i in range(5):
    real = Y_val.iloc[i]
    pred = int(model_linreg.predict(X_val.iloc[i].to_frame().T)[0])
    print(f'Real Value      ----->>>>> {real} $\n'
          f'Predicted Value ----->>>>> {pred} $')
    print()

#------------------------------------------ PREDICTING THE OUTPUT -----------------------------------------
# Visualize The Diffrence Between Real and Predicted in the Machine Learning Model

fig = plt.figure(figsize=(17, 10))
df = df.sort_values(by=['price'])
X = df.drop('price', axis=1)
Y = df['price']
plt.title('Real VS Predicted Price Comparision using Graph')
plt.scatter(range(X.shape[0]), Y, color='red', label='Real')
plt.scatter(range(X.shape[0]), model_linreg.predict(X), marker='.', label='Predict')
plt.legend(loc='best', prop={'size': 8})
plt.show()