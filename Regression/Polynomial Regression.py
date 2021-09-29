import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# importing dataset
dataset = pd.read_excel('Folds5x2_pp.xlsx')

X = dataset.iloc[:, :-1].values  # feature variable
y = dataset.iloc[:, -1].values   # dependent variable

# splitting into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# creating regression model
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X_train)
regressor = LinearRegression()
regressor.fit(X_poly, y_train)
y_pred = regressor.predict(poly_reg.transform(X_test))

# testing accuracy
accuracy = r2_score(y_test, y_pred)
print(f'Accuracy of the Model is : {accuracy * 100: .2f} %', )
