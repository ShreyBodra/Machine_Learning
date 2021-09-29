import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# importing dataset
dataset = pd.read_excel('Folds5x2_pp.xlsx')
print(dataset.head())

X = dataset.iloc[:, :-1].values  # feature variable
y = dataset.iloc[:, -1].values   # dependent variable

# splitting into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# creating regression model
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


# testing accuracy
accuracy = r2_score(y_test, y_pred)
print(f'Accuracy of the Model is : {accuracy * 100: .2f} %', )
