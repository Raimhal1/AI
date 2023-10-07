import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
from sklearn import datasets

housing_data = datasets.load_boston()
X, Y = shuffle(housing_data.data, housing_data.target, random_state=7)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=7)

regressor = AdaBoostRegressor(DecisionTreeClassifier(max_depth=4), n_estimators=400, random_state=7)
regressor.fit(X_train, Y_train)

Y_train_pred = regressor.predict(X_train)
mse = mean_squared_error(Y_train, Y_train_pred)
evs = explained_variance_score(Y_train, Y_train_pred)
print("ADABOOST REGRESSOR")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))

feature_importance = regressor.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
feature_names = housing_data.feature_names

sorted_idx = np.flipud(np.argsort(feature_importance))
pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure()
plt.bar(pos, feature_importance[sorted_idx], align='center')
plt.xticks(pos, feature_names[sorted_idx])
plt.ylabel('Relative Importance')
plt.title('Variable Importance')
plt.show()