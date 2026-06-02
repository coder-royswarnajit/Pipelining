from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


def get_regression_models(random_state=42):

    models = {"Linear Regression": LinearRegression(),
              "SVR": SVR(kernel="rbf"),
              "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=random_state),
              "Decision Tree Regressor": DecisionTreeRegressor(random_state=random_state),
              "KNN Regressor": KNeighborsRegressor(n_neighbors=5)}

    return models