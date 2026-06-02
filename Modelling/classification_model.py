from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def get_classification_models(random_state=42):

    models = {"Logistic Regression": LogisticRegression(max_iter=1000,class_weight="balanced",random_state=random_state),
              "SVC": SVC(kernel="rbf",probability=True,class_weight="balanced",random_state=random_state),
              "Random Forest Classifier": RandomForestClassifier(n_estimators=100,class_weight="balanced",random_state=random_state),
              "Decision Tree Classifier": DecisionTreeClassifier(class_weight="balanced",random_state=random_state),
              "KNN Classifier": KNeighborsClassifier(n_neighbors=5)}

    return models