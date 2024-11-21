from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
import matplotlib.pyplot as plt
iris = load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
c_dt = DecisionTreeClassifier(random_state=1)
c_dt.fit(x_train, y_train)

# Predict on the test set
y_pred = c_dt.predict(x_test)

# Print the accuracy
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Enter sample data for prediction
print("Enter the sample data:")
a = float(input("Enter sepal length in cm: "))
b = float(input("Enter sepal width in cm: "))
c = float(input("Enter petal length in cm: "))
d = float(input("Enter petal width in cm: "))
sample = [[a, b, c, d]]

# Predict class for the sample data
pred = c_dt.predict(sample)
pred_v = [iris.target_names[p] for p in pred]
print("Predicted class:", pred_v)
plt.figure(figsize=(12, 8))  # Adjust the figure size as needed
plot_tree(c_dt, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("Decision Tree Visualization")
plt.show()