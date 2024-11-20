from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

iris=load_iris()
x=iris.data
#print(x)
y=iris.target
#print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
c_knn = KNeighborsClassifier(n_neighbors=3)
c_knn.fit(x_train,y_train)
y_pred = c_knn.predict(x_test)

print("accuracy",metrics.accuracy_score(y_test,y_pred))
print("enter sample data: ")
a = int(input("enter sepal length in cm : "))
b = int(input("enter sepal width in cm : "))
c = int(input("enter petal length in cm : "))
d = int(input("enter petal width in cm : "))

sample = [[a,b,c,d]]

pred = c_knn.predict(sample)
pred_v = [iris.target_names[p] for p in pred]
print(pred_v)