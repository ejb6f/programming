#1.1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df1 = pd.read_csv('car_data.csv')
print(df1.columns)
df1.head()

#1.2
df1.describe()
df1.isnull().sum().sum()
df1['dummy'] = 1
df1.loc[df1['Gender'] == 'Male', 'dummy'] = 0
X = df1.loc[:, ('Age','AnnualSalary')]
y = df1['Purchased']

#1.3
def maxMin(x):
  x = (x-min(x))/(max(x)-min(x))
  return x
X = X.apply(maxMin)

#1.4
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 2)

#1.5
k_bar = 30
k_grid = np.arange(1, k_bar)
SSE = np.zeros(k_bar)

for k in range(k_bar):
  model = KNeighborsClassifier(n_neighbors = k + 1)
  fitted_model = model.fit(X_train, y_train)
  y_hat = fitted_model.predict(X_test)
  SSE[k] = np.sum((y_test - y_hat) ** 2)

SSE_min = np.min(SSE)
min_index = np.where(SSE == SSE_min)
k_star = k_grid[min_index]
print(k_star)
plt.plot(np.arange(0, k_bar), SSE, label = 'Test')

plt.xlabel("k")
plt.ylabel("SSE")
plt.legend(loc='lower right')
plt.title('SSE')
plt.show()
print("The optimal k appears to be 6.")

#1.6
model = KNeighborsClassifier(n_neighbors = 6) 
fitted_model = model.fit(X_train,y_train)
y_hat = fitted_model.predict(X_test)
pd.crosstab(y_test, y_hat)
#print("The model shows that there were 15 times where zero is classified as a one and 6 times where one is classified as a zero. This makes the accuracy rate of the model 179/200 = 0.895, which although could be better, is still quite accurate.")

#1.7
cols2 = ['Age','AnnualSalary','Gender']
X = df1.loc[:, cols2]
y = df1['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2, random_state = 2)

model = KNeighborsClassifier(n_neighbors = 6)
fitted_model = model.fit(X_train.drop('Gender', axis = 1), y_train)
y_hat = fitted_model.predict(X_test.drop('Gender', axis = 1))

y_hat_M = y_hat[ X_test['Gender'] == 'Male']
y_hat_F = y_hat[ X_test['Gender'] == 'Female']
y_M = y_test[ X_test['Gender'] == 'Male']
y_F = y_test[ X_test['Gender'] == 'Female']

pd.crosstab(y_F, y_hat_F)
pd.crosstab(y_M, y_hat_M)
#print("For men, about 80/102 == 0.784 are corret. For women, about 78/98 = 0.796" are correct. The model produces roughly the same accuracy for both sex in its approximation, with a difference in accuracy of 0.012.)
