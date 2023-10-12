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

#2.1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

df2 = pd.read_csv('USA_cars_datasets.csv')
df2 = df2.loc[: , ['price', 'year', 'mileage']]
df2.isna().sum().sum()
#print("There are no NA values")

#2.2
A = df2.loc[:, ('year', 'mileage')]
b = df2['price']

def maxMin(x):
  x = (x-min(x))/(max(x)-min(x))
  return x

A = A.apply(maxMin)

#2.3
X_train, X_test, y_train, y_test = train_test_split(A, b, test_size = .2, random_state = 1)

#2.4
k_values = [3, 10, 25, 50, 100, 300]
SSE_list = []
predicted_values = []

for k in k_values:
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)
    
    y_hat = model.predict(X_test)
    
    SSE = np.sum((y_test - y_hat) ** 2)
    SSE_list.append(SSE)
    
    predicted_values.append(y_hat)

for k, y_pred in zip(k_values, predicted_values):
    plt.figure()
    plt.scatter(y_test, y_pred)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--k')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Value")
    plt.title("Actual Price vs. Predicted Price")
    plt.show()

#6.1
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def createData(noise,N=50):
    np.random.seed(100) # Set the seed for replicability
    # Generate (x1,x2,g) triples:
    X1 = np.array([np.random.normal(1,noise,N),np.random.normal(1,noise,N)])
    X2 = np.array([np.random.normal(3,noise,N),np.random.normal(2,noise,N)])
    X3 = np.array([np.random.normal(5,noise,N),np.random.normal(3,noise,N)])
    # Concatenate into one data frame
    gdf1 = pd.DataFrame({'x1':X1[0,:],'x2':X1[1,:],'group':'a'})
    gdf2 = pd.DataFrame({'x1':X2[0,:],'x2':X2[1,:],'group':'b'})
    gdf3 = pd.DataFrame({'x1':X3[0,:],'x2':X3[1,:],'group':'c'})
    df = pd.concat([gdf1,gdf2,gdf3],axis=0)
    return df

df0_125 = createData(0.125)
df0_25 = createData(0.25)
df0_5 = createData(0.5)
df1_0 = createData(1.0)
df2_0 = createData(2.0)

datasets = [df0_125, df0_25, df0_5, df1_0, df2_0]
noise_levels = [0.125, 0.25, 0.5, 1.0, 2.0]

#6.2
#sns.scatterplot(data = df0_125, x = 'x1',y='x2',hue='group',style='group')
#sns.scatterplot(data = df0_25, x = 'x1',y='x2',hue='group',style='group')
#sns.scatterplot(data = df0_5, x = 'x1',y='x2',hue='group',style='group')
#sns.scatterplot(data = df1_0, x = 'x1',y='x2',hue='group',style='group')
#sns.scatterplot(data = df2_0, x = 'x1',y='x2',hue='group',style='group')
#print('As the noise goes up, the clusters diffuse and therefore collide. This happens until noise reached a level of 2.0, in which it becomes more difficult to idenitfy the clusters, as their distinctness decreases.')

#6.3
def maxMin(x):
  x = (x-min(x))/(max(x)-min(x))
  return x

def scree(data): 
  X = data.loc[ : , ['x1','x2']]
  X = X.apply(maxMin)
  k_bar = 15
  k_grid = np.arange(1,k_bar+1)
  SSE = np.zeros(k_bar)
  for k in range(k_bar):
    model = KMeans(n_clusters = k + 1, max_iter = 300, n_init = 10, random_state = 0)
    model = model.fit(X)
    SSE[k] = model.inertia_
  scree_plot, axes = plt.subplots()
  sns.lineplot(x=k_grid, y=SSE).set_title('Scree Plot')
  axes.set_ylim(0, 35)

scree(data = df0_125)
scree(data = df0_25)
scree(data = df0_5)
scree(df1_0)
scree(df2_0)
print('The existence of the elbow becomes less evident with the increasing of the noise')

#6.4
print("The scree plot approach is efficient for clusters that are separated and distinctive, as it produces a visible elbow. However, this approach would fail to be successful in cases where the clusters are not distinctive or apart from one another, as their lack of differentiation would lead to the spree chart's elbow to become unnoticeable and smooth. Accordingly, as the noise goes up, the clusters spread out, and the elbows become less distinct. Thus, The overlap between the groups therefore reuslt make it difficult to ifentify how many groups to pick or which points belong to what group.")
