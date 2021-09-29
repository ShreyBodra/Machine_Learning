import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# loading dataset
data = pd.read_csv("Mall_Customers.csv")
print(data.head())

x = data.iloc[:,[3,4]].values

# elbow method to find number of clusters
# wcss => within cluster sum of squared distance
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init="k-means++" , random_state = 42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


plt.scatter(range(1,11),wcss,color="red")
plt.plot(range(1,11),wcss, color="blue")
plt.title("The elbow method")
plt.xlabel("No. of cluster")
plt.ylabel("WCSS")
plt.show()

# building model
kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42)
y_pred = kmeans.fit_predict(x)

# visualizing clusters
plt.scatter(x[y_pred==0,0],x[y_pred==0,1], s=100,color="red" , label='cluster1')
plt.scatter(x[y_pred==1,0],x[y_pred==1,1], s=100,color='blue', label="cluster2")
plt.scatter(x[y_pred==2,0],x[y_pred==2,1], s=100,color='green', label='cluster3' )
plt.scatter(x[y_pred==3,0],x[y_pred==3,1], s=100,color='yellow', label = 'cluster4')
plt.scatter(x[y_pred==4,0],x[y_pred==4,1] , s=100,color='black', label='cluster5')
plt.legend()
plt.show()

# adding predicted category to dataset
data['category'] = y_pred
print(data.head())