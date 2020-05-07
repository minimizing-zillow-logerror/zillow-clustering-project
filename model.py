from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# ------------------ #
#    Clustering      #
# ------------------ #


# Elbow code
def create_elbow_graph(df, col):
    X = df[col]
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(9, 6))
        pd.Series({k: KMeans(k).fit(X).inertia_ for k in range(2, 12)}).plot(marker='x')
        plt.xticks(range(2, 12))
        plt.xlabel('k')
        plt.ylabel('inertia')
        plt.title('Change in inertia as k increases')

# Cluster
def create_cluster(df, n_cluster, col, cluster_name):
    X = df[col]
    kmeans = KMeans(n_clusters=n_cluster).fit(X)
    centroid = kmeans.cluster_centers_
    df[cluster_name] = kmeans.predict(X)
    df[cluster_name] = 'cluster_' + (df[cluster_name] + 1).astype('str')
    return df, centroid

# ------------------ #
#    Regression      #
# ------------------ #

# Linear Regression Model
def run_lm(X_train, y_train):
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    y_pred = lm.predict(X_train)
    return lm, y_pred


# Random Forest Regression

def run_rf(X_train, y_train, n_estimators, max_depth):
    rf = RandomForestRegressor(n_estimators = n_estimators, random_state = 123, max_depth = max_depth)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_train)
    return rf, y_pred