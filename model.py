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
    '''
    Function to draw and elbow graph. Takes a dataframe and the name of a column to create a cluster
    '''
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
    '''
    Function to cluster features. Takes a dataframe, the number of clusters and the columns to cluster by
    '''
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
    '''
    Function to run linear regression models. Return the model and the predction for x
    '''
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    y_pred = lm.predict(X_train)
    return lm, y_pred

# print points

def print_predicted_vs_actual(predictions, model, label):
    '''
    Function used to draw plot of actual vs predicted values
    '''
    plt.figure(figsize=(9, 9))

    plt.scatter(predictions.actual, predictions[model], label="label", marker='o')
    plt.scatter(predictions.actual, predictions.baseline, label=r'Baseline ($\hat{y} = \bar{y}$)', marker='o')
    plt.plot(
        [predictions.actual.min(), predictions.actual.max()],
        [predictions.actual.min(), predictions.actual.max()],
        ls=':',
        label='perfect prediction',
        c='grey'
    )

    plt.legend(title='Model')
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.title('Predicted vs Actual')

    plt.show()

# Random Forest Regression

def run_rf(X_train, y_train, n_estimators, max_depth):
    '''
    Function to run random forest regressor model
    '''
    rf = RandomForestRegressor(n_estimators = n_estimators, random_state = 123, max_depth = max_depth)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_train)
    return rf, y_pred

def get_feature_importance(model):
    '''
    Function to return coeficients, as well their importance. Only works for random forest model.
    '''
    features = model.feature_importances_
    return features

def visualize_feature_importance(features, model):
    feature_importance = pd.DataFrame({"features": features, "feature_importance": model.feature_importances_})
    sns.barplot(data=feature_importance, x="features", y="feature_importance")
    plt.xticks(rotation = 45, ha="right")
    plt.title("What features have the most influence?")