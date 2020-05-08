import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import model

# ------------------ #
#    Data Prep       #
# ------------------ #

def data_distribution(df):
    '''
    Function to show the overall distribution for all the numberical features
    '''
    continuious = df.select_dtypes("number")
    f = plt.figure(figsize=(25,20))
    for i in range(continuious.shape[1]):
        f.add_subplot(5,5, i+1)
        sns.distplot(continuious.iloc[:,i], bins=5)
    plt.tight_layout()
    plt.show()

# ------------------ #
#        Viz         #
# ------------------ #

def log_error_cluster_groupings(df):
    '''
    Function to show relationships between features and logerror
    '''
    # We will begin by clustering the target variable - then comparing it to other variables to see if there is a pattern.
    df, centroid = model.create_cluster(df, 5, ["logerror"], "logerror_cluster")
    df.logerror_cluster.value_counts()
    continuious_df = df.select_dtypes("number")
    f = plt.figure(figsize=(25,20))
    for i in range(continuious_df.shape[1]):
        f.add_subplot(5,5, i+1)
        sns.barplot(data=df, x="logerror_cluster", y=continuious_df.iloc[:,i])
        plt.title(continuious_df.columns[i])
    plt.tight_layout()
    plt.show()

def log_error_relationship_clusters(df):
    features =  ["calculatedfinishedsquarefeet", "roomcnt", "taxvaluedollarcnt", "landtaxvaluedollarcnt", "taxamount", "age_home", "value_ratio"]
    f = plt.figure(figsize=(25,20))
    for i in range(len(features)):
        f.add_subplot(4,4, i+1)
        sns.barplot(data=df, x="logerror_cluster", y=features[i])
        plt.title(features[i])
    plt.tight_layout()
    plt.show()

def fips_relationships(df):
    '''
    Function to show relationships between features and fips
    '''
    # Is there a relationship between fips and the other categories that can be visualized by our logerror cluster?
    continuious_df = df.select_dtypes("number")
    f = plt.figure(figsize=(25,20))
    for i in range(20):
        f.add_subplot(5,5, i+1)
        sns.scatterplot(data=df, y=continuious_df.iloc[:,i], x="fips", hue="logerror_cluster")
    plt.tight_layout()
    plt.show()

def k_cluster_all(df, x, n):
    '''
    Function to create clusters based on specified series, and graphing against all other features to see if there is a relationship
    '''
    df = df.select_dtypes(exclude="object")
    kmeans = KMeans(n_clusters=n, random_state=123)
    kmeans.fit(df)
    df["cluster"] = kmeans.predict(df)
    df.cluster = 'cluster_' + (df.cluster + 1).astype('str')

    for col in df.columns:
        if col != x and col != "cluster":
            sns.relplot(data=df, x=x, y=col, hue='cluster', alpha=.3)
            plt.show()
    df.drop(columns="cluster", inplace=True)

def k_cluster_relationships(df, x, features, n):
    '''
    Function to create clusters based on specified series, and graphing against all other features to see if there is a relationship
    '''
    df = df.select_dtypes(exclude="object")
    kmeans = KMeans(n_clusters=n, random_state=123)
    kmeans.fit(df)
    df["cluster"] = kmeans.predict(df)
    df.cluster = 'cluster_' + (df.cluster + 1).astype('str')

    for i  in range(len(features)):   
        sns.relplot(data=df, x=x, y=features[i], hue='cluster', alpha=.3)
        plt.show()
    df.drop(columns="cluster", inplace=True)

def show_tax_location_distribution(df):
    '''
    Function used to show a scatter plot of the tax cluster, as well as the fips, for refernece
    '''
    f, axes = plt.subplots(1, 2, figsize=(15, 9))
    sns.scatterplot(data=df, y="latitude", x="longitude", hue="county", ax=axes[0])
    sns.scatterplot(data=df, y="latitude", x="longitude", hue="tax_location_cluster", ax=axes[1])

def tax_location_cluster_relationships(df):
    '''
    Function used to show the relationship between log error and the tax clusters
    '''
    continuious_df = df.select_dtypes("number")
    f = plt.figure(figsize=(25,20))
    for i in range(continuious_df.shape[1]-2):
        f.add_subplot(5,5, i+1)
        sns.scatterplot(data=df, x=continuious_df.iloc[:,i], y="logerror", hue="tax_location_cluster")
    plt.tight_layout()
    plt.show()


def compare_mse_scores(df):
    sns.barplot(x=df.index, y=df.values)
    plt.xticks(rotation=45, ha="right")
    plt.title("It's hard to beat the baseline")
    plt.ylabel("MSE Score")
    plt.show