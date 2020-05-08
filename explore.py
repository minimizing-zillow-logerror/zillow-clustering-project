import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import model
from scipy import stats

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
    plt.suptitle("How is the data generally distributed?")
    plt.show()

# ------------------ #
#        Stats       #
# ------------------ #


def ttail_t_test(zillow, county_1, county_2):
    alpha = 0.05
    subgroup_1 = zillow[zillow.county == county_1].logerror
    subgroup_2 = zillow[zillow.county == county_2].logerror
    
    tstats, p = stats.ttest_ind(subgroup_1, subgroup_2)
    
    reject = "Because p is less than 0.05, we reject the null hypothesis"
    fail_reject = "Because p is more than 0.05, we fail to reject the null hypothesis"

    if p < alpha:
        print(f'''
            {reject}
            The mean_logerror for LA = {subgroup_1.mean():.2f}
            The mean_logerror for Orange County = {subgroup_2.mean():.2}
            ''')
    else:
        print(f'''
            {fail_reject}
            The mean_logerror for LA = {subgroup_1.mean():.2f}
            The mean_logerror for Orange County = {subgroup_2.mean():.2}
            ''')



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
    plt.suptitle("Do we see any groupings that have variance in logerror?")
    plt.show()

def log_error_relationship_clusters(df):
    features =  ["calculatedfinishedsquarefeet", "roomcnt", "taxvaluedollarcnt", "landtaxvaluedollarcnt", "taxamount", "age_home", "value_ratio"]
    f = plt.figure(figsize=(25,20))
    for i in range(len(features)):
        f.add_subplot(4,4, i+1)
        sns.barplot(data=df, x="logerror_cluster", y=features[i])
        plt.title(features[i])
    plt.tight_layout()
    plt.suptitle("Groupings with the most variance in logerror")
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
        plt.title("What featues have strong, or weak, relationships?")
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
    plt.title("How does our model compare to the baseline?")
    plt.ylabel("MSE Score")
    plt.show

def viz_location_tax_rate(zillow):
    f, axes = plt.subplots(1, 2, figsize=(15, 9))
    sns.scatterplot(data=zillow, y="latitude", x="longitude", hue="county", ax=axes[0])
    sns.scatterplot(data=zillow, y="latitude", x="longitude", hue="tax_location_cluster", ax=axes[1])
    zillow.plot.scatter(y='latitude_centroid', x='longitude_centroid', c='black', marker='x', s=1000,  label='centroid', ax=axes[1])
    plt.suptitle("How does our clusters compare to the county's?")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()