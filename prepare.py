from env import host, user, password
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler
import model
import prepare


# ~~~~~~~~~ Acquire ~~~~~~~~~~ #

# ----------------- #
#     Read SQL      #
# ----------------- #

query = '''

SELECT *
FROM properties_2017
JOIN (
	SELECT parcelid, `logerror`, max(transactiondate)
	FROM predictions_2017
	GROUP BY parcelid, logerror) predictions_2017 USING (parcelid)
LEFT JOIN `typeconstructiontype` USING (typeconstructiontypeid)
LEFT JOIN propertylandusetype USING (propertylandusetypeid)
LEFT JOIN airconditioningtype USING (airconditioningtypeid)
LEFT JOIN architecturalstyletype USING (architecturalstyletypeid)
LEFT JOIN buildingclasstype USING (buildingclasstypeid)
LEFT JOIN `heatingorsystemtype` USING (`heatingorsystemtypeid`)
LEFT JOIN storytype USING (`storytypeid`)
WHERE latitude IS NOT NULL AND longitude IS NOT NULL
;

'''

data_base_name = "zillow"

def sql_database(data_base_name, query):
    '''
    Query to read data from SQL server
    '''
    global host
    global user
    global password
    url = f'mysql+pymysql://{user}:{password}@{host}/{data_base_name}'
    df = pd.read_sql(query, url)
    return df

def run_query_to_csv():
    '''
    Helper function to store a dataframe as a csv.
    '''
    df = sql_database(data_base_name, query)
    df.to_csv("zillow_data.csv")

def read_zillow():
    '''
    Helper function to read zillow csv. It also drops the "Unnamed" column
    '''
    df = pd.read_csv("zillow_data.csv")
    df.drop(columns= "Unnamed: 0", inplace=True)
    return df

def wrangle_geo_data(df):
    '''
     Helper function used to add county and state data to the df
    '''
    data = [["CA", "Los Angeles", 6037], ["CA", "Orange County", 6059], ["CA", "Ventura County", 6111]]
    fips = pd.DataFrame(data, columns= ["state", "county", "fips"])
    df.fips = df.fips.astype(int)
    geo_data = df.merge(fips, left_on="fips", right_on="fips")
    return geo_data


# ~~~~~~~~~ Prep ~~~~~~~~~ #


def change_dtypes(df, col, type):
    ''' 
    Helper function used to changed data types
    '''
    df = df[col].astype(type)
    return df

def drop_null_col(df, ptc=.5):
    ''' 
    Helper function used drop columns with a number of high threshold nulls. Parameter is set to .5
    '''
    df = df.dropna(axis =1, thresh=(df.shape[0] * ptc))
    return df

# ------------------------- #
#    Find Missing Values    # 
# ------------------------- #


def impude_unit_cnt(df):
    '''
    Helper function used to impude unit count by using the propertylandusedesc
    '''
    if df.propertylandusedesc == "Condominium" or df.propertylandusedesc == "Single Family Residential":
        return 1
    else:
        return 0


def impude_values(zillow):
    '''
    Helepr function used to impude some missing values. Most values are impuded either with the mean or the mode
    '''
    zillow.lotsizesquarefeet = zillow.lotsizesquarefeet.fillna(zillow.lotsizesquarefeet.median())

    # heatingorsystemdesc Filled na with "none"
    zillow.heatingorsystemdesc = zillow.heatingorsystemdesc.fillna("None")
    

    # buildingqualitytypeid Filled na with mode = 8.0
    zillow.buildingqualitytypeid = zillow.buildingqualitytypeid.fillna(8.0)

    # Drop that aren't single units
    zillow.unitcnt = zillow.unitcnt.fillna(zillow.apply(lambda col: impude_unit_cnt(col), axis = 1))
    zillow = zillow[zillow.unitcnt == 1]

    zillow.finishedsquarefeet12 = zillow.finishedsquarefeet12.fillna(zillow.finishedsquarefeet12.median())

    # regionidcity - will use the most common region city id to replace the missing values

    zillow.regionidcity = zillow.regionidcity.fillna(zillow.regionidcity.mode()[0])

    #censustractandblock - place with the mode

    zillow.censustractandblock = zillow.censustractandblock.fillna(zillow.censustractandblock.mode()[0])

    zillow.calculatedfinishedsquarefeet = zillow.calculatedfinishedsquarefeet.fillna(zillow.calculatedfinishedsquarefeet.median())

    zillow.yearbuilt = zillow.yearbuilt.fillna(zillow.yearbuilt.mode()[0])

    return zillow

# ------------------- #
# Dectecting Outliers #
# ------------------- #


def get_upper_outliers_iqr(s, k):
    '''
    Given a series and a cutoff value, k, returns the upper outliers for the
    series.

    The values returned will be either 0 (if the point is not an outlier), or a
    number that indicates how far away from the upper bound the observation is.
    '''
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    upper_bound = q3 + k * iqr
    return s.apply(lambda x: max([x - upper_bound, 0]))

def outliers_z_score(ys):
    '''
    Function used to detect outliers using z_score
    '''
    threshold = 3

    mean_y = np.mean(ys)
    stdev_y = np.std(ys)
    z_scores = [(y - mean_y) / stdev_y for y in ys]
    return np.where(np.abs(z_scores) > threshold)

def outliers_percentile(s):
    '''
    Function used to detect outliers using percentiles
    '''
    return s > s.quantile(.99)

def detect_outliers(s, k, method="iqr"):
    ''' 
    Main function to detect outliers. Takes a series, a value for k, and a method for detecting outliers. Standard method for detecting outliers is IQR
    '''
    if method == "iqr":
        upper_bound = get_upper_outliers_iqr(s, k)
        return upper_bound
    elif method == "z_score":
        z_score = outliers_z_score(s)
        return z_score
    elif method == "percentile":
        percentile = outliers_percentile(s)
        return percentile
    
def detect_columns_outliers(df, k, method="iqr"):
    '''
    Function used to detect outliers across the entire dataframe
    '''
    outlier = pd.DataFrame()
    for col in df.select_dtypes("number"):
        is_outlier = detect_outliers(df[col], k, method=method)
        outlier[col] = is_outlier
    return outlier

def drop_outliers(zillow, k, method="iqr"):
    '''
    Function used to drop outliers
    '''
    # outliers = detect_columns_outliers(zillow, k, method=method)
    # zillow = zillow.drop(outliers.lotsizesquarefeet[outliers.lotsizesquarefeet > 10].dropna().index)
    # outliers = detect_columns_outliers(zillow, k, method=method)
    # zillow = zillow.drop(outliers.taxamount[outliers.taxamount > 10].dropna().index)

    zillow = zillow[zillow.lotsizesquarefeet < zillow.lotsizesquarefeet.quantile(.90)]
    zillow = zillow[zillow.taxamount < zillow.taxamount.quantile(.90)]
    zillow = zillow[zillow.taxvaluedollarcnt < zillow.taxvaluedollarcnt.quantile(.90)]

    return zillow

# __ MAIN PREP FUNCTION__ # 

def wrangle_zillow():
    '''
    Main function for prep. It reads the data, impudes relevant missing values and drops colums and rows with too many null values. Also adds geo data (county and state)
    '''
    zillow = pd.read_csv("zillow_data.csv")
    col_drop = ["propertyzoningdesc", "calculatedbathnbr", "fullbathcnt", "Unnamed: 0"]
    zillow.drop(columns = col_drop, inplace=True)
    

    zillow = drop_null_col(zillow)   
    

    col_obj = ["heatingorsystemtypeid", "parcelid", "id", "fips", "latitude", "longitude", "yearbuilt", "assessmentyear", "censustractandblock", "regionidcity", "regionidzip", "regionidcounty", "propertylandusetypeid"]

    zillow[col_obj] = change_dtypes(zillow, col_obj, "object")

    zillow = impude_values(zillow)


    zillow = drop_outliers(zillow, 3)

    zillow = zillow.drop(columns="heatingorsystemtypeid")

    zillow = zillow.dropna()

    zillow = wrangle_geo_data(zillow)

    return zillow


# ------------ #
#   Scaling    #
# ------------ #   


# ~~~~~ Before Splitting ~~~~~ #

# Helper function used to updated the scaled arrays and transform them into usable dataframes
def return_values_explore(scaler, df):
    ''' 
    Function used to scale data. Because of the nature of the project. We needed to scale a dataframe.
    '''
    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns.values).set_index([df.index.values])
    return scaler, df_scaled

# Linear scaler
def min_max_scaler_explore(df):
    '''
    Function used to scale data. Because of the nature of the project. We needed to scale a dataframe.
    '''
    scaler = MinMaxScaler().fit(df)
    scaler, df = return_values_explore(scaler, df)
    return scaler, df

def scale_data_for_exploration(zillow):
    
    df_scaling = zillow.select_dtypes(exclude="object")

    scaler, df_scaled = prepare.min_max_scaler_explore(df_scaling)

    df_object = zillow.select_dtypes("object")

    zillow = pd.concat([df_object, df_scaled], axis=1)


# ~~~~~ After Splitting ~~~~~ #

# Helper function used to updated the scaled arrays and transform them into usable dataframes
def return_values(scaler, train, validate, test):
    '''
    Helper function used to updated the scaled arrays and transform them into usable dataframes
    '''
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    validate_scaled = pd.DataFrame(scaler.transform(validate), columns=validate.columns.values).set_index([validate.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, validate_scaled, test_scaled

# Linear scaler
def min_max_scaler(train,validate, test):
    '''
    Helper function that scales that data. Returns scaler, as well as the scaled dataframes
    '''
    scaler = MinMaxScaler().fit(test)
    scaler, train_scaled, validate_scaled, test_scaled = return_values(scaler, train, validate, test)
    return scaler, train_scaled, validate_scaled, test_scaled

#---------------------#
#       Splitting     #
#---------------------#

def split_data(df):
    '''
    Main function to split data into train, validate, and test datasets. Random_state == 123, train_size = .8
    '''
    train, test = train_test_split(df, random_state =123, train_size=.8)
    train, validate = train_test_split(train, random_state=123, train_size=.75)
    return train, validate, test

#---------------------#
#   Preprocessing     #
#---------------------#

def engineer_features(zillow):
    '''
    Helper function used to create engieneered features
    '''
    # Create a new "age_home"
    zillow["age_home"] = 2017 - zillow.yearbuilt
    # total_property_size (combine calculated square feet of the house plus the size of the lot)
    zillow["total_size"] = zillow.finishedsquarefeet12 + zillow.lotsizesquarefeet
    # value_ratio = qhat is the rate between home tax and property tax
    zillow["value_ratio"] = zillow.landtaxvaluedollarcnt / zillow.taxamount
    # What the tax rate is 
    zillow["tax_rate"] = zillow.taxamount / zillow.taxvaluedollarcnt
    return zillow

def create_cluster_centers_tax_location(zillow):
    '''
    Helper function used to create clusters and add the centroids to the dataframe for modeling
    '''
    zillow, centroid = model.create_cluster(zillow, 8, ["tax_rate", "latitude", "longitude" ], "tax_location_cluster")
    
    centroid_2 = pd.DataFrame({"tax_location_cluster": zillow.tax_location_cluster.unique(), "tax_rate": centroid[:,0], "latitude": centroid[:,1], "longitude": centroid[:,2]})

    zillow = zillow.merge(centroid_2, how='left', on="tax_location_cluster", suffixes=("", "_centroid")).set_index(zillow.index)

    return zillow

def create_cluster_center_size(zillow):
    '''
    Helper function used to create clusters and add the centroids to the dataframe for modeling
    '''
    
    zillow, centroid = model.create_cluster(zillow, 4, ["total_size"], "total_size_cluster")

    centroid_3 = pd.DataFrame({"total_size_cluster":["cluster_1", "cluster_2", "cluster_3","cluster_4"], "total_size_mean": centroid[:,0], })

    zillow = zillow.merge(centroid_3, how='left', on="total_size_cluster", suffixes=("", "_centroid")).set_index(zillow.index)

    return zillow 



def prepare_for_modeling(zillow, features=[]):
    '''
    Main preprocessing function to use for modeling 
    '''
    # Acquire and Prep
    zillow = wrangle_zillow()
    
    # Feature Engineering
    zillow = engineer_features(zillow)

    # Clusterineering
    zillow = create_cluster_centers_tax_location(zillow)
    zillow = create_cluster_center_size(zillow)
    
    # Split
    train, validate, test = split_data(zillow)

    X_train = train[features]
    y_train = train.logerror
    X_validate = validate[features]
    y_validate = validate.logerror
    X_test = test[features]
    y_test = test.logerror

    # scaling

    scaler, train_scaled, validate_scaled, test_scaled = min_max_scaler(X_train, X_validate, X_test)

    return train_scaled, y_train, validate_scaled, y_validate, test_scaled, y_test



def prepare_for_modeling_county(zillow, county, features=[]):
    '''
    Special function used to prepare data for modeling by county 
    '''
    # Acquire and Prep
    zillow = wrangle_zillow()

    #apply county filter
    zillow = zillow[zillow.county == county]
    
    # Feature Engineering
    zillow = engineer_features(zillow)

    # Clusterineering
    zillow = create_cluster_centers_tax_location(zillow)
    zillow = create_cluster_center_size(zillow)
    
    # Split
    train, validate, test = split_data(zillow)

    X_train = train[features]
    y_train = train.logerror
    X_validate = validate[features]
    y_validate = validate.logerror
    X_test = test[features]
    y_test = test.logerror

    # scaling

    scaler, train_scaled, validate_scaled, test_scaled = min_max_scaler(X_train, X_validate, X_test)

    return train_scaled, y_train, validate_scaled, y_validate, test_scaled, y_test
