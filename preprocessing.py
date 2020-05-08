import prepare
import model
import pandas as pd


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
    zillow = prepare.wrangle_zillow()
    
    # Feature Engineering
    zillow = engineer_features(zillow)

    # Clusterineering
    zillow = create_cluster_centers_tax_location(zillow)
    zillow = create_cluster_center_size(zillow)
    
    # Split
    train, validate, test = prepare.split_data(zillow)

    X_train = train[features]
    y_train = train.logerror
    X_validate = validate[features]
    y_validate = validate.logerror
    X_test = test[features]
    y_test = test.logerror

    # scaling

    scaler, train_scaled, validate_scaled, test_scaled = prepare.min_max_scaler(X_train, X_validate, X_test)

    return train_scaled, y_train, validate_scaled, y_validate, test_scaled, y_test



def prepare_for_modeling_county(zillow, county, features=[]):
    '''
    Special function used to prepare data for modeling by county 
    '''
    # Acquire and Prep
    zillow = prepare.wrangle_zillow()

    #apply county filter
    zillow = zillow[zillow.county == county]
    
    # Feature Engineering
    zillow = engineer_features(zillow)

    # Clusterineering
    zillow = create_cluster_centers_tax_location(zillow)
    zillow = create_cluster_center_size(zillow)
    
    # Split
    train, validate, test = prepare.split_data(zillow)

    X_train = train[features]
    y_train = train.logerror
    X_validate = validate[features]
    y_validate = validate.logerror
    X_test = test[features]
    y_test = test.logerror

    # scaling

    scaler, train_scaled, validate_scaled, test_scaled = prepare.min_max_scaler(X_train, X_validate, X_test)

    return train_scaled, y_train, validate_scaled, y_validate, test_scaled, y_test
