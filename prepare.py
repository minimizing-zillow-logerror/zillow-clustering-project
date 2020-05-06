from env import host, user, password
import pandas as pd
import numpy as np

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
    global host
    global user
    global password
    url = f'mysql+pymysql://{user}:{password}@{host}/{data_base_name}'
    df = pd.read_sql(query, url)
    return df

def run_query_to_csv():
    df = sql_database(data_base_name, query)
    df.to_csv("zillow_data.csv")

def read_zillow():
    df = pd.read_csv("zillow_data.csv")
    df.drop(columns= "Unnamed: 0", inplace=True)
    return df

def wrangle_geo_data(df):
    data = [["CA", "Los Angeles", 6037], ["CA", "Orange County", 6059], ["CA", "Ventura County", 6111]]
    fips = pd.DataFrame(data, columns= ["state", "county", "fips"])
    df.fips = df.fips.astype(int)
    geo_data = df.merge(fips, left_on="fips", right_on="fips")
    return geo_data


# ~~~~~~~~~ Prep ~~~~~~~~~ #


# ------------------------- #
#    Find Missing Values    # 
# ------------------------- #



def change_dtypes(df, col, type):
    df = df[col].astype(type)
    return df

def drop_null_col(df, ptc=.5):
    df = df.dropna(axis =1, thresh=(df.shape[0] * ptc))
    return df

def impude_unit_cnt(df):
    if df.propertylandusedesc == "Condominium" or df.propertylandusedesc == "Single Family Residential":
        return 1
    else:
        return 0


def impude_values(zillow):
    
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

# Outliers

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
    threshold = 3

    mean_y = np.mean(ys)
    stdev_y = np.std(ys)
    z_scores = [(y - mean_y) / stdev_y for y in ys]
    return np.where(np.abs(z_scores) > threshold)

def outliers_percentile(s):
    return s > s.quantile(.99)

def detect_outliers(s, k, method="iqr"):
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
    outlier = pd.DataFrame()
    for col in df.select_dtypes(exclude="object"):
        is_outlier = detect_outliers(df[col], k, method=method)
        outlier[col] = is_outlier
    return outlier

def drop_outliers(zillow, k, method="iqr"):
    outliers = detect_columns_outliers(zillow, k, method=method)
    zillow = zillow.drop(outliers.lotsizesquarefeet[outliers.lotsizesquarefeet > 10].dropna().index)
    
    return zillow

# Wrangle

def wrangle_zillow():
    zillow = pd.read_csv("zillow_data.csv")
    zillow = drop_null_col(zillow)

    col_obj = ["heatingorsystemtypeid", "parcelid", "id", "fips", "latitude", "longitude", "yearbuilt", "assessmentyear", "censustractandblock", "regionidcity", "regionidzip", "regionidcounty", "propertylandusetypeid"]

    zillow[col_obj] = change_dtypes(zillow, col_obj, "object")

    zillow = impude_values(zillow)

    col_drop = ["propertyzoningdesc", "heatingorsystemtypeid", "calculatedbathnbr", "fullbathcnt", "Unnamed: 0"]

    zillow.drop(columns = col_drop)

    zillow = drop_outliers(zillow, 3)

    return zillow