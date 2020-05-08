# Zillow Clustering Project - What is driving the errors in the Zestimates?

## Goal:
* Improve our original estimate of the log error by using clustering methodologies

# Data:
* From the Zillow data:
    * 2017 properties and prediction data from zillow database for  single unit/single family homes.

# Deliverables:
> Notebook: A final notebook (what you will deliver in your presenation) that is cleaned. It has markdown talking through thought processes, steps taken, decisions made. It has code commented. It runs the functions instead of the code behind it, in the tasks where functions are stored in a separate module. It includes the project planning steps. Be sure you prepare!! It has an intro, agenda, start with a summary of what you found (as you would in slides) and a conclusion. It has visualizations that incorporate best practices and that show your greatest discoveries in your exploration.

> ReadMe: You will need a ReadMe to indicate how to reproduce your work. You can include your project planning steps here as well, if you would like. Or a version of it.

> Modules: Where functions are necessary, indicated or ideal, it runs the functions in the notebook, but the functions are stored in their respective modules (prepare.py, e.g.). You will have an acquire.py, prepare.py (or wrangle to cover both acquire and prepare), and model.py (fit, predict, evaluate the final model). For bonus, you could add a preprocessing.py (where you will split, scale...or split_scale.py) and explore/clustering module, but it is not necessary.

# Executive Summary 

After extensive exploration, these were out findings:

Through statistical testing, we hypothesised that the the key items that could help us predict logerror were:

* Location: The logerror seemed to have different ranges based on the location, particularly between the the diffrent counties. 
* House age: The model seemed to be able to predict home value more accurately in older house.
* Size of the property: The size of the property had a impact on logerror, with smaller houses having more 
* Home quality: The value of logerror was smaller in houses with higher quality

* Features that do show a strong relationship with our target variable, `logerror`: 
    * `calculatedfinishedsquarefeet`
    * `fips`
    * `rawcensusblocktrack`
    * `roomcnt`
    * `taxvaluedollarcnt`
    * `landtaxvaluedollarcnt`
    * `taxamount`
    * `age_home`
    * `value_ratio`

Based on this, and other key insights that we gathered from exploration, we engieneered our own features to help with modeling. 

    * `age_home`: The age of the house in years as of 2017
    * `total_size`: The total size of the property (`lotsizesquarefeet` + `calculatedsquarefeet12`)
    * `value_ratio`: That is the rate between home tax and property tax
    * `tax_rate`: The `tax amount` / `taxdollarcount`

We use kmeans clustering to create new clusters that could helps us better understand the relationships in the data. We used clustering to also create new features:

* Created new clusters using `longitude`, `latitude` and `tax_rate`, as we found that there was a difference in tax_rate, and by using the mean tax_rate of the centroids, we could predict the level of variance in the `logerror`

# Create a new "age_home"

zillow["age_home"] = 2017 - zillow.yearbuilt

# total_property_size (combine calculated square feet of the house plus the size of the lot)

zillow["total_size"] = zillow.finishedsquarefeet12 + zillow.lotsizesquarefeet

# value_ratio = qhat is the rate between home tax and property tax
zillow["value_ratio"] = zillow.landtaxvaluedollarcnt / zillow.taxamount

# What the tax rate is 

zillow["tax_rate"] = zillow.taxamount / zillow.taxvaluedollarcnt

# Modeling
Three different models were implemented:

### Linear Regression
* Using the features identified above, we run a linear regression model. The model performed better than the baseline, but only slightly

* The same model was used, but split by county, so that each county had it's own model. The results were pulled together, and the overall model actually performed about the same as the base line.

* A random forest regressor was also tried, using the same features. Overall, the model performed muched better than the baseline, but had a negative r-square value, meaning that the model was actually performing worse than the baseline. 

## Summary

We gain a lot of useful insight and knowledge in the exploration phase, and clustering allowed us to visualize the data in new and exiting ways. Unfortunately, it is very hard to predict the logerror, as it is fairly close to zero, and in general, the baseline performs about the same as the models. 

We recommend using the baseline, as Zillow has a strong model, with low logerror variance. 

Data Dictionary:

key|old_key|description|
---|-------|-----------|
aircon|airconditioningtypeid|Type of cooling system present in the home (if any)architectural_style
architecturalstyletypeid|architecturalstyletypeid|Architectural style of the home (i.e. ranch, colonial, split-level, etc…)
area_base|finishedsquarefeet6|Base unfinished and finished area
area_firstfloor_finished|finishedfloor1squarefeet|Size of the finished living area on the first (entry) floor of the home
area_garage|garagetotalsqft|Total number of square feet of all garages on lot including an attached garage
area_live_finished|finishedsquarefeet12|Finished living area
area_liveperi_finished|finishedsquarefeet13|Perimeter living area
area_lot|lotsizesquarefeet|Area of the lot in square feet
area_patio|yardbuildingsqft17|Patio in yard
area_pool|poolsizesum|Total square footage of all pools on property
area_shed|yardbuildingsqft26|Storage shed/building in yard
area_total_calc|calculatedfinishedsquarefeet|Calculated total finished living area of the home
area_total_finished|finishedsquarefeet15|Total area
area_unknown|finishedsquarefeet50|Size of the finished living area on the first (entry) floor of the home
basementsqft|basementsqft|Finished living area below or partially below ground level
build_year|yearbuilt|The Year the principal residence was built
deck|decktypeid|Type of deck (if any) present on parcelfinishedfloor1squarefeet
flag_fireplace|fireplaceflag|Is a fireplace present in this home
flag_tub|hashottuborspa|Does the home have a hot tub or spa
framing|buildingclasstypeid|The building framing type (steel frame, wood frame, concrete/brick)
heating|heatingorsystemtypeid|Type of home heating system
id_fips|fips|Federal Information Processing Standard code - see https://en.wikipedia.org/wiki/FIPS_county_code for more details
id_parcel|parcelid|Unique identifier for parcels (lots)
id_zoning_raw|rawcensustractandblock|Census tract and block ID combined - also contains blockgroup assignment by extension
id_zoning|censustractandblock|Census tract and block ID combined - also contains blockgroup assignment by extension
latitude|latitude|Latitude of the middle of the parcel multiplied by 10e6
longitude|longitude|Longitude of the middle of the parcel multiplied by 10e6
material|typeconstructiontypeid|What type of construction material was used to construct the home
num_75_bath|threequarterbathnbr|Number of 3/4 bathrooms in house (shower + sink + toilet)
num_bathroom_calc|calculatedbathnbr|Number of bathrooms in home including fractional bathroom
num_bathroom|bathroomcnt|Number of bathrooms in home including fractional bathrooms
num_bath|fullbathcnt|Number of full bathrooms (sink, shower + bathtub, and toilet) present in home
num_bedroom|bedroomcnt|Number of bedrooms in home
num_fireplace|fireplacecnt|Number of fireplaces in a home (if any)
num_garage|garagecarcnt|Total number of garages on the lot including an attached garage
num_pool|poolcnt|Number of pools on the lot (if any)
num_room|roomcnt|Total number of rooms in the principal residence
num_story|numberofstories|Number of stories or levels the home has
num_unit|unitcnt|Number of units the structure is built into (i.e. 2 = duplex, 3 = triplex, etc...)
pooltypeid10|pooltypeid10|Spa or Hot Tub
pooltypeid2|pooltypeid2|Pool with Spa/Hot Tub
pooltypeid7|pooltypeid7|Pool without hot tub
quality|buildingqualitytypeid|Overall assessment of condition of the building from best (lowest) to worst (highest)
region_city|regionidcity|City in which the property is located (if any)
region_county|regionidcounty|County in which the property is located
region_neighbor|regionidneighborhood|Neighborhood in which the property is located
region_zip|regionidzip|Zip code in which the property is located
story|storytypeid|Type of floors in a multi-story house (i.e. basement and main level, split-level, attic, etc.). See tab for details.
tax_building|structuretaxvaluedollarcnt|The assessed value of the built structure on the parcel
tax_delinquency_year|taxdelinquencyyear|Year for which the unpaid propert taxes were due
tax_delinquency|taxdelinquencyflag|Property taxes for this parcel are past due as of 2015
tax_land|landtaxvaluedollarcnt|The assessed value of the land area of the parcel
tax_property|taxamount|The total property tax assessed for that assessment year
tax_total|taxvaluedollarcnt|The total tax assessed value of the parcel
tax_year|assessmentyear|The year of the property tax assessmentbasementsqft
zoning_landuse_county|propertycountylandusecode|County land use code i.e. it's zoning at the county level
zoning_landuse|propertylandusetypeid|Type of land use the property is zoned for
zoning_property|propertyzoningdesc|Description of the allowed land uses (zoning) for that property