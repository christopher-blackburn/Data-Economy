import os 
import pandas as pd 
import geopy
from geopy.extra.rate_limiter import RateLimiter


# Change the directory to the job's data
os.chdir('/Users/cblackburn/Dropbox/Data Economy/Data')

# Import the job's data
jobDF = pd.read_csv('job_description_data.csv',header=None)

# Extract the location data
locDF = jobDF[[1,2]]

# Remove underscores
locDF['city'] = locDF[1].apply(lambda x: x.replace('_',' '))

# Compute a location variable
locDF['MSA'] = locDF['city'] + ', ' + locDF[2]

# Compute total observations by location
locDF['one'] = 1
locDF = locDF.groupby('MSA').sum().reset_index()

# Geocode the addresses
locator = geopy.Nominatim(user_agent='myGeocoder')
#location = locator.geocode(“Champ de Mars, Paris, France”)
geocode = RateLimiter(locator.geocode, min_delay_seconds=1)
locDF['location'] = locDF['MSA'].apply(geocode)
locDF['point'] = locDF['location'].apply(lambda loc: tuple(loc.point) if loc else None)
locDF[['latitude', 'longitude', 'altitude']] = pd.DataFrame(locDF['point'].tolist(), index=locDF.index)

# Export the data
locDF.to_csv('job_desciption_locations.csv',index=False)