import pandas as pd 

# Import the data
onet_dist = pd.read_csv('/Users/cblackburn/Dropbox/Data Economy/onet_distance.csv')

# Create an SOC code
onet_dist['soc'] = onet_dist['ONET'].apply(lambda x: x[0:7])
