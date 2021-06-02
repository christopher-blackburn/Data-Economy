import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

'''
----------------------------------------
Load the OES data
----------------------------------------
This bit of code loops through the 
OES data and loads them into a list
for concatenation.
'''

# Import the data
onet_dist = pd.read_csv('E:/Research1/prediction/burning_glass/Chris/Output/onet_distance.csv')

# Create an SOC code
onet_dist['soc'] = onet_dist['ONET'].apply(lambda x: x[0:7])

base_url = 'https://github.com/christopher-blackburn/Datasets/blob/master/Data/national_M{}_dl.{}?raw=true'

occ_data_list = []

for year in range(2010,2020):
    
    if year < 2014:
        
        url = base_url.format(year,'xls')
        
    else:
        
        url = base_url.format(year,'xlsx')
        
    
    occ_data = pd.read_excel(url)
    
    if year == 2010 or year == 2011:
        
        occ_data['GROUP'] = occ_data['GROUP'].fillna('detailed')
    
    try:
    
        occ_data = occ_data[occ_data['GROUP'] == 'detailed']
 
    except:
        
        try:
        
            occ_data = occ_data[occ_data['OCC_GROUP'] == 'detailed']
        
        except:
        
            occ_data = occ_data[occ_data['o_group'] == 'detailed']
        
    try:
    
        occ_data = occ_data[['OCC_CODE','OCC_TITLE','TOT_EMP','A_MEAN']]
        
    except:
        
        occ_data = occ_data[['occ_code','occ_title','tot_emp','a_mean']]
    
    occ_data['YEAR'] = year
    
    if year == 2019:
        
        occ_data.columns = [x.upper() for x in occ_data.columns]
    
    occ_data_list.append(occ_data)
    
    print('{} appended'.format(year))
    
# Concatenate the data together
occ_data = pd.concat(occ_data_list)

'''
----------------------------------------
Compute the labor cost estimate
----------------------------------------
Special Note:
-------------
There is something bizarre going on with
2019. I drop this from the visualization
but it is worth checking into what is 
going on. My suspicion is that
the issue is with the OES data for this
year, but I am not certain. 
'''
  
# Numerator of a weighted-average distance
onet_dist['w_dist'] = onet_dist['ones']*onet_dist['dist']

# Sum up the data by SOC category
soc_dist = onet_dist.groupby(['soc','year'])[['ones','data_skill_bin','w_dist']].sum().reset_index()

# This is the weighted-average distance
soc_dist['w_dist'] = soc_dist['w_dist']/soc_dist['ones']

# P_omega computation
soc_dist['p_omega'] = soc_dist['data_skill_bin']/soc_dist['ones']

# Keep relevant variables
soc_dist = soc_dist[['soc','year','w_dist','p_omega']]

# Rename the columns
soc_dist.columns = ['OCC_CODE','YEAR','w_dist','p_omega']

# Merge the our estimates with salary and employment data from OES
occDist = pd.merge(occ_data,soc_dist,on=['OCC_CODE','YEAR'],how='inner',validate='1:1')

# Force missing salaries to 0 (results in conservative estimate)
occDist['A_MEAN'] = (pd.to_numeric(occDist['A_MEAN'],errors='coerce').fillna(0))

# Compute the spending estimate for each occupation
occDist['data_spending'] = occDist['w_dist']*occDist['p_omega']*occDist['TOT_EMP']*occDist['A_MEAN']

# Compute total spending by year
dataDF = occDist.groupby('YEAR')['data_spending'].sum().reset_index()

# Dropping 2019 because of possible data discrepancies
dataDF = dataDF[dataDF['YEAR'] < 2019]


'''
----------------------------------------
Visualize spending and growth
----------------------------------------
Visualizing spending and growth in 
nominal spending. 
'''

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True,figsize=(6,4))


ax1.bar(dataDF['YEAR'],dataDF['data_spending']/10**9,color=[0/255,76/255,151/255])
ax1.set_title('Labor Costs (Billions USD)')

dataDF['l_1'] = dataDF['data_spending'].shift(1)
dataDF['growth'] = 100*(dataDF['data_spending']/dataDF['l_1']) - 100

ax2.plot(dataDF['YEAR'],dataDF['growth'],color=[216/255,96/255,24/255],marker='o')
ax2.set_xlabel('Year')
ax2.set_title('Nominal Spending Growth (%)')
ax2.set_yticks(np.arange(0, dataDF['growth'].max()+4, 5))

plt.subplots_adjust(hspace=0.35)

fig.savefig('E:/Research1/prediction/burning_glass/Chris/Figures/data_spending_growth.png',bbox_inches='tight')
