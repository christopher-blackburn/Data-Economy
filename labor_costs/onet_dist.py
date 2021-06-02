import pandas as pd
import glob
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

'''
-----------------------------------------------
Extract possible software development skills
-----------------------------------------------
In this initial section of code, I start by 
extracting candidate skills that are 
related to software development. I will
manually go through and check whether 
these skills are relevant for software
development.
'''


softdev_annual_list = []

for year in range(2010,2020):


    softdev_list = []

    for month in range(1,13):
        
        print('Loading data for {}-{}'.format(month,year),end='...')
        
        file_path = 'E:/Research1/prediction/burning_glass/Structured Data/Skill/{}/'.format(year)

        if month < 10:

            skills_data = pd.read_csv(file_path + 'Skills_{}-0{}.txt'.format(year,month)
                                      , sep = '\t',encoding = 'ISO-8859-1')
        else:

            skills_data = pd.read_csv(file_path + 'Skills_{}-{}.txt'.format(year,month)
                                      , sep = '\t',encoding = 'ISO-8859-1')

        skills_data_reduced = skills_data[['Skill','SkillCluster','SkillClusterFamily']].drop_duplicates()
        
        skills_clusters = pd.read_csv('E:/Research1/prediction/burning_glass/Chris/Output/skill_clusters_manual.csv')

        softdev_skills = skills_clusters[skills_clusters['SoftwareCluster'] == 1.0]

        softdev_skills = pd.merge(skills_data_reduced,softdev_skills,on='SkillCluster',how='inner',validate='m:1')

        softdev_list.append(softdev_skills[['Skill','SkillCluster']].drop_duplicates())
        
        print('Appended data successfully.')
        

    softdev_annual_list.append(pd.concat(softdev_list).drop_duplicates()) 
    
    print('Successfully appended {}'.format(year))
    
    
    
# Save the data
softdev_skills = pd.concat(softdev_annual_list).drop_duplicates()
softdev_skills.to_csv('E:/Research1/prediction/burning_glass/Chris/Output/softdev_skills.csv',index=False)


'''
-----------------------------------------------
Manual identification of skills
-----------------------------------------------
After outputing the results above, I go through
and label skills as software development or not.
The next few lines of code update the software
develop skills list based on manual labeling.
'''
softdev_skills = pd.read_csv('E:/Research1/prediction/burning_glass/Chris/Output/softdev_skills_manual.csv')

softdev_skills = softdev_skills[softdev_skills['pass1'] == 1]

softdev_skills = softdev_skills['SkillCluster'].unique()


'''
-----------------------------------------------
Overlap between data and software skills
-----------------------------------------------
A couple functions for identifying data-related
and software development skills. Note: I have
already gone through the data-related skills
and found the keyword search to be pretty good
at identify data-related skills. 

The following code computes number of data-related
and software development skills for each job
posting. It then checks for overlap.
'''

# Check if data-related skill
def is_data_skill(x):
    
    data = 0 
    
    for token in x.lower().split():
        
        if token in ['data','database','databases','dataset']:
            
            data += 1
            
    if data > 0: 
        
        return 1 
    
    else:
        
        return 0
    
def is_softdev_skill(x,softdev_skills):
    
    if x in softdev_skills:
        
        return 1
    
    else: 0
        
        
job_skills_annual = []

for year in range(2010,2020):


    job_skills_list = []

    for month in range(1,13):
        
        print('Loading data for {}-{}'.format(month,year),end='...')
        
        file_path = 'E:/Research1/prediction/burning_glass/Structured Data/Skill/{}/'.format(year)

        if month < 10:

            skills_data = pd.read_csv(file_path + 'Skills_{}-0{}.txt'.format(year,month)
                                      , sep = '\t',encoding = 'ISO-8859-1')
        else:

            skills_data = pd.read_csv(file_path + 'Skills_{}-{}.txt'.format(year,month)
                                      , sep = '\t',encoding = 'ISO-8859-1')

        
        
        skills_data['data_skill'] = skills_data['Skill'].apply(lambda x: is_data_skill(x))
        
        skills_data['softdev_skill'] = skills_data['SkillCluster'].apply(lambda x: is_softdev_skill(x,softdev_skills))

        skills_data['total_skills'] = 1

        job_skills_list.append(skills_data[['BGTJobId','JobDate','data_skill','softdev_skill','total_skills']].groupby(['BGTJobId']).sum().reset_index())
        
        
        print('Appended data successfully.')
        
    job_skills_annual.append(pd.concat(job_skills_list).drop_duplicates())
    
    data = pd.concat(job_skills_annual).drop_duplicates()
    
# A Function for computing the overlap between data skills and software development skills
def overlap(x):
    
    if x['data_skill'] >  0 and x['softdev_skill'] > 0:
        
        return 1
    
    else:
        
        return 0
    
data['overlap'] = data.apply(overlap,axis=1)

def bin_con(x):
    
    if x > 0:
        
        return 1
    
    else:
        
        return 0 
    
    
data['data_skill'] = data['data_skill'].apply(lambda x: bin_con(x))
data['softdev_skill'] = data['softdev_skill'].apply(lambda x: bin_con(x))

'''
-----------------------------------------------
Get ONET Categories for Each Job Posting
-----------------------------------------------
In this section, we grab the ONET category for
each job posting to merge the Skills dataset
with ONET categories.
'''
def annual_sample(year):
    
    # Update the directory information
    textDir = 'E:/Research1/prediction/Burning_glass/Text Data/{}'.format(year)
    mainDir = 'E:/Research1/prediction/Burning_glass/Structured Data/Main/{}'.format(year)
    
    
    rel_files = []
    
    # Randomly sample a week from each month 
    for month in range(1,13):
        
        if month < 10:
                    
            # Randomly sample the file for a month
            rel_file = [month_file for month_file 
                                       in glob.glob(textDir+'/job_info_{}0{}*.csv'.format(year,month))]
            
            
            # Open the main dataset
            job_main = pd.read_csv(mainDir + '/Main_{}-0{}.txt'.format(year,month),
                                   sep='\t',encoding='ISO-8859-1').drop_duplicates()
            
        else:
            
            # Randomly sample a file for the month
            rel_file = [month_file for month_file 
                                       in glob.glob(textDir+'/job_info_{}{}*.csv'.format(year,month))]
            
            # Open the main dataset
            job_main = pd.read_csv(mainDir + '/Main_{}-{}.txt'.format(year,month),
                                   sep='\t',encoding='ISO-8859-1').drop_duplicates()
            
        rel_files.append(job_main[['BGTJobId','ONET']])
        
    # Concatenate all of the monthly files into a single dataset
    job_data = pd.concat(rel_files).drop_duplicates()
    
    job_data['year'] = year
        
    return job_data

# ONET Categories for each job posting
annual_data = pd.concat([annual_sample(year) for year in range(2010,2020)]).drop_duplicates()

# Merge the datasets together
annual_onet_data = pd.merge(data,annual_data,on='BGTJobId', how='inner',validate='1:1')

'''
-----------------------------------------------
Compute an estimate for p_w
-----------------------------------------------
In this section, we grab the ONET category for
each job posting to merge the Skills dataset
with ONET categories.
'''

# Compute total ONET postings by year
annual_onet_counts = annual_data.groupby(['ONET','year']).count().reset_index()
annual_onet_counts.columns = ['ONET','year','total_onet']

# Compute total data skills
annual_onet_data = annual_onet_data.groupby(['ONET','year']).sum().reset_index()

# Merge with the totals
annual_onet_data = pd.merge(annual_onet_data,annual_onet_counts,on=['ONET','year'],how='inner',validate='1:1')

# Compute the proportion of workers with at least one data-related skill
annual_onet_data['prop_data'] = annual_onet_data['data_skill']/annual_onet_data['total_onet']

# Save the data (commented out to avoid overwriting)
#annual_onet_data.to_csv('E:/Research1/prediction/burning_glass/Chris/Output/onet_data.csv',index=False)

'''
-----------------------------------------------
Compute an estimate for time-use adj. factor
-----------------------------------------------
In this section, I use Doc2Vec to compute
similarity metrics to estimate
the time-use adjustment factor. 

Note: I only compute the distance using the
2011 Doc2Vec model. In practice, you would
want to do this annually. However, I would
recommend going about this a few ways...
First, try averaging the distances across 
all years and fixing this quantity in the 
actual labor costs estimate. Otherwise,
you may have issues comparing these across
years. Second, if you want a time-varying
estimate. Then you need to make sure the
basis vectors are consistent across each 
year. There are methods available to do this.
'''

# Load the output from the previous section. 
onet_annual_data = pd.read_csv('E:/Research1/prediction/burning_glass/Chris/Output/onet_data.csv')

# Compute the top 15 occupations with the highest data skill percentag
onet_annual_mean = onet_annual_data.groupby('ONET')['prop_data'].mean().reset_index()
onet_top_data = onet_annual_mean.sort_values(by='prop_data',ascending=False).head(15)

# Load the 2011 Do2vec Model
model = Doc2Vec.load('E:/Research1/prediction/burning_glass/Chris/Word2Vec/w2v2011')

# Extract the ONET embeddings
model_tags = model.docvecs.index2entity
model_vecs = [model.docvecs[tag] for tag in model_tags]

# Extract the Data-Intensive Landmarks
data_vecs = []
for idx, tag in enumerate(model_tags):
    if tag in list(onet_top_data['ONET']):
        data_vecs.append(idx)
        
data_vectors = [model_vecs[idx] for idx in data_vecs]



# Find the minimum distance to the landmark data-intensive occupations
min_distances = []
for idx, vec in enumerate(model_vecs):
    dist = 0.0
    for data_vec in data_vectors:
        if cosine_similarity(vec.reshape(1,1000),data_vec.reshape(1,1000)) > dist:
            dist = cosine_similarity(vec.reshape(1,1000),data_vec.reshape(1,1000))
    min_distances.append([model_tags[idx],dist[0][0]])
    

# Create a dataframe of the distances and merge with the data
distDF = pd.DataFrame.from_dict(dict(ONET=np.array(min_distances)[:,0],
                                    dist=np.array(min_distances)[:,1]))

# Merge with the annual ONET data
onet_dist = pd.merge(onet_annual_data,distDF,on='ONET',how='inner',validate='m:1')

# Save the data (commented out to avoid overwriting)
#onet_dist.to_csv('E:/Research1/prediction/burning_glass/Chris/output/onet_distance.csv,index=False)

