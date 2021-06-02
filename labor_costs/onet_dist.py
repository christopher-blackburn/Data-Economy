'''
-----------------------------------------------
Extract possible software development skills
-----------------------------------------------
In this initial section of code, I start by 

'''

import pandas as pd


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
Manual identification of skills
'''
softdev_skills = pd.read_csv('E:/Research1/prediction/burning_glass/Chris/Output/softdev_skills_manual.csv')

softdev_skills = softdev_skills[softdev_skills['pass1'] == 1]


'''
Overlap between data and software skills
'''
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
        
    return job_data

annual_data = pd.concat([annual_sample(year) for year in range(2010,2020)]).drop_duplicates()

softdev_skills = softdev_skills['SkillCluster'].unique()
