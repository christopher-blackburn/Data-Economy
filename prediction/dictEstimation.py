
# Pandas
import pandas as pd 

# Multiprocessing tools and logger
from tqdm.contrib.concurrent import process_map 

# Doc2Vec loading module
from doc2vec_model import import_doc2vec 

# Annual data constructor 
from annual_sample import annual_sample 

# Time module for logging progam time
from time import time 

# Data-related title extract
from dataDictionary import dataTitle, dataText, dataJob

# Main module
if __name___ == "__main__":
	
	# Loop through the relevant years
	for year in range(2011,2018):

		# Construct the annual dataset
		print('Constructing data for year {}'.format(year),end='...')

		start = time()

		annualDF = annual_sample(year)

		print('Annual data successfully constructed! Time elapsed: {} seconds'.format(time()-start))

		# Identify data-related job titles
		print('Identifying data-related job titles',end='...')

		annualDF['dataTitle'] = annualDF['CleanTitle'].apply(lambda x: dataTitle(x))

		print('Successfully identified {} jobs with data-related titles! Time elapsed: {} seconds'.format(annualDF[annualDF['dataTitle']==1].shape[0],time()-start))

		# Count job postings within ONET categories and merge with the main data
		print('Constructing occupational weights by category',end='...')

		onet_counts = annualDF.groupby(['ONET']).size().reset_index()

		onet_counts.columns = ['ONET','prob']

		annualDF = pd.merge(annualDF,onet_counts,on='ONET',how='inner',validate='m:1')

		print('Successfully constructed weights. Time Elapsed: {} seconds'.format(time()-start))

		# Compute the share of data jobs (based on the title) for each ONET category
		print('Estimating data-related jobs based on job titles',end='...')

		annualDF_dataTitle = pd.merge(annualDF.groupby(['ONET'])['dataTitle'].sum().reset_index(),onet_counts,on=['ONET'], how = 'inner', validate = 'm:1')

		annualDF_dataTitle['onet_datatitle_share'] = annualDF_dataTitle['dataTitle']/annualDF_dataTitle['prob']

		print('Successfully estimated data-related jobs using job titles! Time elapsed: {} seconds '.format(time()-start))


		# Search the text for keywords (this is computationally expensive. use parallel processing)
		print('Searching job posting text for keywords. This will take a few minutes',end='...')

		annualDF['dataText'] = process_map(dataText,list(annualDF['jobText']),chunksize = 2000,max_workers = 60)
		
		#annualDF['dataText_weighted'] = process_map(dataText_weighted,list(annualDF['jobText']),chunksize = 2000,max_workers = 60)		

		print('Successfully identified job postings using the text search method. Time elapsed: {} seconds'.format(time()-start))


		# Compute the share of data jobs (based on job posting text) for each ONET category
		print('Estimating data-related jobs based on job posting text',end='...')

		annualDF_dataText = pd.merge(annualDF.groupby(['ONET'])['dataText'].sum().reset_index(),onet_counts,on='ONET',how='inner',validate='m:1')

		annualDF_dataText['onet_datatext_share'] = annualDF_dataText['dataText']/annualDF_dataText['prob']

		print('Successfully estimated data-related jobs using posting text. Time elapsed: {} seconds'.format(time()-start))


		# Combine the two approaches and compute the shares
		print('Combining estimates from both approaches',end='...')

		annualDF['dataJob'] = (annualDF['dataTitle'] + annualDF['dataText']).apply(lambda x: dataJob(x))

		annualDF_data = pd.merge(annualDF.groupby(['ONET'])['dataJob'].sum().reset_index(),onet_counts,on='ONET',how='inner',validate='m:1')

		annualDF_data['onet_data_share'] = annualDF_data['dataJob']/annualDF_data['prob']

		print('Successfully combined estimates. Time elapsed: {} seconds'.format(time()-start))

		# Combine datasets and save the annual data
		print('Merging data together and saving',end='...')

		annualDF_data_merged = pd.merge(annualDF_data,annualDF_dataTitle,on='ONET',how='inner',validate='1:1')

		annualDF_data_merged = pd.merge(annualDF_data_merged,annualDF_dataText,on='ONET',how='inner',validate='1:1')

		annualDF_data_merged['year'] = year 

		annualDF_data_merged = annualDF_data_merged[['ONET','prob_x','dataJob','dataTitle','dataText','onet_data_share','onet_datatitle_share','onet_datatext_share']]

		try:
			
			annualDF_data_merged = pd.merge(annualDF_data_merged,annualDF[['ONET','ONETName']].drop_duplicates(),on='ONET',how='inner',validate='m:1')
		
		except:

			pass

		savePath = 'E:/Research1/prediction/Burning_glass/Chris/Output/dictEstimation{}.csv'.format(year)
		
		annualDF_data_merged.to_csv(savePath,index=False)

		print('Successfully completed the dictionary-based estimation for {}. Time elapsed: {} seconds'.format(year,time()-start))

		


















