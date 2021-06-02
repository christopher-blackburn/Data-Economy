# getBurningJobs.py

'''
This script accesses the jobs within the Burnign Glass XML files
'''

import os
from xml.dom import minidom 
import pandas as pd 


# Parse the XML document 
mydoc = minidom.parse('US_XML_AddFeed_20180702_20180708.xml')

# Extract each "Job" element
items = mydoc.getElementsByTagName('Job')



# Next, we can iterate within the each of these to extract the relevant information.
job_info_array = []
for node in items:
	try:
		# Extract the Job ID
		jobID = node.getElementsByTagName('JobID')[0].firstChild.nodeValue
	
		# Extract the raw text data from the job
		jobText = node.getElementsByTagName('JobText')[0].firstChild.nodeValue

	except AttributeError:
		# Extract the Job ID
		jobID = node.getElementsByTagName('JobID')[0].firstChild.nodeValue
		
		# Assign an empty job posting text
		jobText = ''

	# Append the job information to the job information array
	job_info_array.append([jobID,jobText])


# Convert to a pandas DataFrame
job_info_df = pd.DataFrame(job_info_array,columns=['JobID','JobText'])
