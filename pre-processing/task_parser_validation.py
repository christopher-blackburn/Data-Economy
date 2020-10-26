import pandas as pd
import os 
import numpy as np

# Change to the validation directory
os.chdir('/Users/cblackburn/Dropbox/Data Economy/Validation')

# Read in the task validation dataset
valid = pd.read_csv('task_parser_validation.csv')

# Create some string for easy referencing
is_task = 'Is Task? (Y=1/N=0)'
pr_task = 'Identified as Task? (Y/N)'
is_skll = 'Is Skill? (Y=1,N=0)'
is_expe = 'Is Experience? (Y=1,N=0)'

# Compute the total number documents
N = valid[['File']].drop_duplicates().shape[0]

# Compute the total number of sentences
N_sentences = valid.shape[0]

# Compute the total number of tasks 
N_tasks = valid[['Is Task? (Y=1/N=0)']].sum()[0]

# Compute total number of correctly predicted tasks
valid['correct_task'] = np.where(((valid[is_task] == 1) & (valid[pr_task] == 1)),1,0)
N_corr_tasks = valid[['correct_task']].sum()[0]

# Compute the total number of false negatives when task = 1
valid['false_negative_tasks'] = np.where(((valid[is_task] == 1) & (valid[pr_task] == 0)),1,0)
N_fneg_tasks = valid[['false_negative_tasks']].sum()[0]

# Compute the false positives for skills and experience
valid['skill_experience'] = np.where(((valid[is_skll] == 1)  | (valid[is_expe] == 1)),1,0)
valid['false_positive_sklls'] = np.where(((valid['skill_experience'] == 1) & (valid[pr_task] == 1)),1,0)
N_fpos_sklls = valid['false_positive_sklls'].sum()
N_sklls = valid['skill_experience'].sum()

# Compute other false positives
N_pred_tasks = valid[pr_task].sum()
N_fpos_oth = N_pred_tasks - N_corr_tasks - N_fpos_sklls


# Compute true negatives for skills and experience
valid['true_negative_sklls'] = np.where(((valid['skill_experience'] == 1) & (valid[pr_task] == 0)),1,0)
N_tneg_sklls = valid['true_negative_sklls'].sum()

# Compute the true negatives for other
N_tneg_oth = N_sentences - N_pred_tasks - N_fneg_tasks

# Create lists for the row elements
task_1 = ['Task $=1$',N_corr_tasks,N_fpos_sklls,N_fpos_oth,N_pred_tasks]
task_0 = ['Task $=0$', N_fneg_tasks,N_tneg_sklls,N_tneg_oth,N_sentences-N_pred_tasks]
totals = ['Total',N_tasks,N_sklls,N_sentences-N_tasks-N_sklls,N_sentences]
column_names = ['Predicted Sentence Type','Task','Skills and Experience','Other','Total']

# Table dataframe
table_df = pd.DataFrame([task_1,task_0,totals],columns=column_names)
print(table_df.to_latex())

