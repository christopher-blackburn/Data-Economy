'''
-------------------------------------------
Unzip the skills files
-------------------------------------------
'''

from zipfile import ZipFile

for year in range(2011,2020):
    
    file_path = 'E:/Research1/prediction/burning_glass/Structured Data/Skill/{}/'.format(year)
    saveDir = 'E:/Research1/prediction/burning_glass/Structured Data/Skill/{}'.format(year)

    for month in range(1,13):

        if month < 10:

            zippedfile = file_path + 'Skills_{}-0{}.zip'.format(year,month)

        else:

            zippedfile = file_path + 'Skills_{}-{}.zip'.format(year,month)


        with ZipFile(zippedfile,'r') as zip_ref: zip_ref.extractall(saveDir)
