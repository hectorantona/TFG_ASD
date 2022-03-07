#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 16:12:43 2019

@author: tales
"""

import os
import numpy as np
import glob
import pandas as pd
import featureGenerator as featGen


def dictval2str(dict, n_vals=2):
    s = ''
    for dict_val in list(dict.values())[0:n_vals]:
        s += '_' + str(dict_val)
    return s


def listFoldersInDir(dir):
    # list all folders including the current path
    folders = [x[0] for x in os.walk(dir)]
    # delete current path from list
    # del folders[0]
    return folders[1:]


#uIDDict = getUserIdsDictionaryFromDirectoryList(dirList, pathStyle, idNDigits):
# where uIDDict is a dictionary {newID,UID} where UID is a list with the 
# different unique user codes, and newID is a contiguous positive integer number 
# (1 to NumberOfUsers) associated with each UID.
# Inputs: dirList a list containing all directory names;
# pathStyle: that should be '/' for linux style or '\' for windows.
# idNDigits: this code assumes that each user ID is identified by the 
# first idNDigits of the folders containing the csv data files.
def getUserIdsDictionaryFromDirectoryList(dirList, pathStyle='/', idNDigits=4):
    # ids = [x.replace(path + pathStyle, '') for x in folders]
    UID = [x.split(pathStyle)[-1][0:idNDigits] for x in dirList]
    newID = np.unique(UID, return_inverse=True)[1];
    return dict(zip(newID + 1, UID))


def getUserIdsDictionaryFromDirectoryRoot(dirPath, binSize, aggCategory, nonAggCategory, pathStyle='/', idNDigits=4):
    return getUserIdsDictionaryFromDirectoryList(listFoldersInDir(dirPath), pathStyle, idNDigits)


def featExtractFromRawDataForAGivenUser(UID, rootDir, binSize, aggCategory, nonAggCategory, pathStyle='/'):
    foldersForUID = glob.glob(rootDir + pathStyle + '*' + UID + '*')
    dataListForUID = []
    #    print(foldersForUID)
    for folderCount in range(len(foldersForUID)):
        # load all csv files in folder
        csvFiles = sorted(glob.glob(foldersForUID[folderCount] + pathStyle + '*.csv'), reverse=True,
                          key=os.path.getsize)

        listPerInstance = []
        for file in csvFiles:
            # loading each csv file
            df = pd.read_csv(file, index_col=None, header=0, dtype=object)
            # using Timestamp as index
            dfIndexTime = df.set_index('Timestamp')
            # the command bellow was done so that pandas could recognize dfIndexTime as a time index
            dfIndexTime = dfIndexTime.set_index(pd.to_datetime(dfIndexTime.index))
            # appending all different data in one userList in which each element
            # is a different instance (e.g., EDA, IBI, etc)
            listPerInstance.append(dfIndexTime)

        dataListForUID.append(listPerInstance)
    userDict = {"dataAll": dataListForUID}
    outputDict = featGen.featGen(userDict, binSize, aggCategory, nonAggCategory)
    return outputDict

# =============================================================================
# 
# # Forms of listing files in a directory
# 
# from os import listdir
#  
# def list_files1(directory, extension):
#     return (f for f in listdir(directory) if f.endswith('.' + extension))
# 
# 
# 	
# from os import walk
#  
# def list_files2(directory, extension):
#     for (dirpath, dirnames, filenames) in walk(directory):
#         return (f for f in filenames if f.endswith('.' + extension))
#     
# 
# 
# from os import getcwd, chdir
#  
# def list_files3(directory, extension):
#     saved = getcwd()
#     chdir(directory)
#     it = glob('*.' + extension)
#     chdir(saved)
#     return it
# =============================================================================
