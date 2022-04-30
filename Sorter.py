# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 12:39:15 2021

@author: Jared
"""
import json
import read_in
import pandas as pd
import os

folder_location = 'C:\\Users\\jared\\Documents\\DigitizedRhinoplasty\\DigitizedRhinoplasty-main\\JSON_Files\\'
directory = os.fsencode(folder_location)

for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if filename.endswith(".JSON"): 
         patient_name = os.path.basename(filename)[:-5]
         print(patient_name)
         
         try:
             read_in.add_new_patient(read_in.create_ratios(patient_name))
         except:
             print('COULDNT GENERATE FOR : ', patient_name) 