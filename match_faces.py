# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 09:18:20 2021

@author: hailey
"""
import math
import pandas as pd
from csv import writer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import shutil
import os
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import Birch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.cm as cm
from collections import Counter, defaultdict
from sklearn.neighbors import NearestCentroid

new_path_location = 'C:\\Users\\jared\\Documents\\DigitizedRhinoplasty\\DigitizedRhinoplasty-main\\'

def euclidean_distance(point1, point2):
    sum_sqrd_dist = 0
    for i in range(len(point1)):
        sum_sqrd_dist += math.pow(point1[i]-point2[i], 2)
    return math.sqrt(sum_sqrd_dist)


def getNeighbors(patient, df):
    dist_list = []
    dfPatient = []
    
    for i in range (len(df['ratio 2'])):
        
        newpath = new_path_location + 'face_group\\face_group1' 
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        
        dist = euclidean_distance([patient.loc[0]['ratio 2'], patient.loc[0]['ratio 5']], [df.loc[i]['ratio 2'], df.loc[i]['ratio 5']])
        dfPatient.append(df.loc[i]['patient'])
        dist_list.append(dist) 
    
    data = pd.DataFrame({'patient': dfPatient, 'distance': dist_list})
    data = data.sort_values('distance')
    
    return data


def KNN(group_number, used_patients):
    newpath = new_path_location + 'face_group_2' 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    df = pd.read_csv('database.csv')
    
    df.head()
    df['class'].unique()
    df.isnull().values.any()
    
    df['class'] = df['class'].map({'faces-1':0, 'faces-2':1, 'faces-3':2}).astype(int) #mapping numbers
    df.head()
    
    x_data = df.drop(['class', 'patient', 'ratio 4', 'ratio 3', 'ratio 1'], axis=1)
    y_data = df['class']
    
    data = pd.DataFrame(x_data, columns = ['ratio 2', 'ratio 5'])
    df.head()
    
    X_train, X_test, y_train, y_test = train_test_split(data, y_data, test_size=0.2, random_state = 1)
    
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train,y_train)
    
    curPatient = pd.read_csv('curPatient.csv')
    
    kNeighbors = getNeighbors(curPatient, df)
    
    folder_location = (newpath + '\\face_group' + str(group_number))
    if os.path.exists(folder_location):
        shutil.rmtree(folder_location)
        
    os.makedirs(folder_location)
            
        
    i = 0
    average_distance = 0
    row_index = 0
    
    for patient in kNeighbors.loc[:,"patient"]:
        if (i >= 8):
            break
        
        if not (patient in used_patients):
            file_location = new_path_location + "ScanInfo\\" + str(patient) + "\\" + str(patient) + ".jpg"
            
            try:
                shutil.copy(file_location, folder_location)
                i = i+1
                used_patients.append(patient)
                average_distance = average_distance + kNeighbors.loc[row_index,"distance"]
            except (FileNotFoundError):
                 pass
    
        row_index = row_index + 1
    
    print("row index = ", row_index)
    print(kNeighbors.head(10))
    print("size of folder = ", len(os.listdir(folder_location)))
    average_distance = average_distance / len(os.listdir(folder_location))
    return [average_distance, used_patients]


def kmeans():
    '''ACCESS THE DATABASE'''
    df = pd.read_csv('database.csv')
    ratios = df.drop(['patient', 'class'], axis=1)
    
    
    '''MAKE THE MODEL'''
    kmeans = KMeans(n_clusters=7, random_state=0).fit(ratios)
    df['kmean'] = kmeans.labels_
    
    
    '''GET THE CENTROIDS'''
    cluster_counts = Counter(kmeans.labels_)
    centroids = kmeans.cluster_centers_
    sum_of_distances = np.zeros((len(centroids), 5))
    
    
    '''MAKE THE GROUP FOLDER'''
    newpath = new_path_location + 'face_group_kmeans' 
    if os.path.exists(newpath):
        shutil.rmtree(newpath)
        
    os.makedirs(newpath)
    
    
    '''ADD EACH PATIENT TO THEIR RESPECTIVE CLUSTER FOLDER'''
    for i in range(len(kmeans.labels_)):
        folder_location = (newpath + '\\face_group' + str(df.loc[i, 'kmean']))
        file_location = new_path_location + "ScanInfo\\" + str(df.loc[i, 'patient']) + "\\" + str(df.loc[i, 'patient']) + ".jpg"
                
        if not (os.path.exists(folder_location)):
            os.makedirs(folder_location)
        
        try:
            shutil.copy(file_location, folder_location)
            '''GET THE TOTAL DISTANCES'''
            for j in range(5):
                sum_of_distances[df.loc[i, 'kmean'], j] = sum_of_distances[df.loc[i, 'kmean'], j] + abs(df.loc[i, 'ratio ' + str(j+1)] - centroids[df.loc[i, 'kmean'], j])
        except (FileNotFoundError):
            pass
      
    
    '''GET AVERAGES OF DISTANCES'''
    for k in range(len(sum_of_distances)):
        for ratio in range(5):
            sum_of_distances[k][ratio] = sum_of_distances[k][ratio] / cluster_counts[k]
            
    average_distance = [0, 0, 0, 0, 0]
    
    for row in range(5):
        for column in range(len(sum_of_distances)):
            average_distance[row] = average_distance[row] + sum_of_distances[column, row]
            
    average_distance[:] = [value / len(sum_of_distances) for value in average_distance]
    
    return average_distance
        
        
def affinity_propagation():
    df = pd.read_csv('database.csv')
    ratios = df.drop(['patient', 'class'], axis=1)
        
    affinity_prop_model = AffinityPropagation(damping=0.9, random_state=5).fit(ratios)
    affinity_prop = affinity_prop_model.predict(ratios)
        
    df['affinity'] = affinity_prop
    
    cluster_counts = Counter(df['affinity'])
    centroids = affinity_prop_model.cluster_centers_
    sum_of_distances = np.zeros((len(centroids), 5))
    
    newpath = new_path_location + 'face_group_affinity' 
    if os.path.exists(newpath):
        shutil.rmtree(newpath)
        
    os.makedirs(newpath)
    
    for i in range(len(affinity_prop)):
        folder_location = (newpath + '\\face_group' + str(df.loc[i, 'affinity']))
        file_location = new_path_location + "ScanInfo\\" + str(df.loc[i, 'patient']) + "\\" + str(df.loc[i, 'patient']) + ".jpg"
                
        if not (os.path.exists(folder_location)):
            os.makedirs(folder_location)
        
        try:
            shutil.copy(file_location, folder_location)
            for j in range(5):
                sum_of_distances[df.loc[i, 'affinity'], j] = sum_of_distances[df.loc[i, 'affinity'], j] + abs(df.loc[i, 'ratio ' + str(j+1)] - centroids[df.loc[i, 'affinity'], j])
        except (FileNotFoundError):
            pass
        
    for k in range(len(sum_of_distances)):
        for ratio in range(5):
            sum_of_distances[k][ratio] = sum_of_distances[k][ratio] / cluster_counts[k]
            
    average_distance = [0, 0, 0, 0, 0]
    
    for row in range(5):
        for column in range(len(sum_of_distances)):
            average_distance[row] = average_distance[row] + sum_of_distances[column, row]
            
    average_distance[:] = [value / len(sum_of_distances) for value in average_distance]
    
    return average_distance


def birch():
    df = pd.read_csv('database.csv')
    ratios = df.drop(['patient', 'class'], axis=1)
            
    birch = Birch(threshold=0.01, n_clusters=7).fit(ratios).predict(ratios)
    df['birch'] = birch
    
    cluster_counts = Counter(df['birch'])
    centroids = NearestCentroid().fit(ratios, birch).centroids_
    sum_of_distances = np.zeros((len(centroids), 5))
    
    newpath = new_path_location + 'face_group_birch' 
    if os.path.exists(newpath):
        shutil.rmtree(newpath)
        
    os.makedirs(newpath)
    
    for i in range(len(birch)):
        folder_location = (newpath + '\\face_group' + str(df.loc[i, 'birch']))
        file_location = new_path_location + "ScanInfo\\" + str(df.loc[i, 'patient']) + "\\" + str(df.loc[i, 'patient']) + ".jpg"
                
        if not (os.path.exists(folder_location)):
            os.makedirs(folder_location)
        
        try:
            shutil.copy(file_location, folder_location)
            for j in range(5):
                sum_of_distances[df.loc[i, 'birch'], j] = sum_of_distances[df.loc[i, 'birch'], j] + abs(df.loc[i, 'ratio ' + str(j+1)] - centroids[df.loc[i, 'birch'], j])
        except (FileNotFoundError):
            pass
        
    '''GET AVERAGES OF DISTANCES'''
    for k in range(len(sum_of_distances)):
        for ratio in range(5):
            sum_of_distances[k][ratio] = sum_of_distances[k][ratio] / cluster_counts[k]
            
    average_distance = [0, 0, 0, 0, 0]
    
    for row in range(5):
        for column in range(len(sum_of_distances)):
            average_distance[row] = average_distance[row] + sum_of_distances[column, row]
            
    average_distance[:] = [value / len(sum_of_distances) for value in average_distance]
    
    return average_distance


def agglomerative():
    df = pd.read_csv('database.csv')
    ratios = df.drop(['patient', 'class'], axis=1)
            
    agglomerative_model = AgglomerativeClustering(n_clusters=7).fit(ratios)
    agglomerative = agglomerative_model.fit_predict(ratios)
            
    df['agglomerative'] = agglomerative
    
    cluster_counts = Counter(df['agglomerative'])
    centroids = NearestCentroid().fit(ratios, agglomerative).centroids_
    sum_of_distances = np.zeros((len(centroids), 5))
    
    newpath = new_path_location + 'face_group_agglomerative' 
    if os.path.exists(newpath):
        shutil.rmtree(newpath)
        
    os.makedirs(newpath)
    
    for i in range(len(agglomerative)):
        folder_location = (newpath + '\\face_group' + str(df.loc[i, 'agglomerative']))
        file_location = new_path_location + "ScanInfo\\" + str(df.loc[i, 'patient']) + "\\" + str(df.loc[i, 'patient']) + ".jpg"
                
        if not (os.path.exists(folder_location)):
            os.makedirs(folder_location)
        
        try:
            shutil.copy(file_location, folder_location)
            for j in range(5):
                sum_of_distances[df.loc[i, 'agglomerative'], j] = sum_of_distances[df.loc[i, 'agglomerative'], j] + abs(df.loc[i, 'ratio ' + str(j+1)] - centroids[df.loc[i, 'agglomerative'], j])
        except (FileNotFoundError):
            pass
    
    '''GET AVERAGES OF DISTANCES'''
    for k in range(len(sum_of_distances)):
        for ratio in range(5):
            sum_of_distances[k][ratio] = sum_of_distances[k][ratio] / cluster_counts[k]
            
    average_distance = [0, 0, 0, 0, 0]
    
    for row in range(5):
        for column in range(len(sum_of_distances)):
            average_distance[row] = average_distance[row] + sum_of_distances[column, row]
            
    average_distance[:] = [value / len(sum_of_distances) for value in average_distance]
    
    return average_distance
    
        
def spectral():
    df = pd.read_csv('database.csv')
    ratios = df.drop(['patient', 'class'], axis=1)
            
    spectral = SpectralClustering(n_clusters=7).fit(ratios).fit_predict(ratios)
    df['spectral'] = spectral
    
    cluster_counts = Counter(df['spectral'])
    centroids = NearestCentroid().fit(ratios, spectral).centroids_
    sum_of_distances = np.zeros((len(centroids), 5))
    
    newpath = new_path_location + 'face_group_spectral' 
    if os.path.exists(newpath):
        shutil.rmtree(newpath)
        
    os.makedirs(newpath)
    
    for i in range(len(spectral)):
        folder_location = (newpath + '\\face_group' + str(df.loc[i, 'spectral']))
        file_location = new_path_location + "ScanInfo\\" + str(df.loc[i, 'patient']) + "\\" + str(df.loc[i, 'patient']) + ".jpg"
                
        if not (os.path.exists(folder_location)):
            os.makedirs(folder_location)
        
        try:
            shutil.copy(file_location, folder_location)
            for j in range(5):
                sum_of_distances[df.loc[i, 'spectral'], j] = sum_of_distances[df.loc[i, 'spectral'], j] + abs(df.loc[i, 'ratio ' + str(j+1)] - centroids[df.loc[i, 'spectral'], j])
        except (FileNotFoundError):
            pass
        
    '''GET AVERAGES OF DISTANCES'''
    for k in range(len(sum_of_distances)):
        for ratio in range(5):
            sum_of_distances[k][ratio] = sum_of_distances[k][ratio] / cluster_counts[k]
            
    average_distance = [0, 0, 0, 0, 0]
    
    for row in range(5):
        for column in range(len(sum_of_distances)):
            average_distance[row] = average_distance[row] + sum_of_distances[column, row]
            
    average_distance[:] = [value / len(sum_of_distances) for value in average_distance]
    
    return average_distance
            
        
def gaussian_mixture():
    df = pd.read_csv('database.csv')
    ratios = df.drop(['patient', 'class'], axis=1)
            
    gaussian = GaussianMixture(n_components=7, random_state=0).fit(ratios).predict(ratios)
            
    df['gaussian'] = gaussian
    
    cluster_counts = Counter(df['gaussian'])
    centroids = NearestCentroid().fit(ratios, gaussian).centroids_
    sum_of_distances = np.zeros((len(centroids), 5))
    
    newpath = new_path_location + 'face_group_gaussian' 
    if os.path.exists(newpath):
        shutil.rmtree(newpath)
        
    os.makedirs(newpath)
    
    for i in range(len(gaussian)):
        folder_location = (newpath + '\\face_group' + str(df.loc[i, 'gaussian']))
        file_location = new_path_location + "ScanInfo\\" + str(df.loc[i, 'patient']) + "\\" + str(df.loc[i, 'patient']) + ".jpg"
                
        if not (os.path.exists(folder_location)):
            os.makedirs(folder_location)
        
        try:
            shutil.copy(file_location, folder_location)
            for j in range(5):
                sum_of_distances[df.loc[i, 'gaussian'], j] = sum_of_distances[df.loc[i, 'gaussian'], j] + abs(df.loc[i, 'ratio ' + str(j+1)] - centroids[df.loc[i, 'gaussian'], j])
        except (FileNotFoundError):
            pass
        
    '''GET AVERAGES OF DISTANCES'''
    for k in range(len(sum_of_distances)):
        for ratio in range(5):
            sum_of_distances[k][ratio] = sum_of_distances[k][ratio] / cluster_counts[k]
            
    average_distance = [0, 0, 0, 0, 0]
    
    for row in range(5):
        for column in range(len(sum_of_distances)):
            average_distance[row] = average_distance[row] + sum_of_distances[column, row]
            
    average_distance[:] = [value / len(sum_of_distances) for value in average_distance]
    
    return average_distance

        
kmeans_results = kmeans()
affinity_propagation_results = affinity_propagation()
birch_results = birch()
agglomerative_results = agglomerative()
spectral_results = spectral()
gaussian_results = gaussian_mixture()

print("KMEANS AVERAGES :\n", kmeans_results)
print("\nAFFINITY PROPAGATION AVERAGES :\n", affinity_propagation_results)
print("\nBIRCH AVERAGES :\n", birch_results)
print("\nAGGLOMERATIVE AVERAGES :\n", agglomerative_results)
print("\nSPECTRAL AVERAGES :\n", spectral_results)
print("\nGAUSSIAN AVERAGES :\n", gaussian_results)

data = {'Kmeans':sum(kmeans_results), 'Affinity Propagation':sum(affinity_propagation_results), 
        'Birch':sum(birch_results), 'Agglomerative':sum(agglomerative_results),
        'Spectral':sum(spectral_results), 'Gaussian Mixture':sum(gaussian_results)}
courses = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize = (10, 8))
 
# creating the bar plot
plt.bar(courses, values, color=['indigo', 'mediumpurple', 'deepskyblue', 'turquoise', 'springgreen', 'greenyellow'],
        width = 0.8)
 
plt.xlabel("Clustering Algorithms")
plt.ylabel("Sum of Average Distances")
plt.title("Comparison of Clustering Algorithms")
plt.show()




