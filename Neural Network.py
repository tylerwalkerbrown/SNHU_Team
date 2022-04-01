#!/usr/bin/env python
# coding: utf-8

# In[48]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
from lxml import html
import csv 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Decision Tree 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
#Time Series packages
from matplotlib.dates import DateFormatter, date2num, WeekdayLocator, DayLocator, MONDAY
import datetime
#Exporting dataset
import openpyxl as xls
#Deep Learning
import tensorflow
from keras.layers import Dense
from keras.models import Sequential 


# In[50]:


# Import the os module
import os

# Get the current working directory
cwd = os.getcwd()

# Print the current working directory
print("Current working directory: {0}".format(cwd))

# Print the type of the returned object
print("os.getcwd() returns an object of type: {0}".format(type(cwd)))


# In[51]:


#Changing directory
#os.chdir('C:/Users/Tyler/Desktop/SNHU')


# In[52]:


#Exporting the data to an excel file
#export.to_excel('SNHU_Pitching.xlsx' ,index = False)


# In[53]:


SNHU_2022 = pd.read_html('https://snhupenmen.com/sports/baseball/stats/2022', header = 0)


# In[263]:


#Time Series Batting Averages Dictionaries
SNHU_Schedule_Dictionary = {'Date': SNHU_2022[6].Date,
                           'Overall BA': SNHU_2022[6].H.cumsum()/ SNHU_2022[6].AB.cumsum(),
                            'W/L': SNHU_2022[6]['W/L'],
                            'H/A': SNHU_2022[6]['Loc'],
                            'Score': SNHU_2022[6]['Score'],
                            'Opponent': SNHU_2022[6]['Opponent'],
                            'Hits' : SNHU_2022[6]['H'],
                            'AB' : SNHU_2022[6]['AB'],
                            'K': SNHU_2022[6]['K'],
                            'PO':SNHU_2022[6]['PO'],
                            'RBI':SNHU_2022[6]['RBI'],
                            'CS':SNHU_2022[6]['CS'],
                            '2B':SNHU_2022[6]['2B'],
                            '3B':SNHU_2022[6]['3B']
                           }                


# In[264]:


#Frames for averages 
overall = pd.DataFrame(SNHU_Schedule_Dictionary)
cumulative = pd.DataFrame(Cum_Avg)


# In[265]:


#Splitting Score
scores = overall.Score.str.split("-", expand = True)
#Appending dictionary
SNHU_Schedule_Dictionary['SNHU'] = scores[0]
SNHU_Schedule_Dictionary['Opp'] = scores[1]


# In[266]:


#Recreating dataframe 
overall = pd.DataFrame(SNHU_Schedule_Dictionary)
overall = overall.dropna()


# In[267]:


overall


# In[145]:


#Model for deep learning
cols = overall.shape[1]
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = (cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))


# In[209]:


for_array = {'Hits':overall['Hits'],
            'Runs':overall['Opp']}


# In[254]:


#Input Data and weights
input_data = np.array(overall['Hits'])
weights = {'weight': 1.25,
          'output': 1.24}


# In[255]:


#Defining relu
def relu(input):
    '''Define your relu activation function here'''
    # Calculate the value for the output of the relu function: output
    output = max(0, input)
    
    # Return the value just calculated
    return(output)

# Calculate node 0 value: node_0_output
node_0_input = (input_data * weights['weight']).sum()
node_0_output = relu(node_0_input)

# Calculate node 1 value: node_1_output
node_1_input = (input_data * weights['weight']).sum()
node_1_output = relu(node_1_input)

# Put node values into array: hidden_layer_outputs
hidden_layer_outputs = np.array([node_0_output, node_1_output])

# Calculate model output (do not apply relu)
model_output = (hidden_layer_outputs * weights['output']).sum()

# Print model output
print(model_output)


# In[258]:


def Win_prediction(input_data, weights):

    # Calculate node 0 value
    node_0_input = (input_data * weights['weight']).sum()
    node_0_output = relu(node_0_input)

    # Calculate node 1 value
    node_1_input = (input_data * weights['weight']).sum()
    node_1_output = relu(node_1_input)

    # Put node values into array: hidden_layer_outputs
    hidden_layer_outputs = np.array([node_0_output, node_1_output])
    
    # Calculate model output
    input_to_final_layer = (hidden_layer_outputs * weights['output']).sum()
    model_output = relu(input_to_final_layer)
    
    # Return model output
    return(model_output)


# In[262]:


# Create empty list to store prediction results
results = []
for input_data_row in input_data:
    # Append prediction to results
    results.append(Win_prediction(input_data_row, weights))

# Print results
print(input_data.sum())


# In[ ]:




