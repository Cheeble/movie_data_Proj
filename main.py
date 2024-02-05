# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 14:52:08 2024

@author: cheeble
"""

import pandas as pd
import numpy as np
import seaborn as sns 

df = pd.read_csv("movie_dataset.csv")
df.head()
print(df.isnull().values.any())

df.dropna(inplace = True)
df = df.reset_index(drop=True)

#df.dropna(thresh=5,axis=0)
#df = df.reset_index(drop=True)
df.info()
df.duplicated().values.any()
#df.drop_duplicates(keep = 'first', inplace = True)
#df.info()
df.columns =df.columns.map(lambda x: x.strip() if isinstance(x, str) else x )
df.rename(columns={'Runtime (Minutes)': 'Runtimes_Minutes'}, inplace = True)
df.rename(columns={'Revenue (Millions)': 'Revenue_Millions'}, inplace = True)
df.info()

#import sqlite3
#connection =sqlite3.connect(movie_dataset.csv)


#iris_versicolor = df[df['class'] == 'Iris-versicolor']


#ab = df[df['Title'] == 'The Dark Knight']
#selected_value = df.loc[df['Column1'] == 1, 'Column2'].values[0]
"""
names = ['The Dark Knight', 'Jason Bourne', 'Trolls','Rogue One']

mov_s = df[['Title', 'Rating']]



def bestRated(x):
    i = 0
    y = 0
    c = df.loc[df['Title'] == names[i], 'Rating'].values[0]
    for f in x:
        
        if  i < int(len(x)):
            sel_mov = df.loc[df['Title'] == names[i+1], 'Rating'].values[0]
            if sel_mov > c:
                y = sel_mov
            else:
                y = c
        g = df.loc[df['Rating'] == y, 'Title'].values[0]
        i = i + 1
    print(g + " : "+ y)

print(bestRated(names)
      
"""
#1. Best rated
sel_mov = df.loc[df['Title'] == 'The Dark Knight', 'Rating'].values[0]
sel_mov2 = df.loc[df['Title'] == 'Jason Bourne', 'Rating'].values[0]
sel_mov3 = df.loc[df['Title'] == 'Trolls', 'Rating'].values[0]
sel_mov4 = df.loc[df['Title'] == 'Rogue One', 'Rating'].values[0]

max_ra = max(sel_mov, sel_mov2, sel_mov3, sel_mov4)
print("*********************************************")
print(sel_mov)
print(sel_mov2)
print(sel_mov3)
print(sel_mov4)
print("*********************************************")
#2. Revenue Average
rev_sum = sum(df['Revenue_Millions'])
rev_no_row = len(df['Revenue_Millions'])
ave_rev = round(rev_sum/rev_no_row)
print("Average of Revenues : " + str(ave_rev))
print("*********************************************")


#3. average revenue of movies from 2015 to 2017 in the dataset
rec_15_17 =  df[(df['Year'] >= 2015) & (df['Year'] <= 2017)]
avg_rev = rec_15_17['Revenue_Millions'].mean()
print(round(avg_rev))
print("*********************************************")


#4. No of movies were released in the year 2016

m_r = df[df['Year'] == 2016].shape[0]
print(round(m_r))
print("*********************************************")


#5. No of movies were directed by Christopher Nolan
m_dir_Nolan = len(df[df['Director'] == 'Christopher Nolan'])
print(m_dir_Nolan)
print("*********************************************")


#6. No of movies in the dataset have a rating of at least 8.0
rat_movie = len(df[df['Rating'] >= 8.0])
print(rat_movie)
print("*********************************************")


#7. the median rating of movies directed by Christopher Nolan
med_rat = df[df['Director'] == 'Christopher Nolan']['Rating'].median()
print(med_rat)
print("*********************************************")


#8. the year with the highest average rating
year_rat = df.groupby('Year')['Rating'].mean().sort_values(ascending=False)
print(year_rat)
print("*********************************************")


#9. the percentage increase in number of movies made between 2006 and 2016
ptg_increase = ((df[df['Year'] == 2016].shape[0] - df[df['Year'] == 2006].shape[0]) / df[df['Year'] == 2006].shape[0]) * 100
print(ptg_increase)
print("*********************************************")


#10. most common actor in all the movies
actors = df['Actors'].str.split(', ').explode()
act_counts = actors.value_counts()
most_com_act = act_counts.idxmax()
print(most_com_act)
print("*********************************************")


#11. No of unique genres are there in the dataset
genres = df['Genre'].str.split(', ').explode()
unique_genres_count =genres.nunique()
print(unique_genres_count)
print("*********************************************\n\n")


df.to_csv("m_data_cleaned.csv")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('m_data_cleaned.csv',index_col=0)

print(df.corr())

correlation_plot =sns.heatmap(df.corr(), cmap="RdBu_r",annot=True, fmt=".2f", vmax=1, vmin=-1, square=True)

plt.show()