#Importing libraries for both processing and visualisation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Importing dataset
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=column_names)
movie_titles = pd.read_csv("Movie_Id_Titles")
movie_titles.head()

#Merging movies and user dataset based on item_id
df = pd.merge(df,movie_titles,on='item_id')
df.head()

#Setting seaborn style 
sns.set_style('white')

#Creating a ratings dataframe with average rating and number of ratings
df.groupby('title')['rating'].mean().sort_values(ascending=False).head()
df.groupby('title')['rating'].count().sort_values(ascending=False).head()
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.head()

#Setting the number of ratings column
ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
ratings.head()

#Plotting a few histograms for better visualisation of data
plt.figure(figsize=(10,4))
ratings['num of ratings'].hist(bins=70)
plt.show()

plt.figure(figsize=(10,4))
ratings['rating'].hist(bins=70)
plt.show()

sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)

#Creating a matrix relating user ids and movie titles
moviemat = df.pivot_table(index='user_id',columns='title',values='rating')
moviemat.head()

#Finding the most rated movie
ratings.sort_values('num of ratings',ascending=False).head(10)
ratings.head()

#Lets choose two movies - Star wars and Liar Liar
#Getting the user ratings of these two movies
starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']
starwars_user_ratings.head()

#To get correlations between two pandas series
similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)

#Removing NaN values and using a Dataframe instead of a series
corr_starwars = pd.DataFrame(similar_to_starwars,columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars.head()

#Sort the dataframe by correlation
corr_starwars.sort_values('Correlation',ascending=False).head(10)

#Filtering out movies with less than 100 ratings
corr_starwars = corr_starwars.join(ratings['num of ratings'])
corr_starwars.head()
corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False).head()

#Repeating the same for the second movie
corr_liarliar = pd.DataFrame(similar_to_liarliar,columns=['Correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation',ascending=False).head()