import numpy as np 
import pandas as pd 
import seaborn as sns 
import plotly.express as px 
import matplotlib.pyplot as plt 

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances 
from scipy.spatial.distance import cdist

#reading the data 
data = pd.read_csv("data.csv")
genre_data = pd.read_csv("data_by_genres.csv")
year_data = pd.read_csv("data_by_year.csv")

print(data.head)

#from this we can decide to make "popularity" the dependent variable 
#to find the correlation between all variables and target variable "popularity"
from yellowbrick.target import FeatureCorrelation

feauture_names = ['acousticness', 'danceability', 'energy', 'instrumentalness',
       'liveness', 'loudness', 'speechiness', 'tempo', 'valence','duration_ms','explicit','key','mode','year']
X = data[feauture_names]
y = data['popularity']

features = np.array(feauture_names) #creating a list of all the feature names 
visualizer = FeatureCorrelation(label=features)

plt.rcParams['figure.figsize']=(20,20)
visualizer.fit(X,y) #fit the variables to the visualizer
visualizer.show()


#next, we can visualize how these variables have changed over time 
sound_feautures = ['acousticness', 'danceability', 'energy', 'instrumentalness',
       'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
fig = px.line(year_data, x='year', y=sound_feautures)
fig.show()

#showing characterisitics according to genre
top_genres = genre_data.nlargest(10 , 'popularity')

fig = px.bar(top_genres, x='genres', y=['valence', 'energy', 'danceability', 'acousticness'], barmode='group')
fig.show()

#using KMeans clustering 

#building cluster
cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=10))])
X = genre_data.select_dtypes(np.number)
cluster_pipeline.fit(X)
genre_data['cluster'] = cluster_pipeline.predict(X)

#visualizing clusters using TSNE
tsne_pipeline = Pipeline([('sclaer', StandardScaler()),('tsne',TSNE(n_components=2, verbose=1))])
genre_embedding = tsne_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x','y'],data=genre_embedding)
projection['genres'] = genre_data['genres']
projection['cluster'] = genre_data['cluster']

fig = px.scatter(projection, x='x', y='y', color='cluster',)
fig.show()


#clustering songs using K-Means
song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), 
                                  ('kmeans', KMeans(n_clusters=20, 
                                   verbose=False,))
                                 ], verbose=False)
X = data.select_dtypes(np.number)
number_cols = list(X.columns)
song_cluster_pipeline.fit(X)
song_cluster_labels = song_cluster_pipeline.predict(X)
data['cluster_label'] = song_cluster_labels

#visualizing these clusters using PCA
pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
song_embedding = pca_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
projection['title'] = data['name']
projection['cluster'] = data['cluster_label']

fig = px.scatter(
    projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'title'])
fig.show()