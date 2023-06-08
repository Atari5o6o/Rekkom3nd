import os 
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
data = pd.read_csv("/Users/anmol/dev/Projects/Rekkom3nd/data.csv")
genre_data = pd.read_csv("/Users/anmol/dev/Projects/Rekkom3nd/data_by_genres.csv")
year_data = pd.read_csv("/Users/anmol/dev/Projects/Rekkom3nd/data_by_year.csv")

print(data.head)

