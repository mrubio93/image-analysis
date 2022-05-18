# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 12:12:26 2022

@author: Manuel Rubio
"""
import pandas as pd
import os
import rasterio as rio

# Define paths
path_0 = "C:\\Users\\33695\\OneDrive - UniLaSalle\\Documents\\Unilasalle\\Image Processing\\Python"
# Path with data
path_data = os.path.join(path_0, "Image_Analysis_TD1\\data")
os.chdir(path_data)
# Path with imgs
path_imgs = os.path.join(path_0, "Image_Analysis_TD2\\imgs")

# Choose relevant cols
fields = ["Rdmt_masse", "Longitude", "Latitude"]
yield_filtered = pd.read_csv("Filtered_Yield.csv", sep=";", usecols = fields)

#%% imgs
os.chdir(path_imgs)
src = rio.open("2021-03-08_LAI.TIFF")
LAI = src.read(1)

# to get geospatial info : src.transform
index_transform = src.transform

# obtener la posición en la matriz de pixeles para cada coordenada
(row, col) = rio.transform.rowcol(index_transform, yield_filtered.Longitude, yield_filtered.Latitude)

#%%  crear nuevo df con valores de index transform
# creamos el df
df_all_data = pd.DataFrame(columns = {"wheat_mass", "col", "row"})

# asignamos las columnas
df_all_data.wheat_mass = yield_filtered.Rdmt_masse
df_all_data.col = col
df_all_data.row = row

# Calcular mediana para cada pixel
raster_yield = df_all_data.groupby(["col", "row"], as_index=False).median()

#%% Index evolution
import glob
import numpy as np
import matplotlib.pyplot as plt

os.chdir(path_imgs)
list_files_LAI = glob.glob("*LAI.TIFF")
list_files_12band = glob.glob("*12band.TIFF")

date_list = []
LAI_list = []
NDVI_list = []
BSI_list = []
ARVI_list = []
OSAVI_list = []

for file in list_files_LAI:
    src = rio.open(file)
    LAI = src.read(1)
    name = "LAI - " + file[:10]
    raster_yield[name] = LAI[raster_yield["row"],raster_yield["col"]]
    date_list.append(file[:10])
    LAI_list.append(np.nanmean(LAI))
    
for file in list_files_12band:
    src = rio.open(file)
    B2 = src.read(2)
    B4 = src.read(4)
    B8 = src.read(8)
    B11 = src.read(11)
    
    #NDVI
    NDVI = (B8 - B4) / (B8 + B4)
    NDVI_list.append(np.nanmean(NDVI))
    
    #BSI
    BSI = -(((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2)))
    BSI_list.append(np.nanmean(BSI))
    
    #ARVI
    ARVI = (B8 - (2 * B4 - B2)) / (B8 + (2 * B4 - B2))
    ARVI_list.append(np.nanmean(ARVI))
    
    #OSAVI
    OSAVI = (1 + 0.16) * (B8 - B4) / (B8 + B4 + 0.16)
    OSAVI_list.append(np.nanmean(OSAVI))
    
    list_indexes_names = ["NDVI","BSI","OSAVI","ARVI"]
    list_indexes = [NDVI,BSI,OSAVI,ARVI]
    
    for k in range(len(list_indexes_names)) :
        column_name = list_indexes_names[k] +  "_" + file[8:10] + "_" + file[5:7]
        raster_yield[column_name] = list_indexes[k][raster_yield["row"],raster_yield["col"]]

# convertir la liste en dates
date_list = pd.to_datetime(date_list, format='%Y-%m-%d')

index_ev = pd.DataFrame({'Date':date_list,'LAI':LAI_list,'NDVI':NDVI_list,'BSI':BSI_list,\
                         'ARVI': ARVI_list, 'OSAVI': OSAVI_list})

#%% Correlation matrix
import seaborn as sb

# Feature selection
def index_selector(df):
    list_indexes_names = ["LAI", "NDVI","BSI","OSAVI","ARVI"]
    index_dict = {}
    index_corrs = {}
    best_idx = []
    
    for name in list_indexes_names:
        x = [  col for col in df if name in col ] 
        index_dict[name] = df[x]
        index_dict[name]["wheat_mass"] = df["wheat_mass"]
        index_corrs[name + " corr"] = index_dict[name].corr()
        
    for name in index_corrs:
        wheat_corr = index_corrs[name].dropna(axis=0, how="any")
        f, ax = plt.subplots(figsize=(10, 8))
        sb.heatmap(wheat_corr, mask=np.zeros_like(wheat_corr, dtype=np.bool), cmap=sb.diverging_palette(220, 10, as_cmap=True),
                    square=True, ax=ax)
        plt.show()
        wheat_corr = index_corrs[name].reset_index()
    
        wheat_corr = wheat_corr.iloc[-1]
        wheat_corr = wheat_corr.drop(wheat_corr.index[0])
    
        selected_index = wheat_corr.index[(wheat_corr >= 0.25) | (wheat_corr <= -0.25) ].tolist()
        for idx in selected_index:
            best_idx.append(idx)
    best_idx = list(set(best_idx))
    
    df_best_idx = df[best_idx]
    df_best_idx['row'] = df['row']
    df_best_idx['col'] = df['col']
    
    df_best_idx = df_best_idx.dropna(axis=0, how='any')
    
    return df_best_idx

# Function to plot the indexes
def index_grapher(df):

    list_indexes_names = ["LAI", "NDVI","BSI","OSAVI","ARVI"]
    index_dict = {}

    for name in list_indexes_names:
        x = [  col for col in df.columns if name in col ] 
        index_dict[name] = df[ x ]

    for name in index_dict.keys():
        
        for i in range(len(index_dict[name])) :
            x = index_dict[name].columns
            plt.plot(x,index_dict[name].iloc[i])
        plt.title(f"{name} evolution 2021")
        plt.ylabel(name)
        plt.xlabel("weeks")
        plt.xticks(rotation=45)
        plt.show()
    return index_dict

index_grapher(raster_yield)
best_index = index_selector(raster_yield)


#%% PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# definir predictores y targets del modelo
raster_yield = raster_yield.dropna(axis=0, how='any')
df_target = raster_yield[["wheat_mass"]]
df_predictors = raster_yield.drop(["row", "col", "wheat_mass"], axis=1)

# create the array
X = df_predictors.to_numpy()
#Prepare data
y = df_target['wheat_mass'].to_numpy()

# Scale
X = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)

# pca to df 
x_pca = pca.fit_transform(X)
pca_df = pd.DataFrame(data=x_pca, columns=['pc1', 'pc2'])

# concatenate df
df_conc = pd.concat((raster_yield, pca_df), axis=1)

# Plot
sb.relplot(data=df_conc, x='pc1', y='pc2', aspect=1.61)
plt.title("PCA")
plt.xlabel(f'Dim 1 | {(pca.explained_variance_ratio_[0] * 100).round()}%')
plt.ylabel(f'Dim 2 | {(pca.explained_variance_ratio_[1] * 100).round()}%')
plt.show()

#%% Clusters
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


### cluster for yield
df_kmeans = df_conc.dropna(axis=0, how='any')
df_kmeans_i = df_kmeans[["wheat_mass"]]

scaler_i = MinMaxScaler().fit(df_kmeans_i)
X_means_i = scaler_i.transform(df_kmeans_i)

sse_i = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(X_means_i)
    sse_i.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse_i)
plt.show()

clusters_i = KMeans(n_clusters=3).fit_predict(X_means_i)
plt.show()

df_clustered = df_conc.dropna(axis=0, how="any")
df_clustered["Cluster_yield"] = clusters_i

sb.scatterplot(data=df_clustered, # indication on which dataframe to use
                x = "row",
                y = "col",
                hue = "Cluster_yield")
plt.show()

### Cluster for indexes
df_kmeans_y = df_kmeans.drop(["row", "col", "wheat_mass"], axis=1)
#df_kmeans_y = df_kmeans[["pc1", "pc2"]] # select only pc1 and 2

scaler_y = MinMaxScaler().fit(df_kmeans_y)
X_means_y = scaler_y.transform(df_kmeans_y)

sse_y = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(X_means_y)
    sse_y.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse_y)
clusters_y = KMeans(n_clusters=3, random_state=1).fit_predict(X_means_y)
plt.show()

df_clustered["Cluster_index"] = clusters_y

sb.scatterplot(data=df_clustered,
                x = "row",
                y = "col",
                hue = "Cluster_index")
plt.show()

# # Plot
# sb.relplot(data=df_clustered, x='pc1', y='pc2', hue='Cluster_index', aspect=1.61)
# plt.title("PCA")
# plt.xlabel(f'Dim 1 | {(pca.explained_variance_ratio_[0] * 100).round()}%')
# plt.ylabel(f'Dim 2 | {(pca.explained_variance_ratio_[1] * 100).round()}%')
# plt.show()

#%% dataframes

#feature selection and scaling
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

d = {} # store new dataframes in the dictionary
dataframe_list = [raster_yield, df_clustered]
raster_yield.name, df_clustered.name = "raster_yield", "df_clustered"

for dataframe in dataframe_list:
    df_predictors = dataframe.drop(["row", "col", "wheat_mass", 'Cluster_index'], axis=1, errors='ignore')
    y = dataframe['wheat_mass'].to_numpy()
    df_name = "best_" + dataframe.name
    
    # create the array
    X = df_predictors.to_numpy()

    # Seleccionar las mejores 10 variables explicatorias
    X.shape
    d[df_name] = SelectKBest(f_regression, k = 10).fit_transform(X, y)
    
#%% Neural network
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

df_predictors = df_clustered.drop(["row", "col", "wheat_mass","Cluster_yield", 'Cluster_index'], axis=1, errors='ignore')
y = df_clustered['wheat_mass'].to_numpy()
X = df_predictors.to_numpy()
param_grid = {"hidden_layer_sizes" : [(100,), (150,)]}
nngrid = GridSearchCV(MLPRegressor(random_state=1, max_iter=5000), param_grid)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state =1)
nngrid.fit(X, y)

print("The result is: ", nngrid.best_score_, "and params are", nngrid.best_params_)

#%% Métodos (svr, kr)
# comparacion de kernels svr y kernel ridge : 
# https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_kernel_ridge_regression.html?highlight=kernel%20ridge
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge

C_range = (1, 50, 100)
gamma_range = (0.01, 0.1, 0.5, 1)
cv = KFold(n_splits=5)

svr = GridSearchCV(
    SVR(kernel="rbf", gamma=0.01),
    param_grid={"C": C_range, "gamma": gamma_range}, cv=cv, n_jobs=-1
)

kr = GridSearchCV(
    KernelRidge(kernel="rbf", gamma=0.01),
    param_grid={"alpha": [1, 0.1, 1e-2, 1e-3], "gamma": gamma_range}, cv=cv, n_jobs=-1
)

scaler = StandardScaler().fit(X) # scaler conserva los datos (media y sd) de X, para escalar futuros inputs
X = scaler.fit_transform(X)
svr.fit(X,y)
kr.fit(X, y)

print(
    "The best parameters for SVR (rbf) are %s with a score of %0.2f"
    % (svr.best_params_, svr.best_score_)
)

print(
    "The best parameters for Kernel Ridge are %s with a score of %0.2f"
    % (kr.best_params_, kr.best_score_)
)

#%% Random forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state =1)
kf = KFold(shuffle=True, n_splits=5)
rf_reg = RandomForestRegressor(max_depth=2, random_state=0)
score = cross_val_score(rf_reg, X_train, y_train, cv=kf)
print(f"The score for the dataframe is {score.mean()}")

#%% Models
from sklearn import linear_model
from sklearn.metrics import r2_score

# elegir df a analizar
df_conc = df_conc.dropna(axis=0, how='any')

#X = df_clustered[df_clustered["Cluster_yield"]==1]
X = df_conc.drop(["wheat_mass", "row", "col", "Cluster_index", "Cluster_yield"], axis=1, errors='ignore')

#X = df_conc[["pc1", "pc2"]]

y = df_conc['wheat_mass'].to_numpy()
#y = df_clustered[df_clustered["Cluster_yield"]==1]
#y = y['wheat_mass'].to_numpy()

scaler = StandardScaler().fit(X)
X = scaler.fit_transform(X)

# selector
#X = SelectKBest(f_regression, k = 10).fit_transform(X, y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state =1)

kf = KFold(shuffle=True, n_splits=5)
rf_reg = RandomForestRegressor(max_depth=3, random_state=0)
svr_rbf_reg = SVR(kernel="rbf", gamma=0.01)
lm = linear_model.LinearRegression()
lm_by = linear_model.BayesianRidge()
k_ridge = KernelRidge(kernel="rbf")
nnet = MLPRegressor(random_state=1, max_iter=10000, solver="adam")
model_list = [rf_reg, svr_rbf_reg, lm, lm_by, k_ridge] # saco nnet porque tarda y da mal
model_names = ["random forest", "SVR", "lm", "lm br","kernel ridge"]

for model, name_model in zip(model_list, model_names):
    score = cross_val_score(model, X_train, y_train, cv=kf)
    print(f"The score for the model {name_model} is {score.mean()}")
    
    score = cross_val_score(lm_by, X_train, y_train, cv=kf, scoring="neg_root_mean_squared_error")

    model = lm.fit(X_train, y_train)

    print(score.mean())

    print("The r2 value is ", r2_score(y_test, model.predict(X_test))) #r2

#%% Bayesian Ridge
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state =1)
kf = KFold(shuffle=True, n_splits=5)

lm_by = linear_model.BayesianRidge()
score = cross_val_score(lm_by, X_train, y_train, cv=kf, scoring="neg_root_mean_squared_error")

lm_by.fit(X_train, y_train)

plt.figure(figsize=(10,10))
plt.scatter(y_test, model.predict(X_test), c='crimson')

p1 = max(max(model.predict(X_test)), max(y_test))
p2 = min(min(model.predict(X_test)), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()
