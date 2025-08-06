# Calling necessary packages

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
%matplotlib inline
import os
import shutil
from ctypes import *
import random
import math
import statistics
import seaborn as sns
import matplotlib.pyplot as plt

#Packages for surrogate model
import chaospy as cp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_percentage_error
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Creating numpy array from HDF5 file

trainFile  = h5py.File("Insert_your_file_name.hdf5",'r')  # 
conc = np.array(trainFile['composition']).astype(np.float32)  # each row is mass fraction of elements in the alloy 718 composition
g_prime = np.array(trainFile['g_prime']).astype(np.float32)   # Overall Mole fraction of gamma prime in the microstructure
g_dprime = np.array(trainFile['g_dprime']).astype(np.float32)    # Overall Mole fraction of gamma double prime in the microstructure


trainFile.flush()
trainFile.close()

#Creating Pandas dataframe 

Ni = []
Cr = []
Nb = []
Mo = []
Ti = []
Al = []
Co = []


for i in range(len(conc)):
    a=conc[i]
    Fe=1-sum(a)   #Fe being the balancing element

    #Elements = [Fe, Ni, Cr, Nb, Mo, Ti, Al, Co] 
    m = Fe/55.845 + a[0]/58.7 + a[1]/52 + a[2]/92.9 + a[3]/95.95 + a[4]/47.867+ a[5]/26.98154 + a[6]/58.933

    #Converting the mass fraction to mole fraction
    Ni.append(a[0]/58.7/m*100)
    Cr.append(a[1]/52/m*100)
    Nb.append(a[2]/92.9/m*100)
    Mo.append(a[3]/95.95/m*100)
    Ti.append(a[4]/47.867/m*100)
    Al.append(a[5]/26.98154/m*100)
    Co.append(a[6]/58.933/m*100)

#Final dataframe
df = pd.DataFrame({"Ni (mole %)": Ni,
                   "Cr (mole %)": Cr,
                   "Nb (mole %)": Nb,
                   "Al (mole %)": Al,
                   "Ti (mole %)": Ti,
                   "Co (mole %)": Co,
                   "Mo (mole %)": Mo,
                   "γ′(mole %)": g_prime,
                   "γ′′ (mole %)": g_dprime
                   })

#Visualization of composition and phase fraction distribution
plt.figure(figsize=(20, 15))
plt.rcParams['font.size'] = 20
plt.rcParams['font.weight'] = 'bold'

# Iterate through the DataFrame's columns to create a plot for each
for i, column in enumerate(df.columns, 1):
    plt.subplot(6, 3, i)  # Adjust the grid size based on your number of plots
    sns.histplot(df[column], kde=True)  # kde=True adds a Kernel Density Estimate plot
    ax = plt.gca()
    # Set the x and y labels with bold font weight
    ax.set_xlabel(column, fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    #plt.title(f'Distribution of {column}')

plt.tight_layout()
plt.show()



#Surrogate model based on Polynomial Chaos Expansion (PCE) method


m = df.iloc[:, 0:7].values.astype('float64')   # Input features
n = df['γ′(mole %)'].values.astype('float64')  # Output is for any specific phase fraction


# Creating uniform distribution of each input feature
Ni_sample = cp.Uniform(np.min(df['Ni (mole %)']), np.max(df['Ni (mole %)']))
Cr_sample = cp.Uniform(np.min(df['Cr (mole %)']), np.max(df['Cr (mole %)']))
Nb_sample = cp.Uniform(np.min(df['Nb (mole %)']), np.max(df['Nb (mole %)']))
Al_sample = cp.Uniform(np.min(df['Al (mole %)']), np.max(df['Al (mole %)']))
Ti_sample = cp.Uniform(np.min(df['Ti (mole %)']), np.max(df['Ti (mole %)']))
Co_sample = cp.Uniform(np.min(df['Co (mole %)']), np.max(df['Co (mole %)']))
Mo_sample = cp.Uniform(np.min(df['Mo (mole %)']), np.max(df['Mo (mole %)']))

#Creating a joint distribution of the input features
joint_dist = cp.J(Ni_sample, Cr_sample, Nb_sample, Al_sample, Ti_sample, Co_sample, Mo_sample)

# Split the data into training (80%), validation (10%), and test (10%) sets
X_train_val, X_test, y_train_val, y_test = train_test_split(m, n, test_size=0.10, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1111, random_state=42)


# Finding optimum polynomial order
orders = range(1, 5)  # Test polynomial orders from 1 to 5
errors = []
accuracy  = []
for order in orders:

    # Generate orthogonal polynomial expansion
    poly_expansion = cp.expansion.stieltjes(order, joint_dist)

    # Fit the PCE model using the training data
    approx_model_gp = cp.fit_regression(poly_expansion, X_train.T, y_train)

    # Predict on the validation set
    y_pred = approx_model_gp(*X_val.T)

    # Calculate the mean squared error (MSE) and R2 accuracy score
    mse = mean_squared_error(y_val, y_pred)
    errors.append(mse)
    r2 = r2_score(y_val, y_pred)
    accuracy.append(r2)

    print(f"Order {order}: Validation MSE = {mse}, Validation R2 = {r2}")

best_order = orders[np.argmin(errors)]  #Best order with the minimum MSE value
print(f"Best polynomial order: {best_order}")

# Creating the optimum polynomial expansion with the best order
poly_expansion = cp.expansion.stieltjes(best_order, joint_dist)

# Fit the final PCE model for the gamma prime phase fraction
approx_model_gp = cp.fit_regression(poly_expansion, X_train.T, y_train)

#Testing the model prediction on test dataset

predicted_output = approx_model_gp(*X_test.T)
mse = mean_squared_error(y_test,predicted_output)
r2 = r2_score(y_test, predicted_output)
print(f"Validation MSE = {mse}, R-squared = {r2}")
r2 = r2_score(y_test, predicted_output)

# Visualize the prediction accuracy of the model prediction
fig = plt.figure(figsize=(10,6))
plt.rcParams['font.size'] = 22
plt.rcParams['font.weight'] = 'bold'
sns.kdeplot(predicted_output,  color='#31a354',  linewidth=5, label='Predicted phase %')
sns.kdeplot(y_test, color='#fdae6b',  label='Actual phase %', linewidth=5)
plt.xlabel('γ′ (mole%)', fontsize = 22, weight='bold')
plt.ylabel('Kernel density', fontsize = 22, weight='bold')
plt.legend(prop={'size': 18,'family': 'Times New Roman','weight': 'bold'}, loc = 'best')
plt.show()


# Calculating Sobol sensitivity indices

main_effect_sobol_indices = cp.Sens_m(approx_model_gp, joint_dist)
total_effect_sobol_indices = cp.Sens_t(approx_model_gp, joint_dist)

df_pce_gp = pd.DataFrame(
    {
        'Elements':['Ni', 'Cr','Nb', 'Al', 'Ti', 'Co','Mo'],
        "Main effect": main_effect_sobol_indices,
        "Total effect": total_effect_sobol_indices
    }
)

fig = plt.figure(figsize=(18,6))
plt.rcParams['font.size'] = 22
plt.rcParams['font.weight'] = 'bold'
color_custom = ['#CF2865','#2b9fc4']
ax = df_pce_gp.plot(x = "Elements", kind = "bar", stacked = False, rot = 0, width = 0.6,  fontsize = 22 , title = "Sobol' sensitivity index", color = color_custom)
csfont = 12
plt.title('γ′(mole %) sensitivity', fontsize = 20, weight='bold')
plt.ylabel('Sobol index values',fontsize = 20, weight='bold')
plt.xlabel('Elements',fontsize = 20, weight='bold')
plt.ylim(0, 1)
y_labels = [0, 0.2, 0.4, 0.6, 0.8]
ax.set_yticks(y_labels)
plt.legend(prop={'size': 18,'family': 'Times New Roman','weight': 'bold'}, loc = 'best')
plt.show()