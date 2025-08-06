
#  _______________________________________________________________________________________Importing necessary packages_____________________________________________________________________________________
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
import openpyxl

#  ________________________________________________________________________________Plot onfiguration (type, size and color)_____________________________________________________________________________
plt.rcParams.update({
    'font.family': 'Calibri',  
    'font.size': 22,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold'
})

technique_colors = {
    'Cast': '#00D58B',   #02EDE8  # blue
    'Wrought': '#02EDE8',  # orange
    'DED': '#ff7f0e',      # green
    'LPBF': '#FE0083',     # red
    'EBM': '#00A4FE'       # purple
}


#  ________________________________________________________________________________Loading and configuring the dataset________________________________________________________________________________


file_path = 'Fig_1_dataset.xlsx'
sheet_names = ['Cast', 'Wrought', 'DED', 'LPBF', 'EBM']
as_built_data = []
heat_treated_data = []

for sheet in sheet_names:
    df = pd.read_excel(file_path, sheet_name=sheet)
    as_built_data.append(df[df['Type'].str.lower() == 'as-built']['Yield strength (MPa)'].dropna())
    heat_treated_data.append(df[df['Type'].str.lower() == 'heat-treated']['Yield strength (MPa)'].dropna())

# === Create boxplot stats from min to max ===
def min_max_boxplot_stats(data_list):
    stats = []
    for data in data_list:
        data = data.dropna()
        stats.append({
            'med': (data.max() + data.min()) / 2,
            'q1': data.min(),
            'q3': data.max(),
            'whislo': data.min(),
            'whishi': data.max(),
            'fliers': []
        })
    return stats

#_________________________________________________________________________________________________Plotting________________________________________________________________________________

fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# As-built
as_built_stats = min_max_boxplot_stats(as_built_data)
as_built_box = axs[0].bxp(as_built_stats, showfliers=True, patch_artist=True,
                          medianprops=dict(visible=False))  # ✅ Hide median line
for patch, technique in zip(as_built_box['boxes'], sheet_names):
    patch.set_facecolor(technique_colors[technique])
axs[0].set_title('As-built', weight='bold')
axs[0].set_xticks(range(1, len(sheet_names) + 1))
axs[0].set_xticklabels(sheet_names, rotation=45, weight='bold')
axs[0].set_ylabel('Yield strength (MPa)', weight='bold')
axs[0].tick_params(axis='y', labelsize=16)

# Heat-treated
heat_treated_stats = min_max_boxplot_stats(heat_treated_data)
heat_treated_box = axs[1].bxp(heat_treated_stats, showfliers=True, patch_artist=True,
                              medianprops=dict(visible=False))  # ✅ Hide median line
for patch, technique in zip(heat_treated_box['boxes'], sheet_names):
    patch.set_facecolor(technique_colors[technique])
axs[1].set_title('Heat-treated', weight='bold')
axs[1].set_xticks(range(1, len(sheet_names) + 1))
axs[1].set_xticklabels(sheet_names, rotation=45, weight='bold')
axs[1].tick_params(axis='y', labelsize=16)

plt.tight_layout()
plt.show()