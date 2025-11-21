# -*- coding: utf-8 -*-
"""
Created on Sat May 24 21:15:12 2025

@author: lenovo
"""

#%% Import packages
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

#%% Read data
data_file = r'..\..\data\exp1_data.csv'
dat = pd.read_csv(data_file)
data_folder = os.getcwd()


#%% Fit parameters for each emotion condition (Aggregate Analysis)
total_fit_params = pd.DataFrame()
for this_emo in ['Negative', 'Neutral', 'Positive']:
    d = dat[dat['emo'] == this_emo]
    d = d.groupby(['contrast', 'validity'])['d'].mean().reset_index()
    dinv = d[d['validity'] == 'invalid'][['d']].values.reshape(1, 5)
    dv = d[d['validity'] == 'valid'][['d']].values.reshape(1, 5)
    contrasts = np.array([0.03, 0.08, 0.20, 0.45, 0.75])
    
    
    # Optimize using zero residual value (Minimize Sum of Squared Errors)
    def objective_function(params, cinv, dinv, c, dv):
        dmax1, c501, dmax2, c502, n = params
        # Calculate SSE (Sum of Squared Errors) for the first group (invalid)
        predicted_dinv = dmax1 * (cinv ** n / (cinv ** n + c501 ** n))
        error_dinv = np.sum((predicted_dinv - dinv) ** 2)
    
        # Calculate SSE for the second group (valid)
        predicted_dv = dmax2 * (c ** n / (c ** n + c502 ** n))
        error_dv = np.sum((predicted_dv - dv) ** 2)
        
        # Total SSE, combining errors from both groups
        return error_dinv + error_dv
    
    # Initial parameter guesses
    initial_guess = [1, 0.5, 1, 0.5, 1]
    # Parameter bounds, adjustable based on actual conditions
    bounds = ((0, 5), (0, 1), (0, 5), (0, 1), (0.1, 10))
    
    
    result = minimize(objective_function, initial_guess, args=(contrasts, dinv, contrasts, dv), bounds=bounds)
    dmax_fit1, c50_fit1, dmax_fit2, c50_fit2, n_fit = result.x
    
    # Print fitted results
    print("Fitted dmax_inv:", dmax_fit1)
    print("Fitted c501_inv:", c50_fit1)
    print("Fitted dmax_val:", dmax_fit2)
    print("Fitted c501_val:", c50_fit2)
    print("Fitted n:", n_fit)
    
    total_fit_params[this_emo] = result.x
    
    def nminv(c):
        dmax = dmax_fit1
        c50 = c50_fit1
        n = n_fit
        d = dmax * (c ** n / (c ** n + c50 ** n))
        return d
    def nmv(c):
        dmax = dmax_fit2
        c50 = c50_fit2
        n = n_fit
        d = dmax * (c ** n / (c ** n + c50 ** n))
        return d
        
    pre_dinv = []
    for i in contrasts:
        pre_dinv.append(nminv(i))
    pre_dv = []
    for i in contrasts:
        pre_dv.append(nmv(i))
    
    SSE = np.sum((pre_dinv - dinv)**2 + (pre_dv - dv)**2)
    SST = np.sum((dinv - np.mean(dinv))**2 + (dv - np.mean(dv))**2)
    R2 = 1 - (SSE/SST)
    print(f' R2: {R2} \n 1-SSE: {1-SSE}')
    
    
    x_fit_1 = np.linspace(np.min(contrasts), np.max(contrasts), 100)
    x_fit_2 = np.linspace(np.min(contrasts), np.max(contrasts), 100)
    
    # Calculate y-values for the fitted curve of the first group
    predicted_dinv_fit = dmax_fit1 * (x_fit_1 ** n_fit / (x_fit_1 ** n_fit + c50_fit1 ** n_fit))
    # Calculate y-values for the fitted curve of the second group
    predicted_dv_fit = dmax_fit2 * (x_fit_2 ** n_fit / (x_fit_2 ** n_fit + c50_fit2 ** n_fit))
    
    # Create figure and axis objects for detailed plot settings
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot raw scatter data and fitted curves
    ax.scatter(contrasts, dinv, label='Invalid (Raw)', marker='o', c='#0587AA')
    ax.scatter(contrasts, dv, label='Valid (Raw)', marker='s', c='#F5AC50')
    
    ax.plot(x_fit_1, predicted_dinv_fit, label='Invalid (Fit)', c='#0587AA')
    ax.plot(x_fit_2, predicted_dv_fit, label='Valid (Fit)', c='#F5AC50')
    ax.set_ylim((0.25, 2.5))
    
    ax.set_xlabel('Contrast')
    ax.set_ylabel("d'")
    ax.set_title(this_emo)
    ax.annotate(f'R2 = {round(R2,2)}', xy=(0.6, 0.5))
    ax.legend()
    plt.show()

total_fit_params.index = ['dmax_invalid', 'c50_invalid', 'dmax_valid', 'c50_valid', 'n']
total_fit_params = total_fit_params.T

# Save parameters
total_fit_params_file = os.path.join(data_folder, 'exp_1fit_params.csv')
total_fit_params.to_csv(total_fit_params_file)


#%% Calculate fitting parameters for each individual under each emotion condition

# Define objective function (placed outside the loop for cleanliness)
def objective_function(params, c, dinv, dv):
    dmax1, c501, dmax2, c502, n = params
    # Naka-Rushton formula
    predicted_dinv = dmax1 * (c ** n / (c ** n + c501 ** n))
    predicted_dv = dmax2 * (c ** n / (c ** n + c502 ** n))
    
    return np.sum((predicted_dinv - dinv) ** 2) + np.sum((predicted_dv - dv) ** 2)

# Define basic variables
total_fit_result = []
contrasts = np.array([0.03, 0.08, 0.20, 0.45, 0.75])
initial_guess = [2, 0.5, 2, 0.5, 1]
bounds = ((0, 5), (0, 0.8), (0, 5), (0, 0.8), (0.1, 10))

# Create directory for saving figures
fig_save_path = os.path.join(data_folder, 'individual_fit_figures')
if not os.path.exists(fig_save_path):
    os.makedirs(fig_save_path)

subjects = sorted(set(dat['sub']))

for this_emo in ['Negative', 'Neutral', 'Positive']:
    d_emo = dat[dat['emo'] == this_emo]
    print(f"Processing Emotion: {this_emo}")
    
    for this_sub in subjects:
        # Filter data
        this_d = d_emo[d_emo['sub'] == this_sub]
        dinv = this_d[this_d['validity'] == 'invalid']['d'].values.flatten()
        dv = this_d[this_d['validity'] == 'valid']['d'].values.flatten()
        
        # Check data integrity
        if len(dinv) != 5 or len(dv) != 5:
            print(f"  Skipping Subject {this_sub}: Data incomplete")
            continue
        
        # Execute fitting
        result = minimize(objective_function, initial_guess, args=(contrasts, dinv, dv), bounds=bounds)
        dmax_fit1, c50_fit1, dmax_fit2, c50_fit2, n_fit = result.x
        
        # Save fitting results
        fit_result = [this_sub, this_emo, dmax_fit1, c50_fit1, dmax_fit2, c50_fit2, n_fit]
        total_fit_result.append(fit_result)
        
        print(f"  Sub {this_sub} Fitted.")

        # =======================================================
        # New: Plot figure for each individual
        # =======================================================
        # 1. Generate x-coordinates for smooth curves
        x_smooth = np.linspace(np.min(contrasts), np.max(contrasts), 100)
        
        # 2. Calculate y-coordinates for smooth curves (Naka-Rushton)
        y_inv_smooth = dmax_fit1 * (x_smooth ** n_fit / (x_smooth ** n_fit + c50_fit1 ** n_fit))
        y_val_smooth = dmax_fit2 * (x_smooth ** n_fit / (x_smooth ** n_fit + c50_fit2 ** n_fit))
        
        # 3. Plotting
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Plot raw data points
        ax.scatter(contrasts, dinv, label='Invalid (Raw)', color='#0587AA', marker='o', s=60, alpha=0.8)
        ax.scatter(contrasts, dv, label='Valid (Raw)', color='#F5AC50', marker='s', s=60, alpha=0.8)
        
        # Plot fitted curves
        ax.plot(x_smooth, y_inv_smooth, label='Invalid (Fit)', color='#0587AA', linewidth=2)
        ax.plot(x_smooth, y_val_smooth, label='Valid (Fit)', color='#F5AC50', linewidth=2)
        
        # Set axes labels and title
        ax.set_xlabel('Contrast')
        ax.set_ylabel("d'")
        ax.set_title(f'Sub: {this_sub} | Emo: {this_emo} | n={n_fit:.2f}')
        ax.set_ylim(bottom=0) # Set y-axis lower limit to 0
        ax.legend()
        
        # 4. Save and close figure (prevent memory leaks)
        pic_name = f'{this_emo}_{this_sub}.png'
        plt.savefig(os.path.join(fig_save_path, pic_name), dpi=300)
        plt.close(fig) # Critical: Close figure after plotting to avoid memory overflow
        # =======================================================

#%% Adjust data format
total_fit_result = pd.DataFrame(total_fit_result)
total_fit_result.columns = ['sub', 'emo', 'dmax_inv', 'c_inv', 'dmax_v', 'c_v', 'n']
total_fit_result.to_csv(os.path.join(data_folder, 'total_fit_result.csv'), index=False)

#%% Convert data format to wide format
wide_df = total_fit_result.pivot_table(
    index='sub',
    columns='emo',
    values=['dmax_inv', 'c_inv', 'dmax_v', 'c_v', 'n'], 
    aggfunc='first'
    )
# Rename columns to "column_emo" format
wide_df.columns = ['_'.join(col) for col in wide_df.columns]
wide_df = wide_df.reset_index()

wide_df.to_csv(os.path.join(data_folder, 'wide_total_fit_result.csv'), index=False)