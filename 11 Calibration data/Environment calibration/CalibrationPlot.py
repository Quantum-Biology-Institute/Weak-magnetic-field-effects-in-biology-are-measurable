###############################################################################
# BASE PATH
base_path = '/Users/clarice/Desktop/'
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import gc
# Clear all figures
plt.close('all')
# Clear all variables (garbage collection)
gc.collect()

import sys
sys.path.append(base_path + '1 Utilities/')
import utilities as ut
import pandas as pd
import matplotlib.pyplot as plt

import plot_config
# Initialize fonts and settings
font_title, font_text, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = plot_config.setup()
SMALL_SIZE = 10
MEDIUM_SIZE = 20#14 # size of main text
BIGGER_SIZE = 20 # size of section text

###############################################################################
# CHANGE HERE ONLY
###############################################################################

# Load the CSV files
df_dc = pd.read_csv('dc_shielding_calibration.csv')
df_ac_dc = pd.read_csv('ac_dc_shielding_calibration.csv')
df_positive_control = pd.read_csv('positive_control_calibration.csv')

# Define function to plot each dataset side by side
def plot_comparison(fig, axes, df_dc, df_ac_dc, df_positive_control):
    
    # Set the labels for the top columns
    my_font_title = FontProperties(fname=font_title.get_file(), size=BIGGER_SIZE) 
    endash = "\u2013"
    axes[0, 0].set_title(f'a) DC shielding (B1{endash}5)\n', fontproperties=my_font_title)
    axes[0, 1].set_title(f'b) AC+DC shielding (B6{endash}7)\n', fontproperties=my_font_title)
    axes[0, 2].set_title(f'c) Positive control (+1{endash}3)\n', fontproperties=my_font_title)
    
    # Elements of stule
    
    # Suptitle
    #plt.suptitle('Calibration', fontsize=BIGGER_SIZE, fontproperties=font_title)
    
    # Remove top right spines
    for i in range(3):
        for j in range(3):
            axes[i, j].spines['top'].set_visible(False)
            axes[i, j].spines['right'].set_visible(False)

    axes[0, 0].set_ylabel('temperature (º C)', fontsize=MEDIUM_SIZE, fontproperties=font_text)
    axes[1, 0].set_ylabel('humidity (% RH)', fontsize=MEDIUM_SIZE, fontproperties=font_text)
    axes[2, 0].set_ylabel('pressure (hPA)', fontsize=MEDIUM_SIZE, fontproperties=font_text)
    
    axes[2, 0].set_xlabel('hours', fontsize=MEDIUM_SIZE, fontproperties=font_text)
    axes[2, 1].set_xlabel('hours', fontsize=MEDIUM_SIZE, fontproperties=font_text)
    axes[2, 2].set_xlabel('hours', fontsize=MEDIUM_SIZE, fontproperties=font_text)
    
    # Define the metric types
    metrics = ['Temperature (ºC)', 'Humidity (%RH)', 'Pressure (hPA)']
    
    # Manually set y-axis limits for each metric
    y_limits = {
        'Temperature (ºC)': (19.95, 20.65),   # Example: adjust these values based on the data
        'Humidity (%RH)': (57.5, 68.5), #was (58,68)
        'Pressure (hPA)': (997.5, 1002.5),
    }

# Plot temperature, humidity, pressure, and light for each dataset
    for i, metric in enumerate(metrics):
        # Plot DC Shielding (adjust x-axis for 6 hours)
        axes[i, 0].plot(df_dc['Time (hours)'], df_dc[f'{metric}_incubator'], label='control box', color='purple')
        axes[i, 0].plot(df_dc['Time (hours)'], df_dc[f'{metric}_hypomagnetic'], label='hypomagnetic chamber', color='#2e99a2')
        #axes[i, 0].set_ylabel(metric)
        #axes[i, 0].legend()
        axes[i, 0].set_xlim([0, 6])  # Set x-axis to 6 hours for DC Shielding
        axes[i, 0].set_ylim(y_limits[metric])  # Set consistent y-axis limits

        # Plot AC + DC Shielding
        axes[i, 1].plot(df_ac_dc['Time (hours)'], df_ac_dc[f'{metric}_incubator'], label='control box', color='purple')
        axes[i, 1].plot(df_ac_dc['Time (hours)'], df_ac_dc[f'{metric}_hypomagnetic'], label='hypomagnetic chamber', color='#2e99a2')
        axes[i, 1].set_xlim([0, 24])  # Set x-axis to 24 hours for AC + DC Shielding
        axes[i, 1].set_ylim(y_limits[metric])  # Set consistent y-axis limits

        # Plot Positive Control
        axes[i, 2].plot(df_positive_control['Time (hours)'], df_positive_control[f'{metric}_incubator'], label='Incubator', color='purple')
        axes[i, 2].plot(df_positive_control['Time (hours)'], df_positive_control[f'{metric}_hypomagnetic'], label='Hypomagnetic', color='#2e99a2')
        axes[i, 2].set_xlim([0, 24])  # Set x-axis to 24 hours for Positive Control
        axes[i, 2].set_ylim(y_limits[metric])  # Set consistent y-axis limits

        # Check if Positive Control is missing data
        if df_positive_control['Time (hours)'].max() < 24:
            print(f"Warning: Positive Control dataset has only {df_positive_control['Time (hours)'].max()} hours of data")


    yticks_temp = [20.0, 20.2, 20.4, 20.6]
    axes[0, 0].set_yticks(yticks_temp)
    axes[0, 1].set_yticks(yticks_temp)
    axes[0, 2].set_yticks(yticks_temp)
    
    yticks_hum = [58,60,62,64,66,68]
    axes[1, 0].set_yticks(yticks_hum)
    axes[1, 1].set_yticks(yticks_hum)
    axes[1, 2].set_yticks(yticks_hum)
    
    yticks_pr = [998, 1000, 1002]
    axes[2, 0].set_yticks(yticks_pr)
    axes[2, 1].set_yticks(yticks_pr)
    axes[2, 2].set_yticks(yticks_pr)
    
    xticks_h = [3,6]
    axes[0, 0].set_xticks(xticks_h)
    axes[1, 0].set_xticks(xticks_h)
    axes[2, 0].set_xticks(xticks_h)
    
    xticks_h = [3,6,9,12,15,18,21,24]
    axes[0, 1].set_xticks(xticks_h)
    axes[1, 1].set_xticks(xticks_h)
    axes[2, 1].set_xticks(xticks_h)
    axes[0, 2].set_xticks(xticks_h)
    axes[1, 2].set_xticks(xticks_h)
    axes[2, 2].set_xticks(xticks_h)
    
    
    # Adjust font properties for tick labels
    for i in range(3):
        for j in range(3):
            for my_label in axes[i, j].get_xticklabels():
                my_label.set_fontproperties(font_text)
                my_label.set_fontsize(MEDIUM_SIZE)
            for my_label in axes[i, j].get_yticklabels():
                my_label.set_fontproperties(font_text)
                my_label.set_fontsize(MEDIUM_SIZE)
        
    legend_font = FontProperties(fname=font_text.get_file(), size=MEDIUM_SIZE)
    
    # Assuming legend_elements is already a list of handles
    legend = axes[0, 1].legend(
        loc='upper center',
        frameon=True,             # Turn off the frame around the legend
        prop=legend_font,          # Adjust font size as needed
        handletextpad=0.5,         # Spacing between the legend symbol and text
        labelspacing=0.5           # Spacing between rows of the legend
    )
    
    # Customize the frame (facecolor sets the background color)
    legend.get_frame().set_facecolor('white')  # Set background color to white
    legend.get_frame().set_edgecolor('none')   # Remove the edge of the frame (border)
    legend.get_frame().set_alpha(1.0)         
    
    # Adjust layout for better fit
    plt.tight_layout()
    plt.show()
    
    
###############################################################################
###############################################################################

# Create figure and axes (4 rows for temp, humidity, pressure, light; 3 columns for each dataset)
fig1, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12), sharex=False)
# Call the function to plot the comparison
plot_comparison(fig1, axes, df_dc, df_ac_dc, df_positive_control)


output_path = base_path + '11 Calibration data/Environment calibration/'
fig_numbers = plt.get_fignums()
fig1.savefig(output_path + 'Calibration.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
