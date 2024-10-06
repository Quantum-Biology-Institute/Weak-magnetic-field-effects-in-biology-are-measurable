###############################################################################
# BASE PATH
base_path = '/Users/clarice/Desktop/'
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
from datetime import datetime
from matplotlib.dates import DateFormatter, HourLocator
import matplotlib.dates as mdates
import matplotlib.lines as mlines

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

# Replace with your CSV file path
file_path =  'lux_readings_hypo_2_orig copy.csv'  
file_path2 = 'lux_readings_inc_orig copy.csv'  

# Read the CSV file with the correct encoding, skip errors if necessary
df  = pd.read_csv(file_path,  encoding='latin1')
df2 = pd.read_csv(file_path2, encoding='latin1')

# Remove unnecessary rows with text data
df  = df[df['Lux'].str.contains('Lux:')]
df2 = df2[df2['Lux'].str.contains('Lux:')]

# Clean the Lux values by removing the 'Lux: ' part
df['Lux']  =  df['Lux'].str.replace('Lux: ', '').astype(float)
df2['Lux'] = df2['Lux'].str.replace('Lux: ', '').astype(float)

# Convert Date/Time to datetime format
df['Date/Time']  = pd.to_datetime(df['Date/Time'])
df2['Date/Time'] = pd.to_datetime(df2['Date/Time'])

# Format x-axis to show only time (HH:MM)
df['Time'] = df['Date/Time'].dt.strftime('%H:%M')
df2['Time'] = df2['Date/Time'].dt.strftime('%H:%M')

my_font_title = FontProperties(fname=font_title.get_file(), size=BIGGER_SIZE) 
endash = "\u2013"

fig, ax = plt.subplots() #figsize=(10, 6))
ax.set_title('a) light levels comparison  \u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0', fontproperties=my_font_title)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel('illuminance (lux)', fontsize=MEDIUM_SIZE, fontproperties=font_text)
ax.set_xlabel('time of day on 2024/10/04', fontsize=MEDIUM_SIZE, fontproperties=font_text)
ax.set_ylim([0.055,0.085])
yticks_lux = [0.06, 0.07, 0.08]
ax.set_yticks(yticks_lux)

# Plot Lux values as a function of time
ms = 1 #markersize
plt.plot(df2['Date/Time'],  df2['Lux'], marker='o', linestyle='', color='purple', label='control box',markersize=ms)
plt.plot( df['Date/Time'],   df['Lux'], marker='o', linestyle='', color='#2e99a2', label= 'hypomagnetic chamber (AC+DC shielding)',markersize=ms) 

# Set custom ticks for specific times
custom_ticks = pd.to_datetime(['2024-10-04 11:30', '2024-10-04 12:00', '2024-10-04 12:30', '2024-10-04 13:00', '2024-10-04 13:30', '2024-10-04 14:00', '2024-10-04 14:30'])
ax.set_xticks(custom_ticks)
ax.set_xticklabels(custom_ticks.strftime('%H:%M'), rotation=45, ha="right")

# Set x-tick format
time_fmt = DateFormatter('%H:%M')
ax.xaxis.set_major_formatter(time_fmt)

for my_label in ax.get_xticklabels():
     my_label.set_fontproperties(font_text)
     my_label.set_fontsize(MEDIUM_SIZE)
for my_label in ax.get_yticklabels():
     my_label.set_fontproperties(font_text)
     my_label.set_fontsize(MEDIUM_SIZE)
     
# Create a custom line object with larger markersize for the legend
control_box_line = mlines.Line2D([], [], color='purple',  marker='o', linestyle='', markersize=5, label='control box')
hypo_line        = mlines.Line2D([], [], color='#2e99a2', marker='o', linestyle='', markersize=5, label='hypomagnetic chamber (AC+DC shielding)')

legend_font = FontProperties(fname=font_text.get_file(), size=0.7*MEDIUM_SIZE)
ax.legend()

 # Assuming legend_elements is already a list of handles
legend = ax.legend(
     handles=[control_box_line, hypo_line],
     loc='center',
     frameon=True,             # Turn off the frame around the legend
     prop=legend_font,          # Adjust font size as needed
     handletextpad=0.5,         # Spacing between the legend symbol and text
     labelspacing=0.5           # Spacing between rows of the legend
 )
 
# Customize the frame (facecolor sets the background color)
legend.get_frame().set_facecolor('white')  # Set background color to white
legend.get_frame().set_edgecolor('none')   # Remove the edge of the frame (border)
legend.get_frame().set_alpha(1.0)         

plt.grid(False)

# Adjust layout for better fit
plt.tight_layout()
plt.show()

# Calculate the mean and standard deviation
lux_mean_inc = df2['Lux'].mean()
lux_std_inc = df2['Lux'].std()

lux_mean_hypo = df['Lux'].mean()
lux_std_hypo = df['Lux'].std()

print("Mean Lux value inc:", lux_mean_inc)
print("Standard deviation of Lux values inc:", lux_std_inc)

print("Mean Lux value hypo:", lux_mean_hypo)
print("Standard deviation of Lux values hypo:", lux_std_hypo)

output_path = base_path + '11 Calibration data/Spectra/'
fig_numbers = plt.get_fignums()
fig.savefig(output_path + 'LLcomparison.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
