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


x_labels = ["Before DG", 
            r"1$^\text{st}$ DG", 
            r"2$^\text{nd}$ DG", 
            r"After 1$^\text{st}$ DO", 
            r"3$^\text{rd}$ DG", 
            "Power supply off", 
            r"4$^\text{th}$ DG", 
            r"5$^\text{th}$ DG", 
            r"After 2$^\text{nd}$ DO", 
            r"After 3$^\text{rd}$ DO", 
            r"After 4$^\text{th}$ DO",
            r"After 5$^\text{th}$ DO", 
            r"After 6$^\text{th}$ DO"]

y_values = [10.29403376,
5.05808788,
4.366979505,
6.626208418,
6.093269566,
6.100128605,
5.066324999,
5.510480741,
7.186233297,
7.55911516,
7.726621577,
7.92271658,
7.827453992]

fig, ax = plt.subplots(figsize=(10,6))

# Plotting the data
ax.plot(np.arange(0, 13), y_values, marker='s', color='blue', linestyle='-', linewidth=2, markersize=6, label='B (nT)')

# Labels and title
#ax.set_xlabel('Test Conditions', fontsize=12)
#ax.set_ylabel('B (nT)')

my_font_title = FontProperties(fname=font_title.get_file(), size=BIGGER_SIZE) 
ax.set_title('Magnetic field changes during degaussing (DG) and door openings (DO)', fontproperties=my_font_title)

# Rotate the x-axis labels
ax.set_xticks(np.arange(0, 13))  # Setting x-ticks
ax.set_xticklabels(x_labels, rotation=45, ha='right')

# Optional: uncomment to show grid
ax.grid(True)

# Legend
#ax.legend(loc='upper right')

# Customizing font properties for x and y tick labels
for my_label in ax.get_xticklabels():
    my_label.set_fontproperties(font_text)
    my_label.set_fontsize(MEDIUM_SIZE)

for my_label in ax.get_yticklabels():
    my_label.set_fontproperties(font_text)
    my_label.set_fontsize(MEDIUM_SIZE)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#ax.set_ylims([4,11])
    
fig.text(0.067, 0.91, 'nT', fontsize=MEDIUM_SIZE, fontproperties=font_text, 
         bbox=dict(facecolor='white', edgecolor='none'), ha='center')

plt.tight_layout()

output_path = base_path + '11 Calibration data/Degaussing door test/'
fig_numbers = plt.get_fignums()
fig.savefig(output_path + 'Degaussing.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF


plt.show()
