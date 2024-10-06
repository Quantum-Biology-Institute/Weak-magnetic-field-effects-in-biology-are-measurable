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
from matplotlib.ticker import MaxNLocator

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


my_font_title = FontProperties(fname=font_title.get_file(), size=BIGGER_SIZE) 
endash = "\u2013"

fig, ax = plt.subplots() #figsize=(10, 6))
ax.set_title('b) spectral comparison  \u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\n', fontproperties=my_font_title)


# Replace with your CSV file path
file_path =  'averaged_spectra_new.csv'  

# Read the CSV file
df = pd.read_csv(file_path)


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel('light intensity\n (integrated counts)', fontsize=MEDIUM_SIZE, fontproperties=font_text)
ax.set_xlabel('wavelength (nm)', fontsize=MEDIUM_SIZE, fontproperties=font_text)

# Plot Lux values as a function of time
ms = 1 #markersize
plt.plot(df['Wavelength'], df['Hypo'], marker='o', linestyle='', color='#2e99a2', label='hypomagnetic chamber (AC+DC shielding)',markersize=ms)
plt.plot(df['Wavelength'], df['Inc'],  marker='o', linestyle='', color='purple', label='control box',markersize=ms)

# Dynamically set x-ticks based on the data range or manually, if desired
ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Ensure x-ticks are integers
ax.set_xticks([200, 400, 600])  # Set specific ticks if desired

ax.set_ylim([-800,2000])
yticks_s = [-500, 0, 500, 1000, 1500]
ax.set_yticks(yticks_s)

for my_label in ax.get_xticklabels():
     my_label.set_fontproperties(font_text)
     my_label.set_fontsize(MEDIUM_SIZE)
for my_label in ax.get_yticklabels():
     my_label.set_fontproperties(font_text)
     my_label.set_fontsize(MEDIUM_SIZE)
     
# Create a custom line object with larger markersize for the legend
control_box_line = mlines.Line2D([], [], color='purple',  marker='o', linestyle='', markersize=5, label='control box')
hypo_line        = mlines.Line2D([], [], color='#2e99a2', marker='o', linestyle='', markersize=5, label='hypomagnetic\nchamber\n(AC+DC shielding)')

legend_font = FontProperties(fname=font_text.get_file(), size=0.7*MEDIUM_SIZE)
ax.legend()

  # Assuming legend_elements is already a list of handles
legend = ax.legend(
      handles=[control_box_line, hypo_line],
      loc='upper right',
      bbox_to_anchor=(1.2, 1),  # Move legend further right (increase first value)
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

output_path = base_path + '11 Calibration data/Spectra/'
fig_numbers = plt.get_fignums()
fig.savefig(output_path + 'Scomparison.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
