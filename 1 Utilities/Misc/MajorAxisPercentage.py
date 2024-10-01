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

def my_plot(fig, ax, y_values, y_values_green, title, y1, y2, y3, y4, y1h, y2h, y3h, y4h):

    # X labels for the plot
    x_labels = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B1--7", "+1", "+2", "+3", "+1--3"]
    
    # Extract even and odd indexed values separately for plotting
    y_values_even = y_values[::2]  # Even-indexed values (0, 2, 4, ...)
    y_values_odd  = y_values[1::2]  # Odd-indexed values (1, 3, 5, ...)
    
    y_values_even_green = y_values_green[::2]  # Even-indexed values (0, 2, 4, ...)
    y_values_odd_green  = y_values_green[1::2]  # Odd-indexed values (1, 3, 5, ...)
    
    # Set x positions for the pairs; both even and odd will share the same x positions
    x_positions = np.array([0,1,2,3,4,5,6,8,9,10])#np.arange(len(x_labels))  # x positions
    
    shift = 0.1
    
    # Plot the even indexed y_values with round markers
    plt.scatter(x_positions-shift, y_values_even, marker='o', color='blue', s=50, label='control $\cdot$ all states')
    plt.scatter(x_positions+shift, y_values_even_green, marker='o', color='green', s=50, label='control $\cdot$ only healthy')
    
    # Plot the odd indexed y_values with square markers
    plt.scatter(x_positions-shift, y_values_odd, marker='s', color='blue', s=50, label='hypomagnetic $\cdot$ all states')
    plt.scatter(x_positions+shift, y_values_odd_green, marker='s', color='green', s=50, label='hypomagnetic $\cdot$ only healthy')
    
    # Plot averages at x = 7 and x = 11
    plt.scatter(7-shift, y1, marker='o', color='blue', s=50)
    plt.scatter(7+shift, y1h, marker='o', color='green', s=50)
    plt.scatter(7-shift, y2, marker='s', color='blue', s=50)
    plt.scatter(7+shift, y2h, marker='s', color='green', s=50)
    
    plt.scatter(11-shift, y3, marker='o', color='blue', s=50)
    plt.scatter(11+shift, y3h, marker='o', color='green', s=50)
    plt.scatter(11-shift, y4, marker='s', color='blue', s=50)
    plt.scatter(11+shift, y4h, marker='s', color='green', s=50)
    
    # Add labels, title, and format x-axis
    plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11], x_labels, fontsize=12)
    #plt.xlabel('X Labels', fontsize=14)
    #plt.ylabel('Y Values', fontsize=14)
    
    my_font_title = FontProperties(fname=font_title.get_file(), size=BIGGER_SIZE) 
    ax.set_title(title, fontproperties=my_font_title)
    
    # Add a legend
    my_font_leg = FontProperties(fname=font_text.get_file(), size=1.7*SMALL_SIZE) 
    leg = plt.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none', prop=my_font_leg)
    leg.get_frame().set_alpha(1.0)
    
    # Add a black vertical dashed line at x=7.5
    my_font_line = FontProperties(fname=font_text.get_file(), size=1.7*SMALL_SIZE) 
    plt.axvline(x=7.5, color='black', linestyle='--', linewidth=1)
    plt.axhline(y=38, color='red', linestyle='--', linewidth=1)
    plt.text(plt.gca().get_xlim()[1], 38, 'Leibovich (20)', color='red', 
         va='top', ha='right', fontproperties=my_font_line)
    
    # Customizing font properties for x and y tick labels
    for my_label in ax.get_xticklabels():
        my_label.set_fontproperties(font_text)
        my_label.set_fontsize(MEDIUM_SIZE)
    
    for my_label in ax.get_yticklabels():
        my_label.set_fontproperties(font_text)
        my_label.set_fontsize(MEDIUM_SIZE)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Show the plot
    plt.tight_layout()

# Day 1, Major Axis difference in percent, same order as concatenated
y_values = [37.734636605796716,
 52.134548098836916,
 45.8952162345755,
 79.98092637807827,
 76.58588121890132,
 101.57244544238348,
 10.4721198866134,
 17.633795341224786,
 26.453119338109826,
 37.63876829590645,
 26.319753978439902,
 23.382090157159414,
 33.839115645960405,
 29.985846098272866,
 19.058012881101945,
 17.281812922082132,
 13.120118926829832,
 24.75996906319078,
 26.250422323116375,
 25.077925954617815]

# Day 1, Major Axis difference in percent, same order as concatenated, healthy only
y_values_green = [40.66071187706562,
 50.20275903125029,
 47.92134780859144,
 79.78490120624045,
 79.75299773560093,
 107.47292466787508,
 10.4721198866134,
 17.633795341224786,
 26.453119338109826,
 32.34404705788207,
 23.19288597419564,
 24.718139909438634,
 36.70920910656284,
 29.985846098272866,
 19.058012881101945,
 17.24575669272682,
 11.220171348973007,
 24.75996906319078,
 26.306638352161034,
 25.02997194136026]

# Major Axis Day 2 in percent, same order as concatenated
y_values_D2 = [115.99460756885911,
 132.40100273934047,
 141.62980777153976,
 53.58886682740634,
 70.38795133747908,
 111.38270390012417,
 84.44243976712146,
 154.1840007381776,
 144.2101187971346,
 225.70344881769424,
 66.17212859632397,
 89.47760418963759,
 34.92339591558395,
 34.78098461597307,
 39.73493737541127,
 45.138895814516374,
 77.15083815789683,
 68.04167600431398,
 71.47746631498052,
 69.86118707833172]

y_values_green_D2 = [123.17817958771928,
 149.92003584335401,
 86.867600933295,
 53.417559807669235,
 73.39019876684232,
 87.77953705666975,
 84.44243976712146,
 148.1929651858699,
 142.7606964358472,
 140.49945001305468,
 67.92888063886647,
 85.52016542168829,
 36.446425436261414,
 33.38464090280474,
 39.73493737541127,
 46.620711070301525,
 75.69911740914601,
 68.04167600431398,
 65.88409306481205,
 64.31505102731843]

# Major Axis Day 3 in percent, same order as concatenated
y_values_D3 = [122.04793005112768,
 125.32750598544786,
 139.84353033680063,
 91.51281689581667,
 115.819417585431,
 200.61763793602853,
 25.155980686050217,
 25.951874574921526,
 221.29396908342503,
 389.98914935504325,
 53.87425716902351,
 55.604837395760754,
 34.23551086139259,
 39.507448345151325,
 25.513286234314634,
 27.476933750662884,
 66.95239565200531,
 163.17313994733541,
 56.12909499341501,
 54.57561271578645]

y_values_green_D3 = [119.30064052002474,
 110.14157826813901,
 168.54473619774862,
 91.51281689581667,
 117.24567058882103,
 198.39858044232614,
 25.155980686050217,
 25.951874574921526,
 134.3506562773693,
 387.27991699266903,
 54.92290482375336,
 43.787992689658594,
 36.41859026620497,
 39.507448345151325,
 25.33931025517881,
 26.87965132677888,
 66.21271934221221,
 163.17313994733541,
 55.001517065894454,
 57.373972450623526]

# Create the plot
fig, ax = plt.subplots(figsize=(10,6))
y1 = 42.16923053975597
y2 = 62.56468631721963
y3 =  23.787802946012743
y4 =  23.30526458817694
y1h = 42.13567171858034
y2h = 62.47026813972497
y3h = 23.70986693761043
y4h = 23.538718979605918
my_plot(fig, ax, y_values, y_values_green, 'Day 1 $\cdot$ major axis $\cdot$ largest % diff. $\cdot$ 5% top and bottom', y1, y2, y3, y4, y1h, y2h, y3h, y4h)

fig2, ax2 = plt.subplots(figsize=(10,6))
y1 = 93.07136686235465
y2 = 143.79455843473463
y3 = 60.65029696439238
y4 =  62.056640783384346
y1h = 87.81538819369486
y2h = 129.76534826029808
y3h = 57.71175019726776
y4h = 60.411730754079095
my_plot(fig2, ax2, y_values_D2, y_values_green_D2, 'Day 2 $\cdot$ major axis $\cdot$ largest % diff. $\cdot$ 5% top and bottom', y1, y2, y3, y4, y1h, y2h, y3h, y4h)

fig3, ax3 = plt.subplots(figsize=(10,6))
y1 = 97.33526487229346
y2 = 122.96677907737221
y3 =  44.77304443333265
y4 =  47.96882726260805
y1h = 90.87721243677805
y2h =  115.59153025656568
y3h = 44.336288059252404
y4h = 49.84522497185141
my_plot(fig3, ax3, y_values_D3, y_values_green_D3, 'Day 3 $\cdot$ major axis $\cdot$ largest % diff. $\cdot$ 5% top and bottom', y1, y2, y3, y4, y1h, y2h, y3h, y4h)

output_path = base_path + '1 Utilities/Misc/'
fig_numbers = plt.get_fignums()
fig.savefig(output_path + 'D1LeibovichComp.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
fig2.savefig(output_path +'D2LeibovichComp.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
fig3.savefig(output_path +'D3LeibovichComp.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF


plt.show()
