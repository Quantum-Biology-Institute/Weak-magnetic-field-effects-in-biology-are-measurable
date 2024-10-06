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
MEDIUM_SIZE = 24#20#14 # size of main text
BIGGER_SIZE = 28#20 # size of section text

###############################################################################

# def my_plot(fig, ax, y_values, y_values_green, title, y1, y2, y3, y4, y1h, y2h, y3h, y4h):
      # Old: way to call it
      #     #result,          comb1,  comb2,  comb3,  comb4 = ut.calculate_percent_diff(concatenated, 5)

#     # X labels for the plot
#     x_labels = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B1--7", "+1", "+2", "+3", "+1--3"]
    
#     # Extract even and odd indexed values separately for plotting
#     y_values_even = y_values[::2]  # Even-indexed values (0, 2, 4, ...)
#     y_values_odd  = y_values[1::2]  # Odd-indexed values (1, 3, 5, ...)
    
#     y_values_even_green = y_values_green[::2]  # Even-indexed values (0, 2, 4, ...)
#     y_values_odd_green  = y_values_green[1::2]  # Odd-indexed values (1, 3, 5, ...)
    
#     # Set x positions for the pairs; both even and odd will share the same x positions
#     x_positions = np.array([0,1,2,3,4,5,6,8,9,10])#np.arange(len(x_labels))  # x positions
    
#     shift = 0.1
#     ms = 150 #was 50
    
#     # Plot the even indexed y_values with round markers
#     plt.scatter(x_positions-shift, y_values_even, marker='o', color='blue', s=ms, label='control $\cdot$ all states')
#     plt.scatter(x_positions+shift, y_values_even_green, marker='o', color='green', s=ms, label='control $\cdot$ only healthy')
    
#     # Plot the odd indexed y_values with square markers
#     plt.scatter(x_positions-shift, y_values_odd, marker='s', color='blue', s=ms, label='hypomagnetic $\cdot$ all states')
#     plt.scatter(x_positions+shift, y_values_odd_green, marker='s', color='green', s=ms, label='hypomagnetic $\cdot$ only healthy')
    
#     # Plot averages at x = 7 and x = 11
#     plt.scatter(7-shift, y1, marker='o', color='blue', s=ms)
#     plt.scatter(7+shift, y1h, marker='o', color='green', s=ms)
#     plt.scatter(7-shift, y2, marker='s', color='blue', s=ms)
#     plt.scatter(7+shift, y2h, marker='s', color='green', s=ms)
    
#     plt.scatter(11-shift, y3, marker='o', color='blue', s=ms)
#     plt.scatter(11+shift, y3h, marker='o', color='green', s=ms)
#     plt.scatter(11-shift, y4, marker='s', color='blue', s=ms)
#     plt.scatter(11+shift, y4h, marker='s', color='green', s=ms)
    
#     # Add labels, title, and format x-axis
#     plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11], x_labels, fontsize=12)
#     #plt.xlabel('X Labels', fontsize=14)
#     #plt.ylabel('Y Values', fontsize=14)
    
#     my_font_title = FontProperties(fname=font_title.get_file(), size=BIGGER_SIZE) 
#     ax.set_title(title, fontproperties=my_font_title)
    
#     # Add a legend
#     my_font_leg = FontProperties(fname=font_text.get_file(), size=1.7*SMALL_SIZE) 
#     leg = plt.legend(loc='upper right', bbox_to_anchor=(1, 1.05), frameon=True, facecolor='white', edgecolor='none', prop=my_font_leg)
#     leg.get_frame().set_alpha(1.0)
    
#     # Add a black vertical dashed line at x=7.5
#     my_font_line = FontProperties(fname=font_text.get_file(), size=1.7*SMALL_SIZE) 
#     plt.axvline(x=7.5, color='black', linestyle='--', linewidth=1)
#     plt.axhline(y=38, color='red', linestyle='--', linewidth=1)
#     plt.text(plt.gca().get_xlim()[1], 38, 'Leibovich (20)', color='red', 
#          va='top', ha='right', fontproperties=my_font_line)
    
#     # Customizing font properties for x and y tick labels
#     for my_label in ax.get_xticklabels():
#         my_label.set_fontproperties(font_text)
#         my_label.set_fontsize(MEDIUM_SIZE)
    
#     for my_label in ax.get_yticklabels():
#         my_label.set_fontproperties(font_text)
#         my_label.set_fontsize(MEDIUM_SIZE)
    
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
    
#     # Show the plot
#     plt.tight_layout()
    
# def my_plot_new(fig, ax, y_values, title, y1, y2, y3, y4, my_color, ymin, ymax):

#     # X labels for the plot
#     x_labels = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B1--7", "+1", "+2", "+3", "+1--3"]
    
#     # Extract even and odd indexed values separately for plotting
#     y_values_even = y_values[::2]  # Even-indexed values (0, 2, 4, ...)
#     y_values_odd  = y_values[1::2]  # Odd-indexed values (1, 3, 5, ...)
    
#     # Set x positions for the pairs; both even and odd will share the same x positions
#     x_positions = np.array([0,1,2,3,4,5,6,8,9,10])#np.arange(len(x_labels))  # x positions
    
#     shift = 0
#     ms = 150 #was 50
    
#     # Plot the even indexed y_values with round markers
#     plt.scatter(x_positions-shift, y_values_even, marker='o', color=my_color, s=ms, label='control $\cdot$ all states')
    
#     # Plot the odd indexed y_values with square markers
#     plt.scatter(x_positions-shift, y_values_odd, marker='s', color=my_color, s=ms, label='hypomagnetic $\cdot$ all states')
    
#     # Plot averages at x = 7 and x = 11
#     plt.scatter(7-shift, y1, marker='o', color=my_color, s=ms)
#     plt.scatter(7-shift, y2, marker='s', color=my_color, s=ms)
    
#     plt.scatter(11-shift, y3, marker='o', color=my_color, s=ms)
#     plt.scatter(11-shift, y4, marker='s', color=my_color, s=ms)
    
#     # Add labels, title, and format x-axis
#     plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11], x_labels, fontsize=12)
#     #plt.xlabel('X Labels', fontsize=14)
#     #plt.ylabel('Y Values', fontsize=14)
    
#     my_font_title = FontProperties(fname=font_title.get_file(), size=BIGGER_SIZE) 
#     ax.set_title(title, fontproperties=my_font_title)
    
#     # Add a legend
#     my_font_leg = FontProperties(fname=font_text.get_file(), size=1.7*SMALL_SIZE) 
#     leg = plt.legend(loc='upper right', bbox_to_anchor=(1, 1.05), frameon=True, facecolor='white', edgecolor='none', prop=my_font_leg)
#     leg.get_frame().set_alpha(1.0)
    
#     # Add a black vertical dashed line at x=7.5
#     my_font_line = FontProperties(fname=font_text.get_file(), size=1.7*SMALL_SIZE) 
#     plt.axvline(x=7.5, color='black', linestyle='--', linewidth=1)
#     plt.axhline(y=38, color='red', linestyle='--', linewidth=1)
#     plt.text(plt.gca().get_xlim()[1], 38, 'Leibovich (20)', color='red', 
#          va='top', ha='right', fontproperties=my_font_line)
    
#     # Customizing font properties for x and y tick labels
#     for my_label in ax.get_xticklabels():
#         my_label.set_fontproperties(font_text)
#         my_label.set_fontsize(MEDIUM_SIZE)
    
#     for my_label in ax.get_yticklabels():
#         my_label.set_fontproperties(font_text)
#         my_label.set_fontsize(MEDIUM_SIZE)
    
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
    
#     # Show the plot
#     plt.tight_layout()

# Modified plot function to handle y_values and y1, y2, y3, y4 with standard deviations
def my_plot_new_with_avg_std(fig, ax, y_values, title, y1, y2, y3, y4, my_color, ymin, ymax, yticks, pos):
    """
    Function to plot values and their corresponding standard deviations, including y1, y2, y3, y4.
    
    Parameters:
    - y_values: List of tuples (value, std) for each point.
    - y1, y2, y3, y4: Tuples for averages to be plotted with their std (value, std).
    """
    
    # X labels for the plot
    x_labels = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B1--7", "+1", "+2", "+3", "+1--3"]
    
    # Extract even and odd indexed values and stds
    y_values_even = [val[0] for val in y_values[::2]]  
    y_std_even = [val[1] for val in y_values[::2]]     
    y_values_odd  = [val[0] for val in y_values[1::2]]  
    y_std_odd  = [val[1] for val in y_values[1::2]]     
    
    x_positions = np.array([0,1,2,3,4,5,6,8,9,10])  
    
    shift = 0.05
    ms = 150  
    
    # Plot even and odd y_values with their std
    ax.errorbar(x_positions-shift, y_values_even, yerr=y_std_even, fmt='o', color=my_color, 
                markersize=ms/10, label='avg. C ± std.' , capsize=0)
    ax.errorbar(x_positions+shift, y_values_odd, yerr=y_std_odd, fmt='s', color=my_color, 
                markersize=ms/10, label='avg. H ± std.', capsize=0)
    
    # Plot averages with std at x = 7 and x = 11
    ax.errorbar(7-shift, y1[0], yerr=y1[1], fmt='o', color=my_color, markersize=ms/10, capsize=0)
    ax.errorbar(7+shift, y2[0], yerr=y2[1], fmt='s', color=my_color, markersize=ms/10, capsize=0)
    
    ax.errorbar(11-shift, y3[0], yerr=y3[1], fmt='o', color=my_color, markersize=ms/10, capsize=0)
    ax.errorbar(11+shift, y4[0], yerr=y4[1], fmt='s', color=my_color, markersize=ms/10, capsize=0)
    
    # X-axis customization
    plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11], x_labels, fontsize=12)
    
    # Title and legend
    my_font_title = FontProperties(fname=font_title.get_file(), size=BIGGER_SIZE) 
    ax.set_title(title, fontproperties=my_font_title)
    
    my_font_leg = FontProperties(fname=font_text.get_file(), size=MEDIUM_SIZE) 
    leg = plt.legend(loc='upper right', bbox_to_anchor=(1, 1.05), frameon=True, facecolor='white', edgecolor='none', prop=my_font_leg)
    leg.get_frame().set_alpha(1.0)
    
    # Add black vertical dashed line and text
    my_font_line = FontProperties(fname=font_text.get_file(), size=MEDIUM_SIZE) 
    plt.axvline(x=7.5, color='black', linestyle='--', linewidth=1)
    plt.axhline(y=38, color='red', linestyle='--', linewidth=1)
    plt.text(plt.gca().get_xlim()[1], 38, 'Leibovich (20)', color='red', va=pos, ha='right', fontproperties=my_font_line)
    
    ax.set_ylim([ymin, ymax])
    ax.set_yticks(yticks)
    
    # Customizing font properties for ticks
    for my_label in ax.get_xticklabels():
        my_label.set_fontproperties(font_text)
        my_label.set_fontsize(MEDIUM_SIZE)
    
    for my_label in ax.get_yticklabels():
        my_label.set_fontproperties(font_text)
        my_label.set_fontsize(MEDIUM_SIZE)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
   
    
    plt.tight_layout()

###############################################################################
# Day 1, Major Axis difference in percent, same order as concatenated
y_values = [(37.734636605796716, 3.9030172307028477),
 (52.134548098836916, 14.4272437908714),
 (45.8952162345755, 9.506626072333962),
 (79.98092637807827, 9.83589431830404),
 (76.58588121890132, 7.56294283911955),
 (101.57244544238348, 23.871478850919114),
 (10.4721198866134, 0.0),
 (17.633795341224786, 0.0),
 (26.453119338109826, 0.9812404309245558),
 (37.63876829590645, 8.315690497421839),
 (26.319753978439902, 12.309368846152276),
 (23.382090157159414, 4.764861700158687),
 (33.839115645960405, 19.126459796280617),
 (29.985846098272866, 6.793340556484826),
 (19.058012881101945, 4.230974158609236),
 (17.281812922082132, 2.2400154350123773),
 (13.120118926829832, 1.4860819236046425),
 (24.75996906319078, 7.468198523234377),
 (26.250422323116375, 5.003003085073777),
 (25.077925954617815, 7.155153681617923)]

y_values_green = [(36.538586330760836, 3.1879736147177153),
 (53.71526984644737, 16.029914980793013),
 (49.56113696118537, 10.102876653202877),
 (79.78490120624045, 10.007728381475022),
 (76.23477437377497, 6.2126932782524005),
 (99.4499866655633, 28.120028230923708),
 (10.4721198866134, 0.0),
 (17.633795341224786, 0.0),
 (26.453119338109826, 0.9812404309245558),
 (28.645378562204243, 5.538924177088565),
 (24.55641334748229, 12.25324191314107),
 (24.490190354578306, 4.90489476342304),
 (36.63659705145878, 20.078714456345523),
 (29.985846098272866, 6.793340556484826),
 (19.058012881101945, 4.230974158609236),
 (17.281812922082132, 2.2400154350123773),
 (13.120118926829832, 1.4860819236046425),
 (24.75996906319078, 7.468198523234377),
 (26.03758474634659, 5.12349258079263),
 (25.37581626921717, 7.245931279294251)]

y1 =  (42.16923053975597, 17.085069470140144)
y2 =  (62.56468631721963, 21.567021707586775)
y3 =  (23.787802946012743, 4.993897344230226)
y4 =  (23.30526458817694, 6.139639418076394)
y1h = (42.06178038220535, 18.506217662291487)
y2h = (62.385582163631206, 22.6903139303496)
y3h = (23.67214374268858, 5.0625806652347345)
y4h = (23.80155761772941, 6.287109470723287)

ymin = 0
ymax = 125
yticks = [25,50,75,100]
fig, ax   = plt.subplots(figsize=(10,6))
my_plot_new_with_avg_std( fig,  ax, y_values,'a) Day 1 $\cdot$ major axis $\cdot$ all states \u00A0\u00A0\u00A0\u00A0\u00A0\u00A0 \n largest % diff. $\cdot$ top and bottom 5%\n',   y1,  y2,  y3,  y4,  'blue', ymin, ymax, yticks, 'bottom')
fig2, ax2 = plt.subplots(figsize=(10,6))
my_plot_new_with_avg_std(fig2, ax2, y_values_green,'a) Day 1 $\cdot$ major axis $\cdot$ only healthy \n largest % diff. $\cdot$ top and bottom 5%\n',                               y1h, y2h, y3h, y4h, 'green', ymin, ymax, yticks, 'bottom')

###############################################################################

# Major Axis Day 2 in percent, same order as concatenated
y_values_D2 = [(115.99460756885911, 7.78514792712537),
 (132.40100273934047, 54.646525219746614),
 (141.62980777153976, 83.12894263699275),
 (53.58886682740634, 10.09368926018543),
 (70.38795133747908, 20.336634109830197),
 (111.38270390012417, 19.27636898972725),
 (84.44243976712146, 0.0),
 (154.1840007381776, 0.0),
 (144.2101187971346, 73.09944081885196),
 (225.70344881769424, 95.4141635658968),
 (66.17212859632397, 18.553770518236266),
 (89.47760418963759, 38.08883814158535),
 (34.92339591558395, 6.8532673050618085),
 (34.78098461597307, 5.880379111523219),
 (39.73493737541127, 10.611845989738635),
 (45.138895814516374, 27.567641349988964),
 (77.15083815789683, 32.3635006925636),
 (68.04167600431398, 39.713247302127165),
 (71.47746631498052, 32.444401688877974),
 (69.86118707833172, 35.20392922747166)]

y_values_green_D2 = [(122.48117524372188, 3.7996712438299327),
 (78.94197500838837, 36.104187559859966),
 (194.04315493357055, 60.75110617364645),
 (55.2973518207209, 11.490459652159),
 (67.43579772053808, 23.307941943453113),
 (103.00212868024306, 11.456577405370146),
 (84.44243976712146, 0.0),
 (154.1840007381776, 0.0),
 (144.2101187971346, 73.09944081885196),
 (225.70344881769424, 95.4141635658968),
 (64.48652277162923, 19.431638956257796),
 (90.10066757947581, 40.47224767987859),
 (37.35530901461758, 6.1536286051983335),
 (34.58611928022588, 5.979286147787986),
 (39.73493737541127, 10.611845989738635),
 (46.40213038724576, 28.71074873536749),
 (77.15083815789683, 32.3635006925636),
 (68.04167600431398, 39.713247302127165),
 (74.70482260588314, 33.73235669799846),
 (66.80495403047779, 30.856872923634032)]

y1 =   (93.07136686235465, 36.63666004552457)
y2 =   (143.79455843473463, 52.49915212403842)
y3 =   (60.65029696439238, 27.737941987277626)
y4 =   (62.056640783384346, 33.870430687442365)
y1h =  (90.61713312833749, 36.67934744860241)
y2h =  (139.93747296701102, 53.19596370209272)
y3h =  (62.67039181559676, 28.847513884409928)
y4h =  (62.06273639347274, 32.068701082751936)

ymin = 0
ymax = 300
yticks = [50, 150, 250]
fig3, ax3   = plt.subplots(figsize=(10,6))
my_plot_new_with_avg_std( fig3,  ax3, y_values_D2,'b) Day 2 $\cdot$ major axis $\cdot$ all states \u00A0\u00A0\u00A0\u00A0\u00A0\u00A0 \n largest % diff. $\cdot$ top and bottom 5%\n',   y1,  y2,  y3,  y4,  'blue', ymin, ymax, yticks, 'top')
fig4, ax4 = plt.subplots(figsize=(10,6))
my_plot_new_with_avg_std(fig4, ax4, y_values_green_D2,'b) Day 2 $\cdot$ major axis $\cdot$ only healthy \n largest % diff. $\cdot$ top and bottom 5%\n',                               y1h, y2h, y3h, y4h, 'green', ymin, ymax, yticks, 'top')




# Major Axis Day 3 in percent, same order as concatenated
y_values_D3 = [(122.04793005112768, 54.05142946054094),
 (125.32750598544786, 39.59491502249009),
 (139.84353033680063, 68.67618260073762),
 (91.51281689581667, 21.567957860444725),
 (115.819417585431, 40.07642699567147),
 (200.61763793602853, 21.237149892627645),
 (25.155980686050217, 0.0),
 (25.951874574921526, 0.0),
 (221.29396908342503, 17.132243273018087),
 (389.98914935504325, 6.847031729833434),
 (53.87425716902351, 29.991417406984272),
 (55.604837395760754, 40.81991082471558),
 (34.23551086139259, 6.5462324061594135),
 (39.507448345151325, 21.753037515822296),
 (25.513286234314634, 2.968222390666444),
 (27.476933750662884, 7.407116193489756),
 (66.95239565200531, 13.555264982061427),
 (163.17313994733541, 52.71807505447312),
 (56.12909499341501, 42.80252781142581),
 (54.57561271578645, 38.95043751711744)]

y_values_green_D3 = [(169.0396065035169, 47.75916377454181),
 (146.72903404216987, 25.651425562549292),
 (77.36084185059234, 15.672410690867517),
 (91.51281689581667, 21.567957860444725),
 (102.53979189873273, 62.49015635376306),
 (209.81753478306624, 20.316768018035678),
 (25.155980686050217, 0.0),
 (25.951874574921526, 0.0),
 (142.74211498524713, 63.09988020347366),
 (389.98914935504325, 6.847031729833434),
 (53.3132391101797, 31.02705539716151),
 (42.95656546407099, 26.068459718213035),
 (36.317478999650014, 6.177027242616022),
 (29.219559549296005, 5.1046568232138965),
 (25.16694822731715, 3.0052378269759874),
 (27.422797761744345, 7.434018318542464),
 (66.95239565200531, 13.555264982061427),
 (163.17313994733541, 52.71807505447312),
 (55.44215225462889, 41.576853160893954),
 (52.48287836066221, 36.35004421669441)]

y1 =   (97.33526487229346, 48.98844120961611)
y2 =   (122.96677907737221, 56.947584206720435)
y3 =   (44.77304443333265, 32.63289053077345)
y4 =   (47.96882726260805, 34.342852573124894)
y1h =  (84.57159667820369, 43.90163530922534)
y2h =  (124.4224967644139, 54.91101325343964)
y3h =  (44.43830769470223, 31.804284089877925)
y4h =  (47.31015937067045, 33.0905182491007)

ymin = 0
ymax = 300
yticks = [50, 150, 250]
fig5, ax5   = plt.subplots(figsize=(10,6))
my_plot_new_with_avg_std( fig5,  ax5, y_values_D2,'c) Day 3 $\cdot$ major axis $\cdot$ all states \u00A0\u00A0\u00A0\u00A0\u00A0\u00A0 \n largest % diff. $\cdot$ top and bottom 5%\n',   y1,  y2,  y3,  y4,  'blue', ymin, ymax, yticks, 'top')
fig6, ax6 = plt.subplots(figsize=(10,6))
my_plot_new_with_avg_std(fig6, ax6, y_values_green_D2,'c) Day 3 $\cdot$ major axis $\cdot$ only healthy \n largest % diff. $\cdot$ top and bottom 5%\n',                               y1h, y2h, y3h, y4h, 'green', ymin, ymax, yticks, 'top')


output_path = base_path + '1 Utilities/Misc/'
fig_numbers = plt.get_fignums()
fig.savefig(output_path + 'D1LeibovichComp.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
fig2.savefig(output_path +'D1LeibovichCompH.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
fig3.savefig(output_path +'D2LeibovichComp.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
fig4.savefig(output_path + 'D2LeibovichCompH.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
fig5.savefig(output_path +'D3LeibovichComp.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
fig6.savefig(output_path +'D3LeibovichCompH.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF


plt.show()
