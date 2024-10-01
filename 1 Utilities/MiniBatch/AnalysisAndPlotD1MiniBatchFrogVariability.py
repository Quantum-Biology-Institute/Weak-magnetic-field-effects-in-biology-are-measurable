###############################################################################
# BASE PATH
base_path = '/Users/clarice/Desktop/'
###############################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

import gc
# Clear all figures
plt.close('all')
# Clear all variables (garbage collection)
gc.collect()

import sys
sys.path.append(base_path + '1 Utilities/')
import utilities as ut

import plot_config
# Initialize fonts and settings
font_title, font_text, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = plot_config.setup()

###############################################################################
# CHANGE HERE ONLY
###############################################################################

save =  False
fig_width = 6.5 # span latex document which is letter-wide, with 1 in margins

# my_frog = 1
# my_pl = [1,2,6]

my_frog = 2
my_pl = [3,4,5]

fig_length = 5
rotate_indices_C_sol = [] # indices of images to rotate in Control, solidity plot
rotate_indices_H_sol = []       # indices of images to rotate in Hypo, solidity plot
rotate_indices_C_elo = [] # indices of images to rotate in Control, elongation plot
rotate_indices_H_elo = []        # indices of images to rotate in Hypo, elongation plot

###############################################################################
###############################################################################

my_conds = ['C', 'H']
folders = []
output_path_txt = []
output_path_figs = base_path + f"3 D1 quantification/Results/B6Frog{my_frog}/"
        
for my_condition in my_conds:

    # Folders containing the images
    folders.append(base_path + f'3 D1 quantification/B6Frog{my_frog}/B6Frog{my_frog}{my_condition}/')
    output_path_txt.append(base_path + f"3 D1 quantification/Results/B6Frog{my_frog}/B6Frog{my_frog}{my_condition}D1_analysis.txt")

# Check if all images have the same size
#image_size = ut.check_image_sizes(folders, my_batch)

# Data containers
filenames           = [[], []]
areas               = [[], []]
perimeters          = [[], []]
minor_axes          = [[], []]
major_axes          = [[], []]
convex_hull_area    = [[], []]
elongations         = [[], []]
roundnesses         = [[], []]
eccentricities      = [[], []]
solidities          = [[], []]
images_with_overlay = [[], []]
hull_images         = [[], []]

# Process images from both folders
for folder_idx, folder in enumerate(folders):
   for idx, filename in enumerate(sorted([f for f in os.listdir(folder) if f.endswith('.png')])):
       if filename.endswith(".png"): 
           image_path = os.path.join(folder, filename)
           filenames[folder_idx].append(filename)
           results = ut.process_image(image_path)
           areas[folder_idx].append(results[0])
           perimeters[folder_idx].append(results[1])
           minor_axes[folder_idx].append(results[2])
           major_axes[folder_idx].append(results[3])
           convex_hull_area[folder_idx].append(results[4])
           elongations[folder_idx].append(results[5])
           roundnesses[folder_idx].append(results[6])
           eccentricities[folder_idx].append(results[7])
           solidities[folder_idx].append(results[8])
           images_with_overlay[folder_idx].append(results[9])
           hull_images[folder_idx].append(results[10])

# Save the results to text files for each folder
for folder_idx, folder in enumerate(folders):
   with open(output_path_txt[folder_idx], 'w') as f:
       f.write("Filenames:\n")
       f.write(str(filenames[folder_idx]) + "\n")
       f.write("Areas:\n")
       f.write(str(areas[folder_idx]) + "\n")
       f.write("Perimeters:\n")
       f.write(str(perimeters[folder_idx]) + "\n")
       f.write("Minor axes:\n")
       f.write(str(minor_axes[folder_idx]) + "\n")
       f.write("Major axes:\n")
       f.write(str(major_axes[folder_idx]) + "\n")
       f.write("Convex hull areas:\n")
       f.write(str(convex_hull_area[folder_idx]) + "\n")
       f.write("Elongations:\n")
       f.write(str(elongations[folder_idx]) + "\n")
       f.write("Roundnesses:\n")
       f.write(str(roundnesses[folder_idx]) + "\n")
       f.write("Eccentricities:\n")
       f.write(str(eccentricities[folder_idx]) + "\n")
       f.write("Solidities:\n")
       f.write(str(solidities[folder_idx]) + "\n")
       f.write("Images with overlay:\n")
       f.write(str(images_with_overlay[folder_idx]) + "\n")
       f.write("Hull images:\n")
       f.write(str(hull_images[folder_idx]) + "\n")

# Print the number of images processed
print(f"Number of images in Control: {len(images_with_overlay[0])}")
print(f"Number of images in Hypo: {len(images_with_overlay[1])}")

plot_title = f'Batch 6 $\cdot$ Frog {my_frog} $\cdot$ Control vs. Hypo $\cdot$ Day 1'

num_images1 = len(images_with_overlay[0])
num_images2 = len(images_with_overlay[1])
num_cols = 10 # force there to be 10 columns of images per condition; alternative: int(np.ceil(np.sqrt(max(num_images1, num_images2))))
num_rows = num_rows = max(int(np.ceil(num_images1 / num_cols)), int(np.ceil(num_images2 / num_cols)))

# Associate frame with right color
framecolors = [[],[]]
for idx in range(2):
    modified_filenames = []
    status_d3 = []

    for my_plate in my_pl:
        # Construct the path using os.path.join for better handling
        Assessment_path = os.path.join(base_path,f"2 Experimental overview/Assessments/B6/B6{'C' if idx == 0 else 'H'}P{my_plate}_assessment.txt")

        # Read the file and extract the necessary lists
        c_content = ut.read_file(Assessment_path)
        c_filenames = ut.extract_list(c_content, 'Filenames:')
        
        # Modify each filename for comparison: include 'D1' and add '.png'
        modified_filenames.extend([f"{filename[:3]}D1{filename[3:]}.png" for filename in c_filenames])
        status_d3.extend(ut.extract_list(c_content, 'StatusD3:'))

    # Find common filenames
    common_filenames = set(modified_filenames).intersection(set(filenames[idx]))
    # Filter status_d3 based on common filenames
    filtered_status_d3 = [status_d3[modified_filenames.index(fname)] for fname in common_filenames]
    # Map status to colors and save to framecolors
    framecolors[idx] = ut.map_status_to_color(filtered_status_d3)

# Sorting using solidity
footer_txt_sol = 'sorted by decreasing solidity $\cdot$ all images same magnification'
# Data containers
filenames_sol           = [[], []]
solidities_sol          = [[], []]
images_with_overlay_sol = [[], []]
hull_images_sol         = [[], []]
framecolors_sol         = [[], []]

for folder_idx in range(2):
    sorted_indices_sol = np.argsort(solidities[folder_idx])[::-1] # [::-1] indicates decreasing; without is increasing
    filenames_sol[folder_idx] = [filenames[folder_idx][i] for i in sorted_indices_sol]
    solidities_sol[folder_idx] = [solidities[folder_idx][i] for i in sorted_indices_sol]
    images_with_overlay_sol[folder_idx] = [images_with_overlay[folder_idx][i] for i in sorted_indices_sol]
    hull_images_sol[folder_idx] = [hull_images[folder_idx][i] for i in sorted_indices_sol]
    #framecolors_sol[folder_idx] = [framecolors[folder_idx][i] for i in sorted_indices_sol]
    
# Rotate images if needed
for idx in rotate_indices_C_sol:
    images_with_overlay_sol[0][idx] = scipy.ndimage.rotate(images_with_overlay_sol[0][idx], 180)
    hull_images_sol[0][idx] = scipy.ndimage.rotate(hull_images_sol[0][idx], 180)

for idx in rotate_indices_H_sol:
    images_with_overlay_sol[1][idx] = scipy.ndimage.rotate(images_with_overlay_sol[1][idx], 180)
    hull_images_sol[1][idx] = scipy.ndimage.rotate(hull_images_sol[1][idx], 180)
    
    
fig_sol = plt.figure(figsize=(fig_width, fig_length))
### JUST FOR 7 and 8
framecolors_sol = framecolors
### JUST FOR 7 and 8
ut.plot_D1(fig_sol, num_rows, num_cols, num_images1, num_images2, images_with_overlay_sol, plot_title, footer_txt_sol, framecolors_sol, BIGGER_SIZE, MEDIUM_SIZE, font_title, font_text, rotate_indices_C_sol, rotate_indices_H_sol)

fig_sol_hull = plt.figure(figsize=(fig_width, fig_length))
ut.plot_D1(fig_sol_hull, num_rows, num_cols, num_images1, num_images2, images_with_overlay_sol, plot_title, footer_txt_sol, framecolors_sol, BIGGER_SIZE, MEDIUM_SIZE, font_title, font_text, rotate_indices_C_sol, rotate_indices_H_sol, hull_images_sol )

# Sorting using elongation
footer_txt_elo = 'sorted by increasing elongation $\cdot$ all images same magnification'
# Data containers
filenames_elo            = [[], []]
elongations_elo          = [[], []]
images_with_overlay_elo  = [[], []]
framecolors_elo          = [[], []]

for folder_idx in range(2):
    sorted_indices_elo = np.argsort(elongations[folder_idx]) 
    filenames_elo[folder_idx] = [filenames[folder_idx][i] for i in sorted_indices_elo]
    elongations_elo[folder_idx] = [elongations[folder_idx][i] for i in sorted_indices_elo]
    images_with_overlay_elo[folder_idx] = [images_with_overlay[folder_idx][i] for i in sorted_indices_elo]
    framecolors_elo[folder_idx] = [framecolors[folder_idx][i] for i in sorted_indices_elo]
    
# Rotate images if needed
for idx in rotate_indices_C_elo:
    images_with_overlay_elo[0][idx] = scipy.ndimage.rotate(images_with_overlay_elo[0][idx], 180)

for idx in rotate_indices_H_elo:
    images_with_overlay_elo[1][idx] = scipy.ndimage.rotate(images_with_overlay_elo[1][idx], 180)
    
fig_elo = plt.figure(figsize=(fig_width, fig_length))
ut.plot_D1(fig_elo, num_rows, num_cols, num_images1, num_images2, images_with_overlay_elo, plot_title, footer_txt_elo, framecolors_elo, BIGGER_SIZE, MEDIUM_SIZE, font_title, font_text, rotate_indices_C_elo, rotate_indices_H_elo)

###############################################################################

# Save all open figures
if save:
    FigName = [f'B6Frog{my_frog}SoliditySorted', f'B6Frog{my_frog}HullImages', f'B6Frog{my_frog}ElongationSorted']
    for i, fig in enumerate(map(plt.figure, plt.get_fignums())):
        fig.savefig(output_path_figs + f'{FigName[i]}.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
        fig.savefig(output_path_figs + f'{FigName[i]}.png', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PNG

plt.show()