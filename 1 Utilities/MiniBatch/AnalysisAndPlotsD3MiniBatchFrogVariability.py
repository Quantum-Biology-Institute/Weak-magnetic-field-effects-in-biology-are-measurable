###############################################################################
# BASE PATH
base_path = '/Users/clarice/Desktop/'
###############################################################################

import matplotlib.pyplot as plt
import numpy as np

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

#### COLOR??????????????

save = False

my_frog = 1

#my_frog = 2

###############################################################################
###############################################################################

plot_title = f'Batch 6  $\cdot$ Frog {my_frog} $\cdot$ Control vs. Hypo $\cdot$ Day 3'
footer_txt = 'sorted by increasing max. distance $\cdot$ all images same magnification'

file_paths = [base_path + f"/5 D3 quantification/B6Frog{my_frog}/B6Frog{my_frog}{'C' if i == 0 else 'H'}" for i in range(2)]
output_path = base_path + f"5 D3 quantification/Results/B6Frog{my_frog}/"

### Check that resize was done
#ut.check_image_sizes(file_paths, my_batch)

### Process the two folders with D2 images
folderC_images, folderC_binaries, folderC_maxlengths, folderC_filenames = ut.process_folder_D2(file_paths[0])
folderH_images, folderH_binaries, folderH_maxlengths, folderH_filenames = ut.process_folder_D2(file_paths[1])

### Plot color images
aspect_ratio = round(folderC_images[0].shape[1]/folderC_images[0].shape[0], 2)
ut.plot_images_custom_aspect(folderC_images, folderH_images, aspect_ratio, plot_title, footer_txt, font_title, font_text, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE)

### Plot the masked images in a new figure
ut.plot_binary_images(folderC_binaries, folderH_binaries, aspect_ratio, plot_title, footer_txt, font_title, font_text, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE)

### Find best global threshold to binarize the images
global_threshold = ut.calculate_global_threshold(folderC_images + folderH_images)
print(f"Chosen Global Threshold: {global_threshold}")

mean_R_C, mean_G_C, mean_B_C, mean_L_C, mean_A_C, mean_BB_C, masked_images_C, inverse_masked_images_C, std_R_C, std_G_C, std_B_C, std_L_C, std_A_C, std_BB_C = ut.process_images_for_colors(folderC_images, global_threshold)
mean_R_H, mean_G_H, mean_B_H, mean_L_H, mean_A_H, mean_BB_H, masked_images_H, inverse_masked_images_H, std_R_H, std_G_H, std_B_H, std_L_H, std_A_H, std_BB_H = ut.process_images_for_colors(folderH_images, global_threshold)
masked_images_C_np = [np.array(img) for img in masked_images_C]
masked_images_H_np = [np.array(img) for img in masked_images_H]
inverse_masked_images_C_np = [np.array(img) for img in inverse_masked_images_C]
inverse_masked_images_H_np = [np.array(img) for img in inverse_masked_images_H]

ut.plot_images_custom_aspect(masked_images_C_np, masked_images_H_np, aspect_ratio, plot_title + '$\cdot$ Masks of binarized white', footer_txt, font_title, font_text, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE)

ut.plot_images_custom_aspect(inverse_masked_images_C_np, inverse_masked_images_H_np, aspect_ratio, plot_title + '$\cdot$ Masks of binarized white', footer_txt, font_title, font_text, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE)

min_size = 30 #min side size of feature; makes it so tiny areas are not chosen
max_size = 75
areas_C, areas_H, best_contours_C, best_contours_H = ut.plot_closest_circle_areas_custom_aspect_new(
    inverse_masked_images_C_np, 
    inverse_masked_images_H_np, 
    min_size,
    max_size,
    aspect_ratio, 
    plot_title, 
    footer_txt, 
    font_title, 
    font_text, 
    SMALL_SIZE, 
    MEDIUM_SIZE, 
    BIGGER_SIZE)

non_circle_areas_C, non_circle_areas_H = ut.plot_non_circle_areas(
    inverse_masked_images_C_np, 
    inverse_masked_images_H_np, 
    best_contours_C, 
    best_contours_H, 
    aspect_ratio, 
    plot_title, 
    footer_txt, 
    font_title, 
    font_text, 
    SMALL_SIZE, 
    MEDIUM_SIZE, 
    BIGGER_SIZE)

# =============================================================================
# # Remove points for which eye segmentation didn't work
# # Mapping batch numbers to their corresponding indices
# batch_indices = {
#     4: {
#         "C": [8, 9],
#         "H": [5]
#     },
#     5: {
#         "C": [0,2,3,24,27,37,39],  # Add the indices for batch 5, C
#         "H": [1,4,7,20,22,23,24,25,26,27,29,31,34,37, 38,39]   # Add the indices for batch 5, H
#     },
#     # Add other batches as needed
# }
# =============================================================================

# =============================================================================
# # Remove points for which eye segmentation didn't work
# if my_batch in batch_indices:
#     for my_C_index in batch_indices[my_batch]["C"]:
#         areas_C[my_C_index] = np.nan
#         non_circle_areas_C[my_C_index] = np.nan
#     
#     for my_H_index in batch_indices[my_batch]["H"]:
#         areas_H[my_H_index] = np.nan
#         non_circle_areas_H[my_H_index] = np.nan   
# 
# =============================================================================
### Get yellows
yellows = [[],[]]
yellows[0], folderC_yellow = ut.calculate_mean_yellow_colors(folderC_images)
yellows[1], folderH_yellow = ut.calculate_mean_yellow_colors(folderH_images)
### Plot yellows
ut.plot_yellow_bands_custom_aspect(folderC_yellow, folderH_yellow, aspect_ratio, plot_title + '$\cdot$ Masks of yellow hues', footer_txt, font_title, font_text, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE)

### Binarize the images using global threshold
binC = []
binH = []
for image in folderC_images:
    binC.append(ut.apply_fixed_threshold(image,global_threshold))
for image in folderH_images:
    binH.append(ut.apply_fixed_threshold(image,global_threshold))
    
### Plot the binary images in a new figure
ut.plot_binary_images_custom_aspect(binC, binH, aspect_ratio, plot_title, footer_txt, font_title, font_text, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE)

### Plot areas (a.k.a. "false features") found 
#areasC, areasH = ut.plot_false_features_custom_aspect(binC, binH, aspect_ratio, plot_title, footer_txt, font_title, font_text, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE)

### Save the results to text files for each folder
with open(output_path + f'B6Frog{my_frog}CD3_analysis.txt', 'w') as f:
    f.write("Filenames:\n")
    f.write(str(folderC_filenames) + "\n")
    f.write("Max lengths:\n")
    f.write(str(folderC_maxlengths) + "\n")
    f.write("Yellow R:\n")
    f.write(str([item[0] for item in yellows[0]]) + "\n")
    f.write("Yellow G:\n")
    f.write(str([item[1] for item in yellows[0]]) + "\n")
    f.write("Yellow B:\n")
    f.write(str([item[2] for item in yellows[0]]) + "\n")
    
    f.write("Mean R:\n")
    f.write(str(mean_R_C) + "\n")
    f.write("Mean G:\n")
    f.write(str(mean_G_C) + "\n")
    f.write("Mean B:\n")
    f.write(str(mean_B_C) + "\n")
    f.write("Mean L:\n")
    f.write(str(mean_L_C) + "\n")
    f.write("Mean A:\n")
    f.write(str(mean_A_C) + "\n")
    f.write("Mean BB:\n")
    f.write(str(mean_BB_C) + "\n")    
    
    f.write("Std R:\n")
    f.write(str(std_R_C) + "\n")
    f.write("Std G:\n")
    f.write(str(std_G_C) + "\n")
    f.write("Std B:\n")
    f.write(str(std_B_C) + "\n")
    f.write("Std L:\n")
    f.write(str(std_L_C) + "\n")
    f.write("Std A:\n")
    f.write(str(std_A_C) + "\n")
    f.write("Std BB:\n")
    f.write(str(std_BB_C) + "\n")
    
    f.write("Eye Areas:\n")
    f.write(str(areas_C) + "\n")
    f.write("Pigmentation Areas:\n")
    f.write(str(non_circle_areas_C) + "\n")
    
with open(output_path + f'B6Frog{my_frog}HD3_analysis.txt', 'w') as f:
    f.write("Filenames:\n")
    f.write(str(folderH_filenames) + "\n")
    f.write("Max lengths:\n")
    f.write(str(folderH_maxlengths) + "\n")
    f.write("Yellow R:\n")
    f.write(str([item[0] for item in yellows[1]]) + "\n")
    f.write("Yellow G:\n")
    f.write(str([item[1] for item in yellows[1]]) + "\n")
    f.write("Yellow B:\n")
    f.write(str([item[2] for item in yellows[1]]) + "\n")
    
    f.write("Mean R:\n")
    f.write(str(mean_R_H) + "\n")
    f.write("Mean G:\n")
    f.write(str(mean_G_H) + "\n")
    f.write("Mean B:\n")
    f.write(str(mean_B_H) + "\n")
    f.write("Mean L:\n")
    f.write(str(mean_L_H) + "\n")
    f.write("Mean A:\n")
    f.write(str(mean_A_H) + "\n")
    f.write("Mean BB:\n")
    f.write(str(mean_BB_H) + "\n")
    
    f.write("Std R:\n")
    f.write(str(std_R_H) + "\n")
    f.write("Std G:\n")
    f.write(str(std_G_H) + "\n")
    f.write("Std B:\n")
    f.write(str(std_B_H) + "\n")
    f.write("Std L:\n")
    f.write(str(std_L_H) + "\n")
    f.write("Std A:\n")
    f.write(str(std_A_H) + "\n")
    f.write("Std BB:\n")
    f.write(str(std_BB_H) + "\n")
    
    f.write("Eye Areas:\n")
    f.write(str(areas_H) + "\n")
    f.write("Pigmentation Areas:\n")
    f.write(str(non_circle_areas_H) + "\n")

# Save all open figures
if save:
    FigName = ['B6Frog{my_frog}LengthSorted', 'B6Frog{my_frog}LengthSortedLongestLine', 'B6Frog{my_frog}LengthSortedRGBMask','B6Frog{my_frog}LengthSortedYellowMask']
    for i, fig in enumerate(map(plt.figure, plt.get_fignums())):
        fig.savefig(output_path + f'{FigName[i]}.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
        fig.savefig(output_path + f'{FigName[i]}.png', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PNG

plt.show()
