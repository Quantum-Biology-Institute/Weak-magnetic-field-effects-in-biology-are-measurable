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

#my_frog = 1

my_frog = 2

###############################################################################
###############################################################################

plot_title = f'Batch6 $\cdot$ Frog {my_frog} $\cdot$ Control vs. Hypo $\cdot$ Day 2'
footer_txt = 'sorted by increasing max. distance $\cdot$ all images same magnification'

file_paths = [base_path + f"4 D2 quantification/B6Frog{my_frog}/B6Frog{my_frog}{'C' if i == 0 else 'H'}" for i in range(2)]
output_path = base_path + f"4 D2 quantification/Results/B6Frog{my_frog}/"

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

#mean_R_C, mean_G_C, mean_B_C, mean_L_C, mean_A_C, mean_BB_C, masked_images_C, std_R_C, std_G_C, std_B_C, std_L_C, std_A_C, std_BB_C = ut.process_images_for_colors(folderC_images,global_threshold)
#mean_R_H, mean_G_H, mean_B_H, mean_L_H, mean_A_H, mean_BB_H, masked_images_H, std_R_H, std_G_H, std_B_H, std_L_H, std_A_H, std_BB_H = ut.process_images_for_colors(folderH_images,global_threshold)
masked_images_C_np = [np.array(img) for img in masked_images_C]
masked_images_H_np = [np.array(img) for img in masked_images_H]
ut.plot_images_custom_aspect(masked_images_C_np, masked_images_H_np, aspect_ratio, plot_title + '$\cdot$ Masks of binarized white', footer_txt, font_title, font_text, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE)

### Get yellows
yellows = [[],[]]
yellows[0], folderC_yellow = ut.calculate_mean_yellow_colors(folderC_images)
yellows[1], folderH_yellow = ut.calculate_mean_yellow_colors(folderH_images)
### Plot yellows
ut.plot_yellow_bands_custom_aspect(folderC_yellow, folderH_yellow, aspect_ratio, plot_title + '$\cdot$ Masks of yellow hues', footer_txt, font_title, font_text, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE)

### Save the results to text files for each folder
with open(output_path + f'B6Frog{my_frog}CD2_analysis.txt', 'w') as f:
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

    
with open(output_path + f'B6Frog{my_frog}HD2_analysis.txt', 'w') as f:
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


# Save all open figures
if save:
    FigName = ['LengthSorted', 'LengthSortedLongestLine', 'LengthSortedYellowMask']
    for i, fig in enumerate(map(plt.figure, plt.get_fignums())):
        fig.savefig(output_path + f'{FigName[i]}.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
        fig.savefig(output_path + f'{FigName[i]}.png', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PNG

plt.show()
