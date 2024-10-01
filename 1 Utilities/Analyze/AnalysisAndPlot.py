###############################################################################
# BASE PATH
base_path = '/Users/clarice/Desktop/'
###############################################################################

import os
import matplotlib.pyplot as plt
import numpy as np
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


save = False

D1 = False
D2 = False
D3 = True

#D1: 1 thru 10 done on 20240911, on pictures resized on 20240911

#D2: 1 thru 9  done on 20240910, on pictures resized on 20240909
#    10        done on 20240912, on pictures resized on 20240912
#REDONE FOR RGB NON-NORMALIZATION TO CALCULATE NON-NORMALIZED RGB YELLOWNESS
#D2: 1 thru 9  redone on 20240919, on pictures resized on 20240909
#    10        redone on 20240919, on pictures resized on 20240912

#D3: 1 thru 9  done on 20240910, on pictures resized on 20240909
#    10        done on 20240914, on pictures resized on 20240914

# my_batch = 1
# my_pl = [1,2,3,4]

# my_batch = 2
# my_pl = [1,2,3,4,5]

# my_batch = 3
# my_pl = [1,2,3,4,5]

# my_batch = 4
# my_pl = [1]

# my_batch = 5
# my_pl = [1, 2]

# my_batch = 6
# my_pl = [1,2,3,4,5,6]

# my_batch = 7
# my_pl = [1,2,3,4]

# my_batch = 8
# my_pl = [1,2,3,4,5,6]

# my_batch = 9
# my_pl = [1]

# my_batch = 10
# my_pl = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

###############################################################################
###############################################################################
# Test that only one of the days is being analyzed
ut.check_single_true(D1, D2, D3) 

if D1:
    my_day = 1
    day_path = '3 D1 quantification/'
elif D2:
    my_day = 2
    day_path = '4 D2 quantification/'
elif D3:
    my_day = 3
    day_path = '5 D3 quantification/'
else:
    raise ValueError('No day chosen!')

my_conds = ['C', 'H']
folders = []
output_path_txt = []
output_path_figs = base_path + f"/Results/B{my_batch}/"
        
for my_condition in my_conds:

    # Folders containing the images
    folders.append(base_path + day_path + f'/B{my_batch}/B{my_batch}{my_condition}/')
    output_path_txt.append(base_path + day_path + f'Results/B{my_batch}/B{my_batch}{my_condition}D{my_day}_analysis.txt')

# Check if all images have the same size
image_size = ut.check_image_sizes(folders, my_batch)

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

total_curvature           = [[], []]
curvature_std             = [[], []]
bounding_box_aspect_ratio = [[], []]
frechet_distance          = [[], []]
AdivP                     = [[], []]
convexity                 = [[], []]

mean_curvature       = [[], []] 
max_curvature        = [[], []]
skewness_curvature   = [[], []]
kurtosis_curvature   = [[], []]
rms_curvature        = [[], []]
normalized_curvature = [[], []]
radius_of_curvature  = [[], []]

my_images = [[], []]

# Process images from both folders
for folder_idx, folder in enumerate(folders):
   for idx, filename in enumerate(sorted([f for f in os.listdir(folder) if f.endswith('.png')])):
       if filename.endswith(".png"): 
           image_path = os.path.join(folder, filename)
           filenames[folder_idx].append(filename)
           results = ut.process_image_NEW(image_path)
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
           total_curvature[folder_idx].append(results[11])
           curvature_std[folder_idx].append(results[12])
           bounding_box_aspect_ratio[folder_idx].append(results[13])
           frechet_distance[folder_idx].append(results[14])
           my_images[folder_idx].append(results[15])
           AdivP[folder_idx].append(results[16])
           convexity[folder_idx].append(results[17])
           mean_curvature[folder_idx].append(results[18])
           max_curvature[folder_idx].append(results[19])
           skewness_curvature[folder_idx].append(results[20])
           kurtosis_curvature[folder_idx].append(results[21])
           rms_curvature[folder_idx].append(results[22])
           normalized_curvature[folder_idx].append(results[23])
           radius_of_curvature[folder_idx].append(results[24])

# Calculate Yellowness for whole image
rgb_yellowness, yellowness_index_cie, lab_b_yellowness, hsv_yellowness = ut.quantify_yellowness(my_images)

# Calculate Yellowness for binarized image
### Find best global threshold to binarize the images
global_threshold = ut.calculate_global_threshold(my_images[0] + my_images[1])
print(f"Chosen Global Threshold: {global_threshold}")
# Binarize the control and hypo condition images
binarized_control_images = [ut.binarize_imageNEW2(img, global_threshold) for img in my_images[0]]
binarized_hypo_images    = [ut.binarize_imageNEW2(img, global_threshold) for img in my_images[1]]
# Combine them into the format needed for plotting
binarized_images = [binarized_control_images, binarized_hypo_images]
plot_title = f'Day {my_day} $\cdot$ Batch {my_batch} $\cdot$ Control vs. Hypo'
footer_txt = 'not sorted $\cdot$ all images same magnification'
fig = plt.figure()
ut.plot_D2_vertical_binary(fig, len(binarized_images[0]), len(binarized_images[1]), binarized_images, plot_title + ' $\cdot$ binary images', footer_txt, ['black', 'red'], BIGGER_SIZE, MEDIUM_SIZE, font_title, font_text)
rgb_B  = [[], []]
cie_B  = [[], []]
labb_B = [[], []]
hsv_B  = [[], []]
# Compute yellowness for control images (index 0 for controls)
rgb_B[0], cie_B[0], labb_B[0], hsv_B[0] = ut.quantify_yellowness_masked(binarized_control_images, my_images[0])
# Compute yellowness for hypo images (index 1 for hypos)
rgb_B[1], cie_B[1], labb_B[1], hsv_B[1] = ut.quantify_yellowness_masked(binarized_hypo_images,    my_images[1])

black_percentages_control, contour_areas = ut.process_images_with_black_percentage(binarized_control_images, my_images[0])
black_percentages_hypo, contour_areas = ut.process_images_with_black_percentage(binarized_hypo_images, my_images[1])
pigmentation =  [black_percentages_control, black_percentages_hypo]


#Save the results to text files for each folder
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
       
        f.write("Curvatures:\n")
        f.write(str(total_curvature[folder_idx]) + "\n")
        f.write("Curvature Stds:\n")
        f.write(str(curvature_std[folder_idx]) + "\n")
        f.write("BBARs:\n")
        f.write(str(bounding_box_aspect_ratio[folder_idx]) + "\n")
        f.write("Frechets:\n")
        f.write(str(frechet_distance[folder_idx]) + "\n")
       
        f.write("RGB Y:\n")
        f.write(str(rgb_yellowness[folder_idx]) + "\n")
        f.write("CIE Y:\n")
        f.write(str(yellowness_index_cie[folder_idx]) + "\n")
        f.write("lab b Y:\n")
        f.write(str(lab_b_yellowness[folder_idx]) + "\n")
        f.write("HSV Y:\n")
        f.write(str(hsv_yellowness[folder_idx]) + "\n")
       
        f.write("RGB Y binary:\n")
        f.write(str(rgb_B[folder_idx]) + "\n")
        f.write("CIE Y binary:\n")
        f.write(str(cie_B[folder_idx]) + "\n")
        f.write("lab b Y binary:\n")
        f.write(str(labb_B[folder_idx]) + "\n")
        f.write("HSV Y binary:\n")
        f.write(str(hsv_B[folder_idx]) + "\n")
       
        f.write("area/perimeter:\n")
        f.write(str(AdivP[folder_idx]) + "\n")
       
        f.write("Convexities:\n")
        f.write(str(convexity[folder_idx]) + "\n")
       
        f.write("Mean curvatures:\n")
        f.write(str(mean_curvature[folder_idx]) + "\n")
        f.write("Max curvatures:\n")
        f.write(str(max_curvature[folder_idx]) + "\n")
        f.write("Skewness curvatures:\n")
        f.write(str(skewness_curvature[folder_idx]) + "\n")
        f.write("Kurtosis curvatures:\n")
        f.write(str(kurtosis_curvature[folder_idx]) + "\n")
        f.write("RMS curvatures:\n")
        f.write(str(rms_curvature[folder_idx]) + "\n")
        f.write("Norm curvatures:\n")
        f.write(str(normalized_curvature[folder_idx]) + "\n")
        f.write("Radius of curvatures:\n")
        f.write(str(radius_of_curvature[folder_idx]) + "\n")
       
        f.write("Pigmentations:\n")
        f.write(str(pigmentation[folder_idx]) + "\n")
       
        # f.write("Images with overlay:\n")
        # f.write(str(images_with_overlay[folder_idx]) + "\n")
        # f.write("Hull images:\n")
        # f.write(str(hull_images[folder_idx]) + "\n")
       
       
if D1:
    rotate_indices_C_sol = []#[70] # indices of images to rotate in Control, solidity plot
    rotate_indices_H_sol = []#[9,
                            # 10,11,12,13,17,18,
                            # 21,23,
                            # 31,32,33,35,37,38,
                            # 40,41,43,44,45,48,
                            # 50,54,55,56,58,
                            # 60,61,63,66,67,69,
                            # 74,78,79] # indices of images to rotate in Hypo, solidity plot
    rotate_indices_C_elo = []#[4,5,9,
                            # 10,13,
                            # 21,22,25,26,29,
                            # 30,32,33,38,39,
                            # 41,43,44,47,48,49,
                            # 52,53,54,55,56,58,
                            # 66,68,69] # indices of images to rotate in Control, elongation plot
    rotate_indices_H_elo = []#[1,5,6,7,
                            # 10,11,13,14,16,17,18,19,
                            # 20,24,25,27,28,29,
                            # 34,35,38,39,
                            # 40,45,46,
                            # 51,52,53,54,56,
                            # 60,61,63,68,
                            # 71,72,77,78]        # indices of images to rotate in Hypo, elongation plot
    
    rotate_indices_C_ecc = [] #[5,9,
                            # 10,
                            # 26,
                            # 30,38,
                            # 43,47,48,
                            # 52,53,54,55,57,58,59,
                            # 66,68,69]
    rotate_indices_H_ecc = []#[1,5,6,7,
                            # 10,11,13,14,16,17,18,19,
                            # 20,24,25,27,29,
                            # 33,34,35,38,
                            # 40,45,46,49,
                            # 52,53,54,56,
                            # 60,61,63,68,
                            # 71,72,77,78]
    
    # Print the number of images processed
    print(f"Number of images in Control: {len(images_with_overlay[0])}")
    print(f"Number of images in Hypo: {len(images_with_overlay[1])}")

    #plot_title = f'd) Batch {my_batch} $\cdot$ Control vs. Hypo $\cdot$ Day 1'
    

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
            Assessment_path = os.path.join(base_path,f"2 Experimental overview/Assessments/B{my_batch}/B{my_batch}{'C' if idx == 0 else 'H'}P{my_plate}_assessment.txt")

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
        
    # Sorting using ecc
    plot_title = 'd) Day 1 $\cdot$ eccentricity $\cdot$ control vs. hypomagnetic'
    footer_txt_ecc = 'sorted by increasing eccentricity $\cdot$ all images same magnification'
    # Data containers
    filenames_ecc           = [[], []]
    solidities_ecc          = [[], []]
    images_with_overlay_ecc = [[], []]
    hull_images_ecc         = [[], []]
    framecolors_ecc         = [[], []]
    
    for folder_idx in range(2):
        sorted_indices_ecc = np.argsort(eccentricities[folder_idx])
        filenames_ecc[folder_idx] = [filenames[folder_idx][i] for i in sorted_indices_ecc]
        solidities_ecc[folder_idx] = [solidities[folder_idx][i] for i in sorted_indices_ecc]
        images_with_overlay_ecc[folder_idx] = [images_with_overlay[folder_idx][i] for i in sorted_indices_ecc]
        hull_images_ecc[folder_idx] = [hull_images[folder_idx][i] for i in sorted_indices_ecc]
        framecolors_ecc[folder_idx] = [framecolors[folder_idx][i] for i in sorted_indices_ecc]
        
    # Rotate images if needed

    for idx in rotate_indices_C_ecc:
        images_with_overlay_ecc[0][idx] = scipy.ndimage.rotate(images_with_overlay_ecc[0][idx], 180)

    for idx in rotate_indices_H_ecc:
        images_with_overlay_ecc[1][idx] = scipy.ndimage.rotate(images_with_overlay_ecc[1][idx], 180)
        
    fig_ecc = plt.figure()
    ut.plot_D1(fig_ecc, num_rows, num_cols, num_images1, num_images2, images_with_overlay_ecc, plot_title, footer_txt_ecc, framecolors_ecc, BIGGER_SIZE, MEDIUM_SIZE, font_title, font_text, rotate_indices_C_ecc, rotate_indices_H_ecc)
    
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
        framecolors_sol[folder_idx] = [framecolors[folder_idx][i] for i in sorted_indices_sol]
        
    # Rotate images if needed
    for idx in rotate_indices_C_sol:
        images_with_overlay_sol[0][idx] = scipy.ndimage.rotate(images_with_overlay_sol[0][idx], 180)
        hull_images_sol[0][idx] = scipy.ndimage.rotate(hull_images_sol[0][idx], 180)

    for idx in rotate_indices_H_sol:
        images_with_overlay_sol[1][idx] = scipy.ndimage.rotate(images_with_overlay_sol[1][idx], 180)
        hull_images_sol[1][idx] = scipy.ndimage.rotate(hull_images_sol[1][idx], 180)
        
        
    #fig_sol = plt.figure(figsize=(fig_width, fig_length))
    fig_sol = plt.figure()
    plot_title = 'a) Day 1 $\cdot$ solidity $\cdot$ control vs. hypomagnetic'
    ut.plot_D1(fig_sol, num_rows, num_cols, num_images1, num_images2, images_with_overlay_sol, plot_title, footer_txt_sol, framecolors_sol, BIGGER_SIZE, MEDIUM_SIZE, font_title, font_text, rotate_indices_C_sol, rotate_indices_H_sol)

   # fig_sol_hull = plt.figure(figsize=(fig_width, fig_length))
    fig_sol_hull = plt.figure()
    plot_title = 'b) Day 1 $\cdot$ solidity (hull) $\cdot$ control vs. hypomagnetic'
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
        
    #fig_elo = plt.figure(figsize=(fig_width, fig_length))
    plot_title = 'b) Day 1 $\cdot$ elongation $\cdot$ control vs. hypomagnetic'
    fig_elo = plt.figure()
    ut.plot_D1(fig_elo, num_rows, num_cols, num_images1, num_images2, images_with_overlay_elo, plot_title, footer_txt_elo, framecolors_elo, BIGGER_SIZE, MEDIUM_SIZE, font_title, font_text, rotate_indices_C_elo, rotate_indices_H_elo)

    ###############################################################################

    # Save all open figures
    if save:
        FigName = [f'B{my_batch}EccentricitySorted',
                   f'B{my_batch}SoliditySorted', 
                   f'B{my_batch}HullImages', 
                   f'B{my_batch}ElongationSorted']
        for i, fig in enumerate(map(plt.figure, plt.get_fignums())):
            fig.savefig(output_path_figs + f'{FigName[i]}.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
            fig.savefig(output_path_figs + f'{FigName[i]}.png', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PNG

    plt.show()
    klklklklklk
    
else:
    
    # Print the number of images processed
    print(f"Number of images in Control: {len(images_with_overlay[0])}")
    print(f"Number of images in Hypo: {len(images_with_overlay[1])}")
    
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
            Assessment_path = os.path.join(base_path,f"2 Experimental overview/Assessments/B{my_batch}/B{my_batch}{'C' if idx == 0 else 'H'}P{my_plate}_assessment.txt")
    
            # Read the file and extract the necessary lists
            c_content = ut.read_file(Assessment_path)
            c_filenames = ut.extract_list(c_content, 'Filenames:')
            
            # Modify each filename for comparison: include 'D{my_day}' and add '.png'
            modified_filenames.extend([f"{filename[:3]}D{my_day}{filename[3:]}.png" for filename in c_filenames])
            status_d3.extend(ut.extract_list(c_content, 'StatusD3:'))
    
        # Find common filenames
        common_filenames = set(modified_filenames).intersection(set(filenames[idx]))
        # Filter status_d3 based on common filenames
        filtered_status_d3 = [status_d3[modified_filenames.index(fname)] for fname in common_filenames]
        # Map status to colors and save to framecolors
        framecolors[idx] = ut.map_status_to_color(filtered_status_d3)
        #print(f"framecolors[{folder_idx}] = {framecolors[folder_idx]}")
        
    # ###############################################################################
    # ###############################################################################
    # # Sorting by yellow color (increasing order)
    # # Data containers for sorted results
    # filenames_y = [[], []]
    # y_sorted = [[], []]
    # images_with_overlay_y = [[], []]
    # hull_images_y = [[], []]
    # framecolors_y = [[], []]
    # images_without_overlay_y = [[], []]
    # sorted_bin_images = [[],[]]
    
    # my_bin_images = [binarized_control_images, binarized_hypo_images]
    # # Loop through both folders (Control and Hypo)
    # for folder_idx in range(2):
    #       filenames_y[folder_idx], y_sorted[folder_idx], images_with_overlay_y[folder_idx], \
    #       hull_images_y[folder_idx], framecolors_y[folder_idx], images_without_overlay_y[folder_idx], sorted_bin_images[folder_idx]  = ut.sort_by_property(
    #           filenames[folder_idx], images_with_overlay[folder_idx], hull_images[folder_idx], framecolors[folder_idx], labb_B[folder_idx], my_images[folder_idx], my_bin_images[folder_idx], increasing=True
    #       ) 
     
    # plot_title = 'Day 2 $\cdot$ l*a*b* yellowness $\cdot$ control vs. hypomagnetic'
    # footer_txt_y = 'sorted by increasing l*a*b* yellowness $\cdot$ all images same magnification'
    
    # fig5_color = plt.figure()
    # ut.plot_D2_vertical_new(fig5_color, len(images_without_overlay_y[0]), len(images_without_overlay_y[1]), images_without_overlay_y, sorted_bin_images, plot_title, footer_txt_y, BIGGER_SIZE, MEDIUM_SIZE, font_title, font_text)
    # fig5_color.savefig(output_path_figs + 'ColorFairest.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)
    
    # klklklk
    # fig7_color = plt.figure()
    # ut.plot_D2_vertical(fig7_color, num_images1, num_images2, images_without_overlay_y, plot_title, footer_txt_y, framecolors_y, BIGGER_SIZE, MEDIUM_SIZE, font_title, font_text)
    # fig7_color.savefig(output_path_figs + 'B4ColorFairest2.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)
    
    
    # #only healthy
    # filtered_control_images = [img for img, color in zip(images_without_overlay_y[0], framecolors_y[0]) if color == "green"]
    # filtered_hypo_images    = [img for img, color in zip(images_without_overlay_y[1], framecolors_y[1]) if color == "green"]
    
    # #filt_bin_C = [img for img, color in zip(sorted_bin_images[0], framecolors_y[0]) if color == "green"]
    # # filt_bin_H = [img for img, color in zip(sorted_bin_images[1], framecolors_y[1]) if color == "green"]
    
    # num = 5;
    # middle_index = len(filtered_control_images) // 2
    # start_index = middle_index - int(np.floor(num//2)) # To include the middle element and half num elements on either side
    # centermost_elements_C = filtered_control_images[start_index:start_index + num]
    # # center_bin_C = filt_bin_C[start_index:start_index + 5]
    
    # middle_index2 = len(filtered_hypo_images) // 2
    # start_index2 = middle_index2 - int(np.floor(num//2))  # To include the middle element and half num elements on either side
    # centermost_elements_H = filtered_hypo_images[start_index2:start_index2 + num]
    # # center_bin_H = filt_bin_H[start_index2:start_index2 + 5]
    
    # new_array4 = [[],[]]
    # new_array4[0] = filtered_control_images[:num] + centermost_elements_C + filtered_control_images[-num:] 
    # new_array4[1] = filtered_hypo_images[:num]    + centermost_elements_H + filtered_hypo_images[-num:] 
    
    # # bin_images_arr = [[],[]]
    # # bin_images_arr[0] = filt_bin_C[:num] + center_bin_C + filt_bin_C[-num:] 
    # # bin_images_arr[1] = filt_bin_H[:num] + center_bin_H + filt_bin_H[-num:] 
    
    # # fig5_color = plt.figure()
    # # ut.plot_D2_vertical_new(fig5_color, 3*num, 3*num, new_array4, bin_images_arr, plot_title, footer_txt_y, BIGGER_SIZE, MEDIUM_SIZE, font_title, font_text)
    # # fig5_color.savefig(output_path_figs + 'ColorFairest.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)
    
    # fig6_color = plt.figure()
    # ut.plot_D2_vertical_nocolor(fig6_color, 3*num, 3*num, new_array4, plot_title, footer_txt_y, BIGGER_SIZE, MEDIUM_SIZE, font_title, font_text)
    # fig6_color.savefig(output_path_figs + '3ColorFairest2.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)
    
    # fig_color = plt.figure()
    # ut.plot_D2_vertical(fig_color, num_images1, num_images2, images_with_overlay_y, plot_title, footer_txt_y, framecolors_y, BIGGER_SIZE, MEDIUM_SIZE, font_title, font_text)
    
    # num = 10;
    # new_array = [[],[]]
    # new_array[0] = images_with_overlay_y[0][-num:] 
    # new_array[1] = images_with_overlay_y[1][:num]
    # new_framecolors_y = [
    # framecolors_y[0][-num:],  # Last num elements of the first nested array
    # framecolors_y[1][:num]    # Firstnum elements of the second nested array
    # ]
    # fig2_color = plt.figure()
    # ut.plot_D2_vertical(fig2_color, num, num, new_array, plot_title, footer_txt_y, new_framecolors_y, BIGGER_SIZE, MEDIUM_SIZE, font_title, font_text)
    
    # fig2_color.savefig(output_path_figs + 'Color.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)
    
    # num = 5;
    # new_array2 = [[],[]]
    # new_array2[0] = images_without_overlay_y[0][:num] + images_without_overlay_y[0][-num:] 
    # new_array2[1] = images_without_overlay_y[1][:num] + images_without_overlay_y[1][-num:] 
    # new_framecolors_y2 = [
    # framecolors_y[0][:num] + framecolors_y[0][-num:], 
    # framecolors_y[1][:num] + framecolors_y[1][-num:]
    # ]
    
    # fig3_color = plt.figure()
    # ut.plot_D2_vertical(fig3_color, 2*num, 2*num, new_array2, plot_title, footer_txt_y, new_framecolors_y2, BIGGER_SIZE, MEDIUM_SIZE, font_title, font_text)
    # fig3_color.savefig(output_path_figs + 'ColorFair.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)
    
    
    
    
    # num = 5;
    # middle_index = len(images_without_overlay_y[0]) // 2
    # start_index = middle_index - 2  # To include the middle element and 2 elements on either side
    # centermost_elements_C = images_without_overlay_y[0][start_index:start_index + 5]
    
    # middle_index2 = len(images_without_overlay_y[1]) // 2
    # start_index2 = middle_index2 - 2  # To include the middle element and 2 elements on either side
    # centermost_elements_H = images_without_overlay_y[1][start_index2:start_index2 + 5]
    
    # new_array3 = [[],[]]
    # new_array3[0] = images_without_overlay_y[0][:num] + centermost_elements_C + images_without_overlay_y[0][-num:] 
    # new_array3[1] = images_without_overlay_y[1][:num] + centermost_elements_H + images_without_overlay_y[1][-num:] 
    # new_framecolors_y3 = [
    # framecolors_y[0][:num] + framecolors_y[0][start_index:start_index + 5]    + framecolors_y[0][-num:], 
    # framecolors_y[1][:num] + framecolors_y[1][start_index2:start_index2 + 5]  + framecolors_y[1][-num:]
    # ]
    
    # fig4_color = plt.figure()
    # ut.plot_D2_vertical(fig4_color, 3*num, 3*num, new_array3, plot_title, footer_txt_y, new_framecolors_y3, BIGGER_SIZE, MEDIUM_SIZE, font_title, font_text)
    # fig4_color.savefig(output_path_figs + 'ColorFair.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)
    

    ###############################################################################
    ###############################################################################
    # Sorting by major axes(increasing order)
    # Data containers for sorted results
    filenames_major_axes = [[], []]
    major_axes_sorted = [[], []]
    images_with_overlay_major_axes = [[], []]
    hull_images_major_axes = [[], []]
    framecolors_major_axes = [[], []]
    images_without_overlay_major_axes = [[], []]
    
    # Loop through both folders (Control and Hypo)
    for folder_idx in range(2):
        filenames_major_axes[folder_idx], major_axes_sorted[folder_idx], images_with_overlay_major_axes[folder_idx], \
        hull_images_major_axes[folder_idx], framecolors_major_axes[folder_idx], images_without_overlay_major_axes[folder_idx]  = ut.sort_by_property(
            filenames[folder_idx], images_with_overlay[folder_idx], hull_images[folder_idx], framecolors[folder_idx], major_axes[folder_idx], my_images[folder_idx], increasing=True
        )
    
    
    plot_title = 'Day 3 $\cdot$ major axis $\cdot$ control vs. hypomagnetic'
    footer_txt_major = 'sorted by increasing major axis $\cdot$ all images same magnification'
    fig_major = plt.figure()
    
    images_to_plot = [[], []]
    images_to_plot[0] = [result[4] for result in ut.process_images_to_plot(images_without_overlay_major_axes[0])]
    images_to_plot[1] = [result[4] for result in ut.process_images_to_plot(images_without_overlay_major_axes[1])]

    
    #images_to_plot = [[],[]]
    #images_to_plot[0] = ut.process_images_to_plot(images_without_overlay_major_axes[0])
    #images_to_plot[1] = ut.process_images_to_plot(images_without_overlay_major_axes[1])
    
    
    ut.plot_D3_vertical_old(fig_major, num_images1, num_images2, images_to_plot, plot_title, footer_txt_major, framecolors_major_axes, BIGGER_SIZE, MEDIUM_SIZE, font_title, font_text)
    fig_major.savefig(output_path_figs + 'BMajorAll.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)
    
    klklklk
    #only healthy
    filtered_control_images = [img for img, color in zip(images_without_overlay_major_axes[0], framecolors_major_axes[0]) if color == "green"]
    filtered_hypo_images    = [img for img, color in zip(images_without_overlay_major_axes[1], framecolors_major_axes[1]) if color == "green"]
     
    num = 5;
    middle_index = len(filtered_control_images) // 2
    start_index = middle_index - int(np.floor(num//2)) # To include the middle element and half num elements on either side
    centermost_elements_C = filtered_control_images[start_index:start_index + num]
    
    middle_index2 = len(filtered_hypo_images) // 2
    start_index2 = middle_index2 - int(np.floor(num//2))  # To include the middle element and half num elements on either side
    centermost_elements_H = filtered_hypo_images[start_index2:start_index2 + num]
    
    new_array4 = [[],[]]
    new_array4[0] = filtered_control_images[:num] + centermost_elements_C + filtered_control_images[-num:] 
    new_array4[1] = filtered_hypo_images[:num]    + centermost_elements_H + filtered_hypo_images[-num:] 
    
    
    fig5_color = plt.figure()
    ut.plot_D2_vertical_nocolor(fig5_color, 3*num, 3*num, new_array4, plot_title, footer_txt_major, BIGGER_SIZE, MEDIUM_SIZE, font_title, font_text)
    fig5_color.savefig(output_path_figs + 'Major.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)
    
    klklklklklk
    ###############################################################################
    # Sorting by elongations (increasing order)
    # Data containers for sorted results
    filenames_elongation = [[], []]
    elongations_sorted = [[], []]
    images_with_overlay_elongation = [[], []]
    hull_images_elongation = [[], []]
    framecolors_elongation = [[], []]
    
    # Loop through both folders (Control and Hypo)
    for folder_idx in range(2):
        filenames_elongation[folder_idx], elongations_sorted[folder_idx], images_with_overlay_elongation[folder_idx], \
        hull_images_elongation[folder_idx], framecolors_elongation[folder_idx] = ut.sort_by_property(
            filenames[folder_idx], images_with_overlay[folder_idx], hull_images[folder_idx], framecolors[folder_idx], elongations[folder_idx], images_without_overlay=None, increasing=True
        )
    
    footer_txt_elo = 'sorted by increasing elongation $\cdot$ all images same magnification'
    fig_elo = plt.figure()
    ut.plot_D2_vertical(fig_elo, num_images1, num_images2, images_with_overlay_elongation, plot_title, footer_txt_elo, framecolors_elongation, BIGGER_SIZE, MEDIUM_SIZE, font_title, font_text)
    
    ###############################################################################
    # Sorting by eccentricities (increasing order)
    filenames_ecc = [[], []]
    eccentricities_sorted = [[], []]
    images_with_overlay_ecc = [[], []]
    hull_images_ecc = [[], []]
    framecolors_ecc = [[], []]
    
    for folder_idx in range(2):
        filenames_ecc[folder_idx], eccentricities_sorted[folder_idx], images_with_overlay_ecc[folder_idx], \
        hull_images_ecc[folder_idx], framecolors_ecc[folder_idx] = ut.sort_by_property(
            filenames[folder_idx], images_with_overlay[folder_idx], hull_images[folder_idx], framecolors[folder_idx], eccentricities[folder_idx], images_without_overlay=None, increasing=True)
    
    footer_txt_ecc = 'sorted by increasing eccentricity $\cdot$ all images same magnification'
    fig_ecc = plt.figure()
    ut.plot_D2_vertical(fig_ecc, num_images1, num_images2, images_with_overlay_ecc, plot_title, footer_txt_ecc, framecolors_ecc, BIGGER_SIZE, MEDIUM_SIZE, font_title, font_text)
    
    
        
    ###############################################################################
    # Sorting by solidities (decreasing order)
    filenames_sol = [[], []]
    solidities_sol = [[], []]
    images_with_overlay_sol = [[], []]
    hull_images_sol = [[], []]
    framecolors_sol = [[], []]
    
    for folder_idx in range(2):
        filenames_sol[folder_idx], solidities_sol[folder_idx], images_with_overlay_sol[folder_idx], \
        hull_images_sol[folder_idx], framecolors_sol[folder_idx] = ut.sort_by_property(
            filenames[folder_idx], images_with_overlay[folder_idx], hull_images[folder_idx], framecolors[folder_idx], solidities[folder_idx], images_without_overlay=None, increasing=False)
        
    footer_txt_sol = 'sorted by decreasing solidity $\cdot$ all images same magnification'
    fig_sol = plt.figure()
    ut.plot_D2_vertical(fig_sol,  num_images1, num_images2, images_with_overlay_sol, plot_title, footer_txt_sol, framecolors_sol, BIGGER_SIZE, MEDIUM_SIZE, font_title, font_text)
    
    fig_sol_hull = plt.figure()
    ut.plot_D2_vertical(fig_sol_hull, num_images1, num_images2, images_with_overlay_sol, plot_title + 'HULL', footer_txt_sol, framecolors_sol, BIGGER_SIZE, MEDIUM_SIZE, font_title, font_text, hull_images_sol )
    
    ###############################################################################
    # Sorting by curvature (increasing order)
    
    # Data containers for sorted results
    filenames_curvature = [[], []]
    total_curvature_sorted = [[], []]
    images_with_overlay_curvature = [[], []]
    hull_images_curvature = [[], []]
    framecolors_curvature = [[], []]
    
    # Loop through both folders (Control and Hypo)
    for folder_idx in range(2):
        filenames_curvature[folder_idx], total_curvature_sorted[folder_idx], images_with_overlay_curvature[folder_idx], \
        hull_images_curvature[folder_idx], framecolors_curvature[folder_idx] = ut.sort_by_property(
            filenames[folder_idx], images_with_overlay[folder_idx], hull_images[folder_idx], framecolors[folder_idx], total_curvature[folder_idx], images_without_overlay=None, increasing=True
        )
    
    footer_txt_curvature = 'sorted by increasing curvature $\cdot$ all images same magnification'
    fig_curvature = plt.figure()
    ut.plot_D2_vertical(fig_curvature, num_images1, num_images2, images_with_overlay_curvature, plot_title, footer_txt_curvature, framecolors_curvature, BIGGER_SIZE, MEDIUM_SIZE, font_title, font_text)
    
    ###############################################################################
    # Sorting by curvature std
    # Data containers for sorted results
    filenames_curvature_std = [[], []]
    curvature_std_sorted = [[], []]
    images_with_overlay_curvature_std = [[], []]
    hull_images_curvature_std = [[], []]
    framecolors_curvature_std = [[], []]
    
    # Loop through both folders (Control and Hypo)
    for folder_idx in range(2):
        filenames_curvature_std[folder_idx], curvature_std_sorted[folder_idx], images_with_overlay_curvature_std[folder_idx], \
        hull_images_curvature_std[folder_idx], framecolors_curvature_std[folder_idx] = ut.sort_by_property(
            filenames[folder_idx], images_with_overlay[folder_idx], hull_images[folder_idx], framecolors[folder_idx], curvature_std[folder_idx], images_without_overlay=None,  increasing=True
        )
    
    footer_txt_curvature_std = 'sorted by increasing curvature std $\cdot$ all images same magnification'
    fig_curvature_std = plt.figure()
    ut.plot_D2_vertical(fig_curvature_std, num_images1, num_images2, images_with_overlay_curvature_std, plot_title, footer_txt_curvature_std, framecolors_curvature_std, BIGGER_SIZE, MEDIUM_SIZE, font_title, font_text)
    
    ###############################################################################
    # Sorting by BBAR
    # Data containers for sorted results
    filenames_bounding_box_aspect_ratio = [[], []]
    bounding_box_aspect_ratio_sorted = [[], []]
    images_with_overlay_bounding_box_aspect_ratio = [[], []]
    hull_images_bounding_box_aspect_ratio = [[], []]
    framecolors_bounding_box_aspect_ratio = [[], []]
    
    # Loop through both folders (Control and Hypo)
    for folder_idx in range(2):
        filenames_bounding_box_aspect_ratio[folder_idx], bounding_box_aspect_ratio_sorted[folder_idx], images_with_overlay_bounding_box_aspect_ratio[folder_idx], \
        hull_images_bounding_box_aspect_ratio[folder_idx], framecolors_bounding_box_aspect_ratio[folder_idx] = ut.sort_by_property(
            filenames[folder_idx], images_with_overlay[folder_idx], hull_images[folder_idx], framecolors[folder_idx], bounding_box_aspect_ratio[folder_idx], images_without_overlay=None, increasing=True
        )
    
    footer_txt_bounding_box_aspect_ratio = 'sorted by increasing bounding box aspect ratio $\cdot$ all images same magnification'
    fig_bounding_box_aspect_ratio = plt.figure()
    ut.plot_D2_vertical(fig_bounding_box_aspect_ratio, num_images1, num_images2, images_with_overlay_bounding_box_aspect_ratio, plot_title, footer_txt_bounding_box_aspect_ratio, framecolors_bounding_box_aspect_ratio, BIGGER_SIZE, MEDIUM_SIZE, font_title, font_text)
    
    ###############################################################################
    # Sorting by Frechet distance
    # Data containers for sorted results
    filenames_frechet_distance = [[], []]
    frechet_distance_sorted = [[], []]
    images_with_overlay_frechet_distance = [[], []]
    hull_images_frechet_distance = [[], []]
    framecolors_frechet_distance = [[], []]
    
    # Loop through both folders (Control and Hypo)
    for folder_idx in range(2):
        filenames_frechet_distance[folder_idx], frechet_distance_sorted[folder_idx], images_with_overlay_frechet_distance[folder_idx], \
        hull_images_frechet_distance[folder_idx], framecolors_frechet_distance[folder_idx] = ut.sort_by_property(
            filenames[folder_idx], images_with_overlay[folder_idx], hull_images[folder_idx], framecolors[folder_idx], frechet_distance[folder_idx], images_without_overlay=None, increasing=True
        )
    
    footer_txt_frechet_distance = 'sorted by increasing frechet distance $\cdot$ all images same magnification'
    fig_frechet_distance = plt.figure()
    ut.plot_D2_vertical(fig_frechet_distance, num_images1, num_images2, images_with_overlay_frechet_distance, plot_title, footer_txt_frechet_distance, framecolors_frechet_distance, BIGGER_SIZE, MEDIUM_SIZE, font_title, font_text)
    
    ###############################################################################
    
    # Save all open figures
    if save:
        FigName = [f'D{my_day}_binary',
                   'D{my_day}_yellow_sorted',
                   'D{my_day}_length_sorted',
                   'D{my_day}_elongation_sorted',
                   'D{my_day}_eccentricity_sorted',
                   'D{my_day}_solidity_sorted', 
                   'D{my_day}_solidity_sorted_hull', 
                   'D{my_day}_curvature_sorted', 
                   'D{my_day}_scurvature_std_sorted' , 
                   'D{my_day}_BBAR_sorted' , 
                   'D{my_day}_Frechet_sorted'] 
        for i, fig in enumerate(map(plt.figure, plt.get_fignums())):
            fig.savefig(output_path_figs + f'{FigName[i]}.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
            #fig.savefig(output_path + f'{FigName[i]}.png', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PNG
    
    plt.show()
