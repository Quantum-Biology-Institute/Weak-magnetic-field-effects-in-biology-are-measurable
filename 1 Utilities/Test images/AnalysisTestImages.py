###############################################################################
# BASE PATH
base_path = '/Users/clarice/Desktop/'
###############################################################################

import os
import matplotlib.pyplot as plt
import gc
# Clear all figures
plt.close('all')
# Clear all variables (garbage collection)
gc.collect()

import numpy as np

import sys
sys.path.append(base_path + '1 Utilities/')
import utilities as ut

import plot_config
# Initialize fonts and settings
font_title, font_text, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = plot_config.setup()

###############################################################################
# CHANGE HERE ONLY
###############################################################################

my_conds = ['18', '19', '20', '21', '22', '23', '24', '25', '32', '33', '35', '37', '38', '39', '40', '41', '42']
folders = []
output_path_txt = []

for my_condition in my_conds:
    # Folders containing the images
    folder = base_path + f'1 Utilities/Test images/{my_condition}/'
    output_file = base_path + f"1 Utilities/Test images/Results/Stage{my_condition}_analysis.txt"
    folders.append(folder)
    output_path_txt.append(output_file)

    # Data containers (Reset for each folder)
    filenames           = []
    areas               = []
    perimeters          = []
    minor_axes          = []
    major_axes          = []
    convex_hull_area    = []
    elongations         = []
    roundnesses         = []
    eccentricities      = []
    solidities          = []
    images_with_overlay = []
    hull_images         = []
    
    total_curvature           = []
    curvature_std             = []
    bounding_box_aspect_ratio = []
    frechet_distance          = []
    AdivP                     = []
    convexity                 = []
    
    mean_curvature       = []
    max_curvature        = []
    skewness_curvature   = []
    kurtosis_curvature   = []
    rms_curvature        = []
    normalized_curvature = []
    radius_of_curvature  = []
    
    my_images = []
    
    # Process images in the current folder
    for filename in sorted([f for f in os.listdir(folder) if f.endswith('.png')]):
        image_path = os.path.join(folder, filename)
        filenames.append(filename)
        results = ut.process_image_NEW(image_path)
        areas.append(results[0])
        perimeters.append(results[1])
        minor_axes.append(results[2])
        major_axes.append(results[3])
        convex_hull_area.append(results[4])
        elongations.append(results[5])
        roundnesses.append(results[6])
        eccentricities.append(results[7])
        solidities.append(results[8])
        images_with_overlay.append(results[9])
        hull_images.append(results[10])
        total_curvature.append(results[11])
        curvature_std.append(results[12])
        bounding_box_aspect_ratio.append(results[13])
        frechet_distance.append(results[14])
        my_images.append(results[15])
        AdivP.append(results[16])
        convexity.append(results[17])
        mean_curvature.append(results[18])
        max_curvature.append(results[19])
        skewness_curvature.append(results[20])
        kurtosis_curvature.append(results[21])
        rms_curvature.append(results[22])
        normalized_curvature.append(results[23])
        radius_of_curvature.append(results[24])
    
    # Save the results to a text file for each folder
    with open(output_file, 'w') as f:
        f.write("Filenames:\n")
        f.write(str(filenames) + "\n")
        f.write("Areas:\n")
        f.write(str(np.mean(areas)) + "\n")
        f.write("Perimeters:\n")
        f.write(str(np.mean(perimeters)) + "\n")
        f.write("Minor axes:\n")
        f.write(str(np.mean(minor_axes)) + "\n")
        f.write("Major axes:\n")
        f.write(str(np.mean(major_axes)) + "\n")
        f.write("Convex hull areas:\n")
        f.write(str(np.mean(convex_hull_area)) + "\n")
        f.write("Elongations:\n")
        f.write(str(np.mean(elongations)) + "\n")
        f.write("Roundnesses:\n")
        f.write(str(np.mean(roundnesses)) + "\n")
        f.write("Eccentricities:\n")
        f.write(str(np.mean(eccentricities)) + "\n")
        f.write("Solidities:\n")
        f.write(str(np.mean(solidities)) + "\n")
        
        f.write("Curvatures:\n")
        f.write(str(np.mean(total_curvature)) + "\n")
        f.write("Curvature Stds:\n")
        f.write(str(np.mean(curvature_std)) + "\n")
        f.write("BBARs:\n")
        f.write(str(np.mean(bounding_box_aspect_ratio)) + "\n")
        f.write("Frechets:\n")
        f.write(str(np.mean(frechet_distance)) + "\n")
        
        f.write("area/perimeter:\n")
        f.write(str(np.mean(AdivP)) + "\n")
        
        f.write("Convexities:\n")
        f.write(str(np.mean(convexity)) + "\n")
        
        f.write("Mean curvatures:\n")
        f.write(str(np.mean(mean_curvature)) + "\n")
        f.write("Max curvatures:\n")
        f.write(str(np.mean(max_curvature)) + "\n")
        f.write("Skewness curvatures:\n")
        f.write(str(np.mean(skewness_curvature)) + "\n")
        f.write("Kurtosis curvatures:\n")
        f.write(str(np.mean(kurtosis_curvature)) + "\n")
        f.write("RMS curvatures:\n")
        f.write(str(np.mean(rms_curvature)) + "\n")
        f.write("Norm curvatures:\n")
        f.write(str(np.mean(normalized_curvature)) + "\n")
        f.write("Radius of curvatures:\n")
        f.write(str(np.mean(radius_of_curvature)) + "\n")

