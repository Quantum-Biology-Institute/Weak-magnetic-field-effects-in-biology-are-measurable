###############################################################################
# BASE PATH
base_path = '/Users/clarice/Desktop/'
###############################################################################

import sys
sys.path.append(base_path + '1 Utilities/')
import utilities as ut

import plot_config
# Initialize fonts and settings
font_title, font_text, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = plot_config.setup()

###############################################################################
# CHANGE HERE ONLY
###############################################################################

# 1-10 resized on 20240911

# my_bat = [1] #(426, 426)
# my_bat = [2] #(531, 531)
# my_bat = [3] #(695, 695)
# my_bat = [4] #(319, 319)
# my_bat = [5] #(610, 610)
# my_bat = [6] #(322, 322)
# my_bat = [7] #(599, 599)
# my_bat = [8] #(413, 413)
# my_bat = [9] #(492, 492)
#my_bat = [10]#(539, 539)

###############################################################################
###############################################################################
my_conds = ['C', 'H']
folders = []

for my_batch in my_bat:
        
    for my_condition in my_conds:

        # Folders containing the images
        folders.append(base_path + f'3 D1 quantification/B{my_batch}/B{my_batch}{my_condition}/')

# Find the maximum dimension among all images
max_dimension = ut.find_max_dimension(folders)

# Resize all images to the smallest possible square size
ut.resize_images_to_square(folders, max_dimension)

# Check if all images have the same size
ut.check_image_sizes(folders, my_batch)