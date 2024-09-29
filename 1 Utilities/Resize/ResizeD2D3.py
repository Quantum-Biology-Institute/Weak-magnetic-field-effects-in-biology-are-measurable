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
# All D2 images resized
# All D3 images resized

# Resize D2 or D3 images?
D2 = False  #If True, do D2; False, do D3 instead

#D2, 1 thru 9 done on 20240909
#   10        done on 20240912
#my_bat = [1] #(1023, 424)
#my_bat = [2] #(1300, 488)
#my_bat = [3] #(1057, 342)
#my_bat = [4] #(927,  279)
#my_bat = [5] #(932,  314)
#my_bat = [6] #(990,  317)
#my_bat = [7] #(1158, 436)
#my_bat = [8] #(1126, 373)
#my_bat = [9] #(1262, 527)
#my_bat = [10]#(1132, 425)

#D3, 1 thru 9 done on 20240909
#   10        done on 20240914
#my_bat = [1] #(1302, 451)
#my_bat = [2] #(1725, 531)
#my_bat = [3] #(1603, 416)
#my_bat = [4] #(1239, 289)
#my_bat = [5] #(1403, 457)
#my_bat = [6] #(1424, 426)
#my_bat = [7] #(1563, 497)
#my_bat = [8] #(1318, 440)
#my_bat = [9] #(1320, 383)
#my_bat = [10]#(1394, 486)

###############################################################################
###############################################################################
my_conds = ['C', 'H']
folders = []

if D2 == True: #D2 path
    day_path = '4 D2 quantification/'
else: #D3 path
    day_path = '5 D3 quantification/'

for my_batch in my_bat:
        
    for my_condition in my_conds:

        # Folders containing the images
        folders.append(base_path + day_path + f'B{my_batch}/B{my_batch}{my_condition}/')
            
            
ut.process_and_save_D2_images(folders[0], folders[1])

# Check if all images have the same size
ut.check_image_sizes(folders, my_batch)