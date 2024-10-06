###############################################################################
# BASE PATH
base_path = '/Users/clarice/Desktop/'
###############################################################################

import matplotlib.pyplot as plt

import sys
sys.path.append(base_path + '1 Utilities/')
import utilities as ut

import plot_config
# Initialize fonts and settings
font_title, font_text, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = plot_config.setup()

###############################################################################
# CHANGE HERE ONLY
###############################################################################

save = True

# DONE 20241006
# my_bat = [1]
# my_pl = [1,2,3,4]

# DONE 20241006
# my_bat = [2]
# my_pl = [1,2,3,4,5]

# DONE 20241006
# my_bat = [3]
# my_pl = [1,2,3,4,5]

# DONE 20241006
# my_bat = [4]
# my_pl = [1]

# DONE 20241006
# my_bat = [5]
# my_pl = [1,2]

# DONE 20241006
# my_bat = [6]
# my_pl = [1,2,3,4,5,6]

# DONE 20241006
# my_bat = [7]
# my_pl = [1,2,3,4]

# DONE 20241006
# my_bat = [8]
# my_pl = [1,2,3,4,5,6]
# Control 6 is up to 37 only #for i in range(1, 38)
# Hypo 6 is up to 38 only #for i in range(1, 39)
# see plot_48 function part involving variable "endnum"

# DONE 20241006
# my_bat = [9]
# my_pl = [1] 
# Control and Hypo 1 are up to 39 only #for i in range(1, 40)
# see plot_48 function part involving variable "endnum"

# DONE 20241006
my_bat = [10]
my_pl = [1,2,3,4,5,6,7,8,9,10]

###############################################################################
###############################################################################
my_conds = ['C', 'H']
my_conds_dict = {'C': 'control box', 'H': 'hypomagnetic chamber'}
fig_width = 11 
fig_length = 8.5
my_dpi = 600
longerwellplates = [6, 7, 8, 9, 10] #48 well plates

for my_batch in my_bat:
    
    for my_plate in my_pl:
        
        for my_condition in my_conds:

            if my_batch in longerwellplates: #48-well plates
                fig, axes = plt.subplots(nrows=8, ncols=6, figsize=(fig_width,2*fig_length), dpi=my_dpi) 
            else: #24-well plates
                fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(fig_width,fig_length), dpi=my_dpi) 
            
            plt.suptitle(f'Batch B{my_batch} $\cdot$ {my_conds_dict[my_condition]} $\cdot$ plate {my_plate}/{len(my_pl)} $\cdot$ days 1, 2 and 3', fontsize=BIGGER_SIZE, fontproperties=font_title)

            # Replace these paths with the actual paths to your directories
            labels_day = ['1', '2', '3']
            image_paths = [base_path + f'2 Experimental overview/B{my_batch}/B{my_batch}{my_condition}/B{my_batch}{my_condition}D{labels_day[i]}/B{my_batch}{my_condition}D{labels_day[i]}P{my_plate}' for i in range(3)]

            assessment_path = base_path + f'2 Experimental overview/Assessments/B{my_batch}/B{my_batch}{my_condition}P{my_plate}_assessment.txt'

            output_path_pdf = base_path + f'2 Experimental overview/Results/B{my_batch}/B{my_batch}{my_condition}P{my_plate}.pdf' # Change this path as needed
            output_path_png = base_path + f'2 Experimental overview/Results/B{my_batch}/B{my_batch}{my_condition}P{my_plate}.png'

            # Read the file
            file_content = ut.read_file(assessment_path)

            # Extract the lists
            stages = [ut.extract_list(file_content, f'StageD{i+1}:') for i in range(3)]
            
            statusD3 = ut.extract_list(file_content, 'StatusD3:')
            colorlist = ut.map_status_to_color(statusD3)
            conditionlist = ut.extract_list(file_content, 'Condition:')
            
            # do surviving via stages
            # if my_batch in longerwellplates: #48-well plates
            #     surviving = [48 - stages[i].count('D') for i in range(3)]
            # else:
            #     surviving = [24 - stages[i].count('D') for i in range(3)]
            
            # do surviving via statusD3
            if my_batch in longerwellplates: #48-well plates
                surviving = [48 - statusD3.count('D') for i in range(3)]
            else:
                surviving = [24 - statusD3.count('D') for i in range(3)]

            # Replace 'D' and calculate average for filtered_stageD1
            avg_stages = []
            std_stages = []
            median_stages = []
            for i in range(3):
                avg, std, median = ut.replace_D_and_calculate_avg_std(stages[i])
                avg_stages.append(avg)
                std_stages.append(std)
                median_stages.append(median) # currently not used

            avg_stages = [round(avg_stages[i],1) for i in range(3)]
            std_stages = [round(std_stages[i],1) for i in range(3)]
          
            if my_batch in longerwellplates: #48-well plates
                ut.plot_poster_48(fig, axes, font_title, font_text, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, colorlist, my_batch, my_condition, my_plate, image_paths[0], image_paths[1], image_paths[2], stages, surviving, avg_stages, std_stages, output_path_pdf, output_path_png, conditionlist, save, my_dpi)
            else:
                ut.plot_poster(fig, axes, font_title, font_text, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, colorlist, my_batch, my_condition, my_plate, image_paths[0], image_paths[1], image_paths[2], stages, surviving, avg_stages, std_stages, output_path_pdf, output_path_png, conditionlist, save, my_dpi)
