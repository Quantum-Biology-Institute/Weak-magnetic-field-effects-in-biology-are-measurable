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

my_bat = [1]
my_pl = [1,2,3,4]

###############################################################################
###############################################################################
my_conds = ['C', 'H']
my_conds_dict = {'C': 'control box', 'H': 'hypomagnetic chamber'}
fig_width = 8.5
fig_length = 11
my_dpi = 600

for my_batch in my_bat:
    
    for my_plate in my_pl:
        
        for my_condition in my_conds:

            fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(fig_width,fig_length), dpi=my_dpi) 
            
            plt.suptitle(f'Batch B{my_batch} $\cdot$ {my_conds_dict[my_condition]} $\cdot$ plate {my_plate}/{len(my_pl)} $\cdot$ days 1 through 7', fontsize=BIGGER_SIZE, fontproperties=font_title)

            # Replace these paths with the actual paths to your directories
            labels_day = ['1', '2', '3', '4', '5', '6', '7']
            image_paths = [base_path + f'2 Experimental overview/B{my_batch}/B{my_batch}{my_condition}/B{my_batch}{my_condition}D{labels_day[i]}/B{my_batch}{my_condition}D{labels_day[i]}P{my_plate}' for i in range(7)]

            assessment_path = base_path + f'2 Experimental overview/Assessments/B{my_batch}/B{my_batch}{my_condition}P{my_plate}_assessment.txt'

            output_path_pdf = base_path + f'2 Experimental overview/Results/B{my_batch}/B{my_batch}{my_condition}P{my_plate}_7days.pdf' # Change this path as needed
            output_path_png = base_path + f'2 Experimental overview/Results/B{my_batch}/B{my_batch}{my_condition}P{my_plate}_7days.png'

            # Read the file
            file_content = ut.read_file(assessment_path)

            # Extract the lists
            stages = [ut.extract_list(file_content, f'StageD{i+1}:') for i in range(3)]
            
            statusD3 = ut.extract_list(file_content, 'StatusD3:')
            colorlist = ut.map_status_to_color(statusD3)
            conditionlist = ut.extract_list(file_content, 'Condition:')
            
            if my_batch == 6: #48-well plates
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
          
            ut.plot_poster_7days(fig, axes, font_title, font_text, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, colorlist, my_batch, my_condition, my_plate, image_paths[0], image_paths[1], image_paths[2], image_paths[3], image_paths[4], image_paths[5], image_paths[6], stages, surviving, avg_stages, std_stages, output_path_pdf, output_path_png, conditionlist, save, my_dpi)

