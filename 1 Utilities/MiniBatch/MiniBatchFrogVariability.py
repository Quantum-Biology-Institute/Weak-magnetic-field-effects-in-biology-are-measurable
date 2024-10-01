###############################################################################
# BASE PATH
base_path = '/Users/clarice/Desktop/'
###############################################################################

import numpy as np
import matplotlib.pyplot as plt

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
SMALL_SIZE = 10
MEDIUM_SIZE = 14 # size of main text
BIGGER_SIZE = 20 # size of section text

###############################################################################
# CHANGE HERE ONLY
###############################################################################

save = True
my_dpi = 600
batches_to_analyze = [6,6]

my_frogs = [1,2]
plate_array = [[1,2,6], [3,4,5]]

#Mini batch:
#Plates 1,2,6 —- female 1
#Plates 3,4,5 -- female 2

D1 = False# WORKS FOR DAY 1
D2 = False#True#False # WORKS FOR DAY 2
D3 = True

#not hooked up in utilitis violin functions, the 2 below
# to modify in Utilities: plot_violins and plot_avg_violin
my_yticks = [[0.9, 0.95, 1.0], 
             [0.9, 0.95, 1.0], 
             [0.9, 0.95, 1.0], 
             [0.9, 0.95, 1.0], 
             [0.9, 0.95, 1.0], 
             [0.9, 0.95, 1.0],
             [0.9, 0.95, 1.0],
             [0.9, 0.95, 1.0],
             [0.9, 0.95, 1.0],
             [0.9, 0.95, 1.0],
             [0.9, 0.95, 1.0],
             [0.9, 0.95, 1.0]] 
my_yticklabels = my_yticks

###############################################################################
###############################################################################

# Test that only one of the days is being analyzed
ut.check_single_true(D1, D2, D3) 

if D1:
    tests                 = ['Area',   'Perimeter',   'Elongation',   'Roundness',    'Eccentricity', 'Solidity']
    variables             = ['Areas:', 'Perimeters:', 'Elongations:', 'Roundnesses:', 'Eccentricities:', 'Solidities:']
    my_day = '1'
    day_path = '3 D1 quantification/'
    miny     = [0.5, 0.5,  0.00, 0.70, 0.00, 0.95]
    maxy     = [1.5, 1.5,  0.35, 1.00, 1.00, 1.01]
    yticks = [[0.75, 1.0, 1.25], [0.75, 1.0, 1.25], [0, 0.1, 0.2, 0.3], [0.75, 0.8, 0.85, 0.9, 0.95], [0, 0.25, 0.5, 0.75, 1.0], [0.95, 0.975, 1.0]]
    miny_avg = miny
    maxy_avg = maxy
    
    my_lines = [[] for _ in range(len(tests))]
    my_labels = [[] for _ in range(len(tests))]
    directory_path = base_path + "1 Utilities/Test images/ResultsD1/"
    initial_stage = 20
    end_stage = 24
    
    my_lines[0] = [1.0] # Area
    my_lines[1] = [1.0] # Perimeter
    my_lines[2] = ut.extract_mean_quantity(directory_path, "Mean elongation:", initial_stage, end_stage) # Elongation
    my_lines[3] = ut.extract_mean_quantity(directory_path, "Mean roundness:", initial_stage, end_stage)      # Roundness
    my_lines[4] = ut.extract_mean_quantity(directory_path, "Mean eccentricity:", initial_stage, end_stage)     # Eccentricity
    my_aux = ut.extract_mean_quantity(directory_path, "Mean solidity:", initial_stage, end_stage)     # Solidity
    
    #Check if they increase/decrease as expected
    if not ut.is_increasing(my_lines[2]): # Elongation should be increasing
        raise ValueError("Elongation is not strictly increasing!")
    if not ut.is_decreasing(my_lines[3]): # Roundness should be decreasing
        raise ValueError("Roundness is not strictly decreasing!")
    if not ut.is_increasing(my_lines[4]): # Eccentricity should be increasing
        raise ValueError("Eccentricity is not strictly increasing!")   
    if not ut.is_decreasing(my_lines[5]): # Roundness should be decreasing
        raise ValueError("Solidity is not strictly decreasing!")
        
    for my_index in [2,3,4]:
        my_labels[my_index] = [f'stage {i}' for i in range(initial_stage, end_stage + 1)]
    my_labels[5] = ['stage 20', 'stages 21, 22', 'stage 23']
    my_lines[5] = [my_aux[0], my_aux[2], my_aux[3]]
    
elif D2:
    tests             = ['Max length',  'RGB Mean R' , 'RGB Mean G',  'RGB Mean B',  'Lab Mean L',  'Lab Mean A',  'Lab Mean B', 'Yellow R',  'Yellow G',  'Yellow B']
    variables         = ['Max lengths:','Mean R:',     'Mean G:',     'Mean B:',     'Mean L:',     'Mean A:',     'Mean BB:',   'Yellow R:', 'Yellow G:', 'Yellow B:' ]
    my_day = '2'
    day_path = '4 D2 quantification/'
    miny     = [0.5,   0,   0,   0,   0, -128, -128,   0,   0,   0] 
    maxy     = [1.5, 255, 255, 255, 100,  127,  127, 255, 255, 255] 
    yticks = [[0.75, 1.0, 1.25], [0, 127.5, 255], [0, 127.5, 255], [0, 127.5, 255],[0, 127.5, 255],[0, 127.5, 255],[0, 127.5, 255],[0, 127.5, 255],[0, 127.5, 255],[0, 127.5, 255]]
    miny_avg = miny
    maxy_avg = maxy
    
    my_lines = [[] for _ in range(len(tests))]
    my_labels = [[] for _ in range(len(tests))]
    my_lines[0] = [1.0] # Max length
    my_labels.append('')
    for my_index in [1,2,3,4,5,5,6,7,9]:
        my_lines[my_index] = []
        my_labels[my_index] = ''
    
    
elif D3:
    tests             = ['Max length',   'Eye size',   'Pigmentation size',   'RGB Mean R' , 'RGB Mean G',  'RGB Mean B',  'Lab Mean L',  'Lab Mean A',  'Lab Mean B', 'Yellow R',  'Yellow G',  'Yellow B']
    variables         = ['Max lengths:', 'Eye Areas:', 'Pigmentation Areas:', 'Mean R:',     'Mean G:',     'Mean B:',     'Mean L:',     'Mean A:',     'Mean BB:',   'Yellow R:', 'Yellow G:', 'Yellow B:']
    my_day = '3'
    day_path = '5 D3 quantification/'
    miny     = [0.5, 0.75, 0,75,   0,   0,   0,   0, -128, -128,   0,   0,   0] 
    maxy     = [1.5, 1.25, 1.25, 255, 255, 255, 100,  127,  127, 255, 255, 255] 
    yticks = [[0.75, 1.0, 1.25], [0.75, 1.0, 1.25], [0.75, 1.0, 1.25],[0, 127.5, 255], [0, 127.5, 255], [0, 127.5, 255],[0, 127.5, 255],[0, 127.5, 255],[0, 127.5, 255],[0, 127.5, 255],[0, 127.5, 255],[0, 127.5, 255]]
    miny_avg = miny
    maxy_avg = maxy
    
    my_lines = [[] for _ in range(len(tests))]
    my_labels = [[] for _ in range(len(tests))]
    my_lines[0] = [1.0] # Max length
    my_labels.append('')
    for my_index in [1,2,3,4,5,5,6,7,9]:
        my_lines[my_index] = []
        my_labels[my_index] = ''
else: 
    print('No day chosen')
    
###############################################################################
###############################################################################

for idx in range(len(tests)):
    
    print(f'************** Working on plot for {tests[idx]}!')
    hor_lines  = my_lines[idx]
    hor_labels = my_labels[idx]
    
    concatenated = []
    conditionlabels = []
    filenames = [[],[]]
    assess_filenames = [[], []]
    modified_filenames = [[], []]
    d3_status = [[],[]]
    common_filenames = [[], []]
    filtered_status_d3 = [[], []]
    stickcolors = [[], []]

    healthy_concatenated = []
    healthy_stickcolors = [[],[]]
    healthy_conditionlabels = []
    
    my_ind_batch = 0

    for my_frog in my_frogs: 

        stats_path = [base_path + day_path + f"Results/B6Frog{my_frog}/B6Frog{my_frog}{'C' if my_idx == 0 else 'H'}D{my_day}_analysis.txt" for my_idx in range(2)]
            
        if tests[idx] in ['Area', 'Perimeter', 'Max length', 'Eye size', 'Pigmentation size']: # Normalize to mean of Control
            control_data = ut.extract_array(stats_path[0], variables[idx])/np.mean(ut.extract_array(stats_path[0], variables[idx]))
            hypo_data = ut.extract_array(stats_path[1], variables[idx])/np.mean(ut.extract_array(stats_path[0], variables[idx]))
        else:
            control_data = ut.extract_array(stats_path[0], variables[idx])
            hypo_data = ut.extract_array(stats_path[1], variables[idx])

        concatenated.append(control_data)
        conditionlabels.append(f'B6 Frog{my_frog} C\n(n={len(control_data)})')
        concatenated.append(hypo_data)
        conditionlabels.append(f'B6 Frog{my_frog} H\n(n={len(hypo_data)})')
        
        # Read the file and extract the necessary lists
        for my_ind in range(2):
            c_content = ut.read_file(stats_path[my_ind])
            filenames[my_ind] = ut.extract_list(c_content, 'Filenames:')  # Filenames in D1_analysis; all plates in the same txt file

        # Read the assessment files, for all plates in the batches
        for my_ind in range(2):
            for my_plate in plate_array[my_frog-1]:
                assess_path = [f"/Users/clarice/Desktop/2 Experimental overview/Assessments/B6Frog{my_frog}/B6{'C' if my_idx == 0 else 'H'}P{my_plate}_assessment.txt" for my_idx in range(2)]
                c_content = ut.read_file(assess_path[my_ind])
                assess_filenames[my_ind].extend(ut.extract_list(c_content, 'Filenames:'))  # Filenames separate plate assessment txt files
                d3_status[my_ind].extend(ut.extract_list(c_content, 'StatusD3:'))

            # Modify each filename for comparison: include 'D{my_day}' and add '.png'
            modified_filenames[my_ind] = [f"{filename[:3]}D{my_day}{filename[3:]}.png" for filename in assess_filenames[my_ind]]

            # Find common filenames
            common_filenames[my_ind] = set(modified_filenames[my_ind]).intersection(set(filenames[my_ind]))
        
            # Filter status_d3 based on common filenames
            filtered_status_d3[my_ind] = [d3_status[my_ind][modified_filenames[my_ind].index(fname)] for fname in common_filenames[my_ind]]

            # Map status to colors and save to framecolors
            stickcolors[my_ind].append(ut.map_status_to_color(filtered_status_d3[my_ind]))

            # Filter for healthy (StatusD3 = 1) cases
            healthy_indices = [i for i, status in enumerate(filtered_status_d3[my_ind]) if status == 1]

            if my_ind == 0:  # Control data
                healthy_concatenated.append([control_data[i] for i in healthy_indices])
                healthy_stickcolors[my_ind].append([stickcolors[my_ind][-1][i] for i in healthy_indices])
                healthy_conditionlabels.append(f'B6 Frog{my_frog} C\n(n={len(healthy_indices)})')  # Add healthy condition label for control
            else:  # Hypo data
                healthy_concatenated.append([hypo_data[i] for i in healthy_indices])
                healthy_stickcolors[my_ind].append([stickcolors[my_ind][-1][i] for i in healthy_indices])
                healthy_conditionlabels.append(f'B6 Frog{my_frog} H\n(n={len(healthy_indices)})')  # Add healthy condition label for hypo

        my_ind_batch = my_ind_batch + 1 
        
    # Calculate the sum of the lengths of the even-indexed elements in concatenated
    sum_even_concatenated = sum(len(concatenated[i]) for i in range(0, len(concatenated), 2))
    # Calculate the sum of the lengths of the odd-indexed elements in concatenated
    sum_odd_concatenated = sum(len(concatenated[i]) for i in range(1, len(concatenated), 2))
    # CCalculate the sum of the stickcolors
    sum_even_stickcolors = sum(len(stickcolors[0][i]) for i in range(len(stickcolors[0])))
    sum_odd_stickcolors = sum(len(stickcolors[1][i]) for i in range(len(stickcolors[1])))
    # Compare and raise an error if they do not match
    if sum_even_concatenated != sum_even_stickcolors:
        raise ValueError(f"Mismatch: sum of even concatenated lengths ({sum_even_concatenated}) != sum of even stickcolors lengths ({sum_even_stickcolors})")
    if sum_odd_concatenated != sum_odd_stickcolors:
        raise ValueError(f"Mismatch: sum of odd concatenated lengths ({sum_odd_concatenated}) != sum of odd stickcolors lengths ({sum_odd_stickcolors})")
    # Debug: Print lengths of concatenated data and stick_colors
    # for i, data in enumerate(concatenated):
    #     print(f"concatenated[{i}] length: {len(data)}")
    # for i, batch in enumerate(stickcolors):
    #     for j, condition in enumerate(batch):
    #         print(f"stickcolors[{i}][{j}] length: {len(condition)}")
        
    
    ### Individual violins
    xpos = list(range(2*len(batches_to_analyze))) # number of positions in x axis
    
    # Remove last stage lines (usually stage 24)
    if tests[idx] in ['Eccentricity', 'Roundness', 'Elongation']:
        hor_lines.pop()
        hor_labels.pop()
        
    # Extract and concatenate the even-numbered arrays, corresponding to Control
    even_arrays = np.concatenate([concatenated[i] for i in range(0, len(concatenated), 2)])
    # Extract and concatenate the odd-numbered arrays, corresponding to Hypo
    odd_arrays = np.concatenate([concatenated[i] for i in range(1, len(concatenated), 2)])
    # Flatten stickcolors
    #stickcolors2[0] = [color for sublist in stickcolors[0] for color in sublist]
    #stickcolors2[1] = [color for sublist in stickcolors[1] for color in sublist]
    # Ensure that the length of stickcolors matches the concatenated arrays
    #if len(stickcolors2[0]) != len(even_arrays):
     #   raise ValueError("Mismatch between the number of colors in stickcolors[0] and the data in even_arrays")
    #if len(stickcolors2[1]) != len(odd_arrays):
     #   raise ValueError("Mismatch between the number of colors in stickcolors[1] and the data in odd_arrays")

    healthy_even_arrays = np.concatenate([healthy_concatenated[i] for i in range(0, len(healthy_concatenated), 2)])
    healthy_odd_arrays = np.concatenate([healthy_concatenated[i] for i in range(1, len(healthy_concatenated), 2)])
    # Flatten healthy_stickcolors
    #healthy_stickcolors[0] = [color for sublist in healthy_stickcolors[0] for color in sublist]
    #healthy_stickcolors[1] = [color for sublist in healthy_stickcolors[1] for color in sublist]
    # Ensure that the length of healthy_stickcolors matches the healthy concatenated arrays
    #if len(healthy_stickcolors[0]) != len(healthy_even_arrays):
     #   raise ValueError("Mismatch between the number of colors in healthy_stickcolors[0] and the data in healthy_even_arrays")
    #if len(healthy_stickcolors[1]) != len(healthy_odd_arrays):
     #   raise ValueError("Mismatch between the number of colors in healthy_stickcolors[1] and the data in healthy_odd_arrays")
        
    ### All stats
    batches_str = '_'.join(map(str, batches_to_analyze))
    save_avg_path = base_path + day_path + f'Results/MiniBatch/Batches{batches_str}_{tests[idx]}_Comparison.txt'
    save_avg_path_healthy = base_path + day_path + f'Results//MiniBatch/Batches{batches_str}_{tests[idx]}_Comparison_HEALTHY.txt'
    mannwhitney_results, dunn_results = ut.stats_and_save_minibatch(save_avg_path, concatenated, even_arrays, odd_arrays, batches_str, conditionlabels)
    mannwhitney_results_healthy, dunn_results_healthy = ut.stats_and_save_minibatch(save_avg_path_healthy, healthy_concatenated, healthy_even_arrays, healthy_odd_arrays, batches_str, conditionlabels)
       
    # Batch statistical comparison
    for my_first_ind in range(len(concatenated)):
        for my_second_ind in range(my_first_ind + 1, len(concatenated)):
            save_stats_path_batch = base_path + day_path + f'Results/MiniBatch/BatchStats/{tests[idx]}_{conditionlabels[my_first_ind]}_{conditionlabels[my_second_ind]}_Comparison.txt'
            ut.perform_pairwise_tests([concatenated[my_first_ind], concatenated[my_second_ind]], [conditionlabels[my_first_ind], conditionlabels[my_second_ind]], save_stats_path_batch, save=True)
          
    for my_first_ind in range(len(healthy_concatenated)):
        for my_second_ind in range(my_first_ind + 1, len(healthy_concatenated)):
            save_stats_path_batch_healthy = base_path + day_path + f'Results/MiniBatch/BatchStats/{tests[idx]}_{conditionlabels[my_first_ind]}_{conditionlabels[my_second_ind]}_Comparison_HEALTHY.txt'
            ut.perform_pairwise_tests([healthy_concatenated[my_first_ind], healthy_concatenated[my_second_ind]], [conditionlabels[my_first_ind], conditionlabels[my_second_ind]], save_stats_path_batch_healthy, save=True)

    #mannwhitney_results[0][3] #Frog1C + Frog 1H
    #mannwhitney_results[1][3] #Frog2C + Frog 2H
    #value = dunn_results.loc['Batch2C', 'Batch1C'] #Value of Dunn    
    p_array = [""] * 3  # Initialize a list with 3 empty strings
    my_alpha = 0.01
    if dunn_results.loc['Batch2C', 'Batch1C'] >= my_alpha:    
        p_array[0] = 'n.s.'
    else:
        p_array[0] = 'p = ' + "{:.2e}".format(dunn_results.loc['Batch2C', 'Batch1C'])   
    if mannwhitney_results[0][3] >= my_alpha: #Frog1C + Frog 1H
        p_array[1] = 'n.s.'
    else:
        p_array[1] = 'p = ' + "{:.2e}".format(mannwhitney_results[0][3])
    if mannwhitney_results[1][3] >= my_alpha: #Frog1C + Frog 1H
        p_array[2] = 'n.s.'
    else:
        p_array[2] = 'p = ' + "{:.2e}".format(mannwhitney_results[1][3])    
   
    fig1, axs1 = plt.subplots(dpi=my_dpi)
    plt.suptitle(tests[idx] + f' by mini-batch $\cdot$ Day {my_day} $\cdot$ all states' , fontsize=BIGGER_SIZE, fontproperties=font_title)
    ut.plot_violins_adjusted(fig1, 
                    axs1, 
                    concatenated, 
                    conditionlabels, 
                    stickcolors, 
                    xpos, 
                    miny[idx], maxy[idx], 
                    yticks[idx],
                    font_title, font_text, 
                    SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE,
                    hor_lines, hor_labels,
                    p_array)

    p_array_healthy = [""] * 3  # Initialize a list with 3 empty strings
    if dunn_results_healthy.loc['Batch2C', 'Batch1C'] >= my_alpha:    
        p_array_healthy[0] = 'n.s.'
    else:
        p_array_healthy[0] = 'p = ' + "{:.2e}".format(dunn_results_healthy.loc['Batch2C', 'Batch1C'])   
    if mannwhitney_results_healthy[0][3] >= my_alpha: #Frog1C + Frog 1H
        p_array_healthy[1] = 'n.s.'
    else:
        p_array_healthy[1] = 'p = ' + "{:.2e}".format(mannwhitney_results_healthy[0][3])
    if mannwhitney_results_healthy[1][3] >= my_alpha: #Frog1C + Frog 1H
        p_array_healthy[2] = 'n.s.'
    else:
        p_array_healthy[2] = 'p = ' + "{:.2e}".format(mannwhitney_results_healthy[1][3]) 
        
    ### Healthy individual violins
    fig1_healthy, axs1_healthy = plt.subplots(dpi=my_dpi)
    plt.suptitle(tests[idx] + f' by mini-batch $\cdot$ Day {my_day} $\cdot$ only healthy', fontsize=BIGGER_SIZE, fontproperties=font_title)
    ut.plot_violins_adjusted(fig1_healthy, 
                    axs1_healthy, 
                    healthy_concatenated, 
                    healthy_conditionlabels, 
                    healthy_stickcolors, 
                    xpos, 
                    miny[idx], maxy[idx], 
                    yticks[idx],
                    font_title, font_text, 
                    SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE,
                    hor_lines, hor_labels,
                    p_array_healthy)
 
###############################################################################

output_path = base_path + day_path + 'Results/MiniBatch/'

# Save all open figures
if save:        
    fig_numbers = plt.get_fignums()
    num_figs = len(fig_numbers) // 2  # Each test has 4 figures: fig1, fig1_healthy

    for i in range(num_figs):
        fig1 = plt.figure(fig_numbers[2 * i])
        fig1_healthy = plt.figure(fig_numbers[2 * i + 1])
        
        # Save the regular figures
        fig1.savefig(output_path + f'Batches{batches_str}_{tests[i]}_each_batch.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
        fig1.savefig(output_path + f'Batches{batches_str}_{tests[i]}_each_batch.png', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PNG
        
        # Save the healthy figures
        fig1_healthy.savefig(output_path + f'Batches{batches_str}_{tests[i]}_each_batch_HEALTHY.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
        fig1_healthy.savefig(output_path + f'Batches{batches_str}_{tests[i]}_each_batch_HEALTHY.png', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PNG
        
plt.show()