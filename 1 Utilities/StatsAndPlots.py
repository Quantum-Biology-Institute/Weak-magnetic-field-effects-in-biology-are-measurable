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
import SetupDays as SD

import plot_config
# Initialize fonts and settings
font_title, font_text, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = plot_config.setup()
SMALL_SIZE = 10
MEDIUM_SIZE = 14 # size of main text
BIGGER_SIZE = 20 # size of section text

from sklearn.utils import resample

###############################################################################
# CHANGE HERE ONLY
###############################################################################

D1 = False
D2 = False
D3 = True

save = True
batches_to_analyze = [1,2,3,4,5,6,7,8,9,10]

PositiveControls = 3 # Last 3 batches are positive control

###############################################################################
###############################################################################

# Test that only one of the days is being analyzed
ut.check_single_true(D1, D2, D3) 

if D1:
    my_day = '1'
    day_path = '3 D1 quantification/'
    
    tests, variables, miny, maxy, yticks, my_lines, my_labels, my_units = SD.Setup_D1();
    miny_avg = miny
    maxy_avg = maxy
 
elif D2:
    my_day = '2'
    day_path = '4 D2 quantification/'

    tests, variables, miny, maxy, yticks, my_lines, my_labels, my_units = SD.Setup_D2D3();
    miny_avg = miny
    maxy_avg = maxy
            
elif D3:
    my_day = '3'
    day_path = '5 D3 quantification/'
   
    tests, variables, miny, maxy, yticks, my_lines, my_labels, my_units = SD.Setup_D2D3();
    miny_avg = miny
    maxy_avg = maxy
    
else: 
    print('No day chosen')
    

###############################################################################
###############################################################################
# Number of plates per batch; note: index 0 corresponds to batch 1, etc
no_plates = [4, 5, 5, 1, 2, 6, 4, 6, 1, 10]

for idx in range(len(tests)):
    
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
    
    my_fil = []

    for my_batch in batches_to_analyze: 

        stats_path = [base_path + day_path + f"Results/B{my_batch}/B{my_batch}{'C' if my_idx == 0 else 'H'}D{my_day}_analysis.txt" for my_idx in range(2)]
            
        if my_units[idx] == 'norm.': 
            # Normalize to mean of Control
            control_data = ut.extract_array(stats_path[0], variables[idx])/np.mean(ut.extract_array(stats_path[0], variables[idx]))
            hypo_data = ut.extract_array(stats_path[1], variables[idx])/np.mean(ut.extract_array(stats_path[0], variables[idx]))
        else:
            control_data = ut.extract_array(stats_path[0], variables[idx])
            hypo_data = ut.extract_array(stats_path[1], variables[idx])
            
        concatenated.append(control_data)
        conditionlabels.append(f'B{my_batch}C')
        concatenated.append(hypo_data)
        conditionlabels.append(f'B{my_batch}H')
        
        # Read the file and extract the necessary lists
        for my_ind in range(2):
            c_content = ut.read_file(stats_path[my_ind])
            filenames[my_ind] = ut.extract_list(c_content, 'Filenames:')  # Filenames in D1_analysis; all plates in the same txt file

        # Read the assessment files, for all plates in the batches
        for my_ind in range(2):
            for my_plate in range(1, no_plates[my_batch - 1] + 1):
                assess_path = [f"/Users/clarice/Desktop/2 Experimental overview/Assessments/B{my_batch}/B{my_batch}{'C' if my_idx == 0 else 'H'}P{my_plate}_assessment.txt" for my_idx in range(2)]
                c_content = ut.read_file(assess_path[my_ind])
                assess_filenames[my_ind].extend(ut.extract_list(c_content, 'Filenames:'))  # Filenames separate plate assessment txt files
                d3_status[my_ind].extend(ut.extract_list(c_content, 'StatusD3:'))

            # Modify each filename for comparison: include 'D{my_day}' and add '.png' -- won't work for batches > 9!!!
            #modified_filenames[my_ind] = [f"{filename[:3]}D{my_day}{filename[3:]}.png" for filename in assess_filenames[my_ind]]
            
            # Modify each filename for comparison: include 'D{my_day}' and add '.png'
            modified_filenames[my_ind] = [
                f"{filename[:filename.index('H') + 1]}D{my_day}{filename[filename.index('H') + 1:]}.png"
                if 'H' in filename 
                else f"{filename[:filename.index('C') + 1]}D{my_day}{filename[filename.index('C') + 1:]}.png"
                for filename in assess_filenames[my_ind]
                ]

            # Find common filenames
            common_filenames[my_ind] = set(modified_filenames[my_ind]).intersection(set(filenames[my_ind]))
        
            # Filter status_d3 based on common filenames
            filtered_status_d3[my_ind] = [d3_status[my_ind][modified_filenames[my_ind].index(fname)] for fname in common_filenames[my_ind]]
            # if there's a mismatch, debug:
            # print(f"Common filenames: {len(common_filenames[my_ind])}, "
            #       f"Filtered Status D3: {len(filtered_status_d3[my_ind])}")

            # Map status to colors and save to framecolors
            stickcolors[my_ind].append(ut.map_status_to_color(filtered_status_d3[my_ind]))
            # if there's a mismatch, debug:
            # print(f"Batch {my_batch}, Condition {my_ind}, # of filenames: {len(filenames[my_ind])}, "
            #       f"# of modified_filenames: {len(modified_filenames[my_ind])}, "
            #       f"# of stickcolors: {len(stickcolors[my_ind][-1])}")
            
            # if len(common_filenames[my_ind]) != len(stickcolors[my_ind][my_batch-1]):
            #     for i, filename in enumerate(common_filenames[my_ind]):
            #         if i >= len(stickcolors[my_ind][my_batch-1]):
            #             print(f"Missing stickcolor for: {filename}")
            #     raise ValueError(f"Mismatch in batch {my_batch}, condition {my_ind}.")
            

            # Filter for healthy (StatusD3 = 1) cases
            healthy_indices = [i for i, status in enumerate(filtered_status_d3[my_ind]) if status == 1]
            
            # Debugging information
            # print(f"Batch {my_batch}, condition {my_ind} ")            
            # print(f"Length of common_filenames[{my_ind}]: {len(common_filenames[my_ind])}")
            # print(f"Length of concatenated[{2*(my_batch-1) + my_ind}]: {len(concatenated[2*(my_batch-1) + my_ind])}")
            # print(f"Length of filtered_status_d3[{my_ind}]: {len(filtered_status_d3[my_ind])}")
            # print(f"Length of healthy_indices: {len(healthy_indices)}\n\n")

            if my_ind == 0:  # Control data
                healthy_concatenated.append([control_data[i] for i in healthy_indices])
                healthy_stickcolors[my_ind].append([stickcolors[my_ind][-1][i] for i in healthy_indices])
            else:  # Hypo data
                healthy_concatenated.append([hypo_data[i] for i in healthy_indices])
                healthy_stickcolors[my_ind].append([stickcolors[my_ind][-1][i] for i in healthy_indices])
                
            # if True:
            #     print(my_batch)
            #     print(my_ind)
            #     print(f"\n---- Debugging Batch {my_batch} ----")
            #     print(f"Condition {my_ind}:")
    
            #     # Print all common filenames and stickcolors for comparison
            #     print(f"Common filenames: {len(common_filenames[my_ind])}")
            #     print(f"Stickcolors length: {len(stickcolors[my_ind][-1])}")
    
            #     # Check for mismatched filenames (i.e., filenames in `common_filenames` but not in `stickcolors`)
            #     for i, fname in enumerate(common_filenames[my_ind]):
            #         if i >= len(stickcolors[my_ind][-1]):
            #             print(f"Mismatch - filename found in common_filenames but missing stickcolor: {fname}")
    
            #     # Check for extra stickcolors entries
            #     if len(stickcolors[my_ind][-1]) > len(common_filenames[my_ind]):
            #         print(f"Extra stickcolor for Condition {my_ind}: Stickcolor has more entries than common_filenames")
            
    # Debug
    #extra_image = ut.find_extra_image(filenames, filtered_status_d3)        

    # Calculate the sum of the lengths of the even-indexed elements in concatenated
    sum_even_concatenated = sum(len(concatenated[i]) for i in range(0, len(concatenated), 2))
    # Calculate the sum of the lengths of the odd-indexed elements in concatenated
    sum_odd_concatenated = sum(len(concatenated[i]) for i in range(1, len(concatenated), 2))
    # Calculate the sum of the stickcolors
    sum_even_stickcolors = sum(len(stickcolors[0][i]) for i in range(len(stickcolors[0])))
    sum_odd_stickcolors = sum(len(stickcolors[1][i]) for i in range(len(stickcolors[1])))
    # Compare and raise an error if they do not match
    
    # #Debug: Print lengths of concatenated data and stick_colors
    # for i, data in enumerate(concatenated):
    #     print(f"concatenated[{i}] length: {len(data)}")

    # for i, batch in enumerate(stickcolors):
    #     for j, condition in enumerate(batch):
    #         print(f"stickcolors[{i}][{j}] length: {len(condition)}")

    # Print the actual difference and batch where the mismatch occurs
    # if sum_even_concatenated != sum_even_stickcolors:
    #     print("Mismatch found in even-indexed elements.")
    #     for i in range(0, len(concatenated), 2):
    #         if len(concatenated[i]) != len(stickcolors[0][i//2]):
    #             print(f"Mismatch in batch {i//2 + 1}, Control group:")
    #             print(f"concatenated[{i}] length: {len(concatenated[i])}, stickcolors length: {len(stickcolors[0][i//2])}")
    #             raise ValueError(f"Mismatch in batch {i//2 + 1}, Control group.")

    # if sum_odd_concatenated != sum_odd_stickcolors:
    #     print("Mismatch found in odd-indexed elements.")
    #     for i in range(1, len(concatenated), 2):
    #         if len(concatenated[i]) != len(stickcolors[1][i//2]):
    #             print(f"Mismatch in batch {i//2 + 1}, Hypo group:")
    #             print(f"concatenated[{i}] length: {len(concatenated[i])}, stickcolors length: {len(stickcolors[1][i//2])}")
    #             raise ValueError(f"Mismatch in batch {i//2 + 1}, Hypo group.")
    
    #if there's a mismatch, debug:
    # print(f"Even: concatenated lengths = {sum_even_concatenated}, stickcolors lengths = {sum_even_stickcolors}")
    # print(f"Odd: concatenated lengths = {sum_odd_concatenated}, stickcolors lengths = {sum_odd_stickcolors}")
 
    if sum_even_concatenated != sum_even_stickcolors:
        raise ValueError(f"Mismatch: sum of even concatenated lengths ({sum_even_concatenated}) != sum of even stickcolors lengths ({sum_even_stickcolors})")
    if sum_odd_concatenated != sum_odd_stickcolors:
        raise ValueError(f"Mismatch: sum of odd concatenated lengths ({sum_odd_concatenated}) != sum of odd stickcolors lengths ({sum_odd_stickcolors})")    
    
    ### Individual violins
    xpos = list(range(2*len(batches_to_analyze))) # number of positions in x axis
    
    conditionlabels = ut.rename_labels(conditionlabels, PositiveControls)
    
    fig1a, axs1a = plt.subplots()
    if my_units[idx] != '':
        plt.suptitle(f'a) Day {my_day} $\cdot$ ' + tests[idx] + f' ({my_units[idx]}) $\cdot$ all states\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0', fontsize=BIGGER_SIZE, fontproperties=font_title)
    else:
        plt.suptitle(f'a) Day {my_day} $\cdot$ ' + tests[idx] + ' $\cdot$ all states\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0', fontsize=BIGGER_SIZE, fontproperties=font_title)
    ut.plot_violins(fig1a, 
                    axs1a, 
                    concatenated, 
                    conditionlabels, 
                    stickcolors, 
                    xpos, 
                    miny[idx], maxy[idx], 
                    yticks[idx],
                    font_title, font_text, 
                    SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE,
                    hor_lines, hor_labels)

    fig1a_width, fig1a_height = fig1a.get_size_inches()
    ### Healthy individual violins
    fig1b, axs1b = plt.subplots(figsize=(fig1a_width, fig1a_height))
    if my_units[idx] != '':
        plt.suptitle(f'a) Day {my_day} $\cdot$ ' + tests[idx] + f' ({my_units[idx]}) $\cdot$ only healthy' , fontsize=BIGGER_SIZE, fontproperties=font_title)
    else:
        plt.suptitle(f'a) Day {my_day} $\cdot$ ' + tests[idx] + ' $\cdot$ only healthy' , fontsize=BIGGER_SIZE, fontproperties=font_title)
    ut.plot_violins(fig1b, 
                    axs1b, 
                    healthy_concatenated, 
                    conditionlabels, 
                    healthy_stickcolors, 
                    xpos, 
                    miny[idx], maxy[idx], 
                    yticks[idx],
                    font_title, font_text, 
                    SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE,
                    hor_lines, hor_labels)
    
    ### Average violin
    fig2a, axs2a = plt.subplots(figsize=(fig1a_width, fig1a_height))
    if my_units[idx] != '':
        plt.suptitle(f'b) Day {my_day} $\cdot$ ' + tests[idx] + f' ({my_units[idx]}) $\cdot$ all states\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0', fontsize=BIGGER_SIZE, fontproperties=font_title)
    else:
        plt.suptitle(f'b) Day {my_day} $\cdot$ ' + tests[idx] + ' $\cdot$ all states\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0', fontsize=BIGGER_SIZE, fontproperties=font_title)
    
    ### Note that even_ and odd_ arrays only have experiment conditions, no positive controls!!!
    even_arrays, odd_arrays, stickcolors, even_arrays_pc, odd_arrays_pc, stickcolors_pc = ut.process_arrays(concatenated, stickcolors, PositiveControls)

    labels = ut.generate_tuple(concatenated, PositiveControls)
    ut.plot_avg_violin(fig2a, 
                       axs2a, 
                       even_arrays, 
                       odd_arrays, 
                       stickcolors, 
                       labels,
                       miny_avg[idx], maxy_avg[idx], 
                       yticks[idx], 
                       font_title, font_text, 
                       SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE,
                       hor_lines, hor_labels,
                       even_arrays_pc, odd_arrays_pc, stickcolors_pc, PositiveControls)

    ### Healthy average violin
    fig2b, axs2b = plt.subplots(figsize=(fig1a_width, fig1a_height))
    if my_units[idx] != '':
        plt.suptitle(f'b) Day {my_day} $\cdot$ ' + tests[idx] + f' ({my_units[idx]}) $\cdot$ only healthy', fontsize=BIGGER_SIZE, fontproperties=font_title)
    else:
        plt.suptitle(f'b) Day {my_day} $\cdot$ ' + tests[idx] + ' $\cdot$ only healthy', fontsize=BIGGER_SIZE, fontproperties=font_title)
    
    healthy_even_arrays, healthy_odd_arrays, healthy_stickcolors, healthy_even_arrays_pc, healthy_odd_arrays_pc, healthy_stickcolors_pc = ut.process_arrays(healthy_concatenated, healthy_stickcolors, PositiveControls)
    
    ut.plot_avg_violin(fig2b, 
                       axs2b, 
                       healthy_even_arrays, 
                       healthy_odd_arrays, 
                       healthy_stickcolors, 
                       labels,
                       miny_avg[idx], maxy_avg[idx], 
                       yticks[idx],
                       font_title, font_text, 
                       SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE,
                       hor_lines, hor_labels,
                       healthy_even_arrays_pc, healthy_odd_arrays_pc, healthy_stickcolors_pc, PositiveControls)

    batches_str = '_'.join(map(str, batches_to_analyze))
    save_avg_path = base_path + day_path + f'Results/D{my_day}_{tests[idx]}/Batches{batches_str}_{tests[idx]}_Comparison.txt'
    save_avg_path_healthy = base_path + day_path + f'Results/D{my_day}_{tests[idx]}/Batches{batches_str}_{tests[idx]}_Comparison_HEALTHY.txt'
    
    if PositiveControls != 0:
        my_str = batches_str[:-2*PositiveControls]
        
    dunn_results_C,   dunn_results_H,   kw_statistic_C,   p_value_kw_C,   kw_statistic_H,   p_value_kw_H,   ARTANOVA,   tukey_interaction_ARTANOVA  = ut.stats_and_save(save_avg_path,
                                                                                                                               concatenated, 
                                                                                                                               even_arrays, odd_arrays, 
                                                                                                                               my_str,
                                                                                                                               conditionlabels)
    dunn_results_C_h, dunn_results_H_h, kw_statistic_C_h, p_value_kw_C_h, kw_statistic_H_h, p_value_kw_H_h, ARTANOVA_h, tukey_interaction_ARTANOVA_h = ut.stats_and_save(save_avg_path_healthy, 
                                                                                                                               healthy_concatenated, 
                                                                                                                               healthy_even_arrays, healthy_odd_arrays, 
                                                                                                                               my_str, 
                                                                                                                               conditionlabels) 
        
    # Apply log10 transformation (ignoring NaNs and negative values) for both datasets
    log_dunn_results_C = np.log10(dunn_results_C)
    log_dunn_results_H = np.log10(dunn_results_H)

    # Replace -inf with NaN for clarity
    log_dunn_results_C[log_dunn_results_C == -np.inf] = np.nan
    log_dunn_results_H[log_dunn_results_H == -np.inf] = np.nan

    # Calculate global min and max for the log-transformed values
    global_vmin = np.nanmin([log_dunn_results_C, log_dunn_results_H])
    global_vmax = np.nanmax([log_dunn_results_C, log_dunn_results_H])
    
    ##### DUNN POSTHOC IS VALID IF KW P VALUE < 0.01
    if my_units[idx] != '':
        plot_title = f'g) Day {my_day} $\cdot$ ' + tests[idx] + f' ({my_units[idx]}) $\cdot$ all states'
    else:
        plot_title = f'g) Day {my_day} $\cdot$ ' + tests[idx] + ' $\cdot$ all states'
    my_str = 'K.-W. stat. = ' + "{:.1f}".format(kw_statistic_C) + f' $\cdot$ p value = {p_value_kw_C:.2e}' 
    fig3a, axs3a = plt.subplots()
    ut.plot_dunn_test_matrix(fig3a, axs3a, dunn_results_C, plot_title, font_title, font_text, 
                             SMALL_SIZE, 0.74*MEDIUM_SIZE, 0.7*BIGGER_SIZE, 
                             my_str, vmin=global_vmin, vmax=global_vmax)
    
    fig3a_width, fig3a_height = fig3a.get_size_inches()
    my_str = 'K.-W. stat. = ' + "{:.1f}".format(kw_statistic_H) + f' $\cdot$ p value = {p_value_kw_H:.2e}' 
    fig3b, axs3b = plt.subplots(figsize=(fig3a_width, fig3a_height))
    ut.plot_dunn_test_matrix(fig3b, axs3b, dunn_results_H, plot_title, font_title, font_text, 
                             SMALL_SIZE, 0.74*MEDIUM_SIZE, 0.7*BIGGER_SIZE, my_str, vmin=global_vmin, vmax=global_vmax, ishypo=True)
    
    if my_units[idx] != '':
        plot_title_h = f'h) Day {my_day} $\cdot$ ' + tests[idx] + f' ({my_units[idx]}) $\cdot$ only healthy'
    else:
        plot_title_h = f'h) Day {my_day} $\cdot$ ' + tests[idx] + ' $\cdot$ only healthy'
    my_str = 'K.-W. stat. = ' + "{:.1f}".format(kw_statistic_C_h) + f' $\cdot$ p value = {p_value_kw_C_h:.2e}' 
    fig4a, axs4a = plt.subplots(figsize=(fig3a_width, fig3a_height))
    ut.plot_dunn_test_matrix(fig4a, axs4a, dunn_results_C_h, plot_title_h, font_title, font_text, 
                             SMALL_SIZE, 0.74*MEDIUM_SIZE, 0.7*BIGGER_SIZE, my_str, vmin=global_vmin, vmax=global_vmax, ishealthy=True)
    
    my_str = 'K.-W. stat. = ' + "{:.1f}".format(kw_statistic_H_h) + f' $\cdot$ p value = {p_value_kw_H_h:.2e}' 
    fig4b, axs4b = plt.subplots(figsize=(fig3a_width, fig3a_height))
    ut.plot_dunn_test_matrix(fig4b, axs4b, dunn_results_H_h, plot_title_h, font_title, font_text, 
                             SMALL_SIZE, 0.74*MEDIUM_SIZE, 0.7*BIGGER_SIZE,  my_str, vmin=global_vmin, vmax=global_vmax, ishypo=True, ishealthy=True)
       
    ##### TUKEY INTERACTION FOR ART-ANOVA IS VALID IF ANY OF ART-ANOVA P-VALUES < 0.01
    # Some p-values in Tukey are so close to zero, the function cannot return them. These appear as white squares (with statistically signiticantly different black border)
    # Programmatically extract p-values from the ANOVA table
    p_ARTANOVA = ARTANOVA.loc[["C(Condition)", "C(Batch)", "C(Condition):C(Batch)"], "PR(>F)"]
    # Output the p-values as a numpy array
    p_ARTANOVA_array = p_ARTANOVA.values
    f_stat_array = ARTANOVA['F'].dropna().values
    
    p_ARTANOVA_h = ARTANOVA_h.loc[["C(Condition)", "C(Batch)", "C(Condition):C(Batch)"], "PR(>F)"]
    # Output the p-values as a numpy array
    p_ARTANOVA_array_h = p_ARTANOVA_h.values
    f_stat_array_h = ARTANOVA_h['F'].dropna().values
    
    if my_units[idx] != '':
        plot_title = f'i) Day {my_day} $\cdot$ ' + tests[idx] + f' ({my_units[idx]}) $\cdot$ all states'
    else:
        plot_title = f'i) Day {my_day} $\cdot$ ' + tests[idx] + ' $\cdot$ all states'
        
    if my_units[idx] != '':
        plot_title_h = f'j) Day {my_day} $\cdot$ ' + tests[idx] + f' ({my_units[idx]}) $\cdot$ only healthy'
    else:
        plot_title_h = f'j) Day {my_day} $\cdot$ ' + tests[idx] + ' $\cdot$ only healthy'
    
    vmin, vmax = ut.get_global_min_max(tukey_interaction_ARTANOVA, tukey_interaction_ARTANOVA_h)

    my_str = ('ART-A. stat. cond. = '       + "{:.1f}".format(f_stat_array[0]) + f' $\cdot$ p value = {p_ARTANOVA_array[0]:.2e}\n' 
            + 'ART-A. stat. batch = '       + "{:.1f}".format(f_stat_array[1]) + f' $\cdot$ p value = {p_ARTANOVA_array[1]:.2e}\n'
            + 'ART-A. stat. cond.:batch = ' + "{:.1f}".format(f_stat_array[2]) + f' $\cdot$ p value = {p_ARTANOVA_array[2]:.2e}')
    fig5a, axs5a = plt.subplots(figsize=(6, 4.3))
    fig5b, axs5b = plt.subplots(figsize=(6, 4.3))
    ut.plot_tukey_ARTanova(fig5a, axs5a, fig5b, axs5b, tukey_interaction_ARTANOVA, plot_title, font_title, font_text, 
                            SMALL_SIZE, 0.74*MEDIUM_SIZE, 0.7*BIGGER_SIZE, my_str, vmin, vmax,numbatch=len(batches_to_analyze))

    my_str = ('ART-A. stat. cond. = '       + "{:.1f}".format(f_stat_array_h[0]) + f' $\cdot$ p value = {p_ARTANOVA_array_h[0]:.2e}\n' 
            + 'ART-A. stat. batch = '       + "{:.1f}".format(f_stat_array_h[1]) + f' $\cdot$ p value = {p_ARTANOVA_array_h[1]:.2e}\n'
            + 'ART-A. stat. cond.:batch = ' + "{:.1f}".format(f_stat_array_h[2]) + f' $\cdot$ p value = {p_ARTANOVA_array_h[2]:.2e}')
    fig6a, axs6a = plt.subplots(figsize=(6, 4.3))
    fig6b, axs6b = plt.subplots(figsize=(6, 4.3))
    ut.plot_tukey_ARTanova(fig6a, axs6a, fig6b, axs6b, tukey_interaction_ARTANOVA_h, plot_title_h, font_title, font_text, 
                            SMALL_SIZE, 0.74*MEDIUM_SIZE, 0.7*BIGGER_SIZE, my_str, vmin, vmax,numbatch=len(batches_to_analyze), ishealthy=True)
    
    # Batch statistical comparison
    for my_first_ind in range(len(concatenated)):
        for my_second_ind in range(my_first_ind + 1, len(concatenated)):
            save_stats_path_batch = base_path + day_path + f'Results/BatchStats/{tests[idx]}_{conditionlabels[my_first_ind]}_{conditionlabels[my_second_ind]}_Comparison.txt'
            ut.perform_pairwise_tests([concatenated[my_first_ind], concatenated[my_second_ind]], [conditionlabels[my_first_ind], conditionlabels[my_second_ind]], save_stats_path_batch, save=True)
          
    for my_first_ind in range(len(healthy_concatenated)):
        for my_second_ind in range(my_first_ind + 1, len(healthy_concatenated)):
            save_stats_path_batch_healthy = base_path + day_path + f'Results/BatchStats/{tests[idx]}_{conditionlabels[my_first_ind]}_{conditionlabels[my_second_ind]}_Comparison_HEALTHY.txt'
            ut.perform_pairwise_tests([healthy_concatenated[my_first_ind], healthy_concatenated[my_second_ind]], [conditionlabels[my_first_ind], conditionlabels[my_second_ind]], save_stats_path_batch_healthy, save=True)
    
    my_mean_avg_control = []
    my_err_control      = []
    my_mean_avg_control_healthy = []
    my_err_control_healthy      = []
    
    my_mean_avg_hypo    = []
    my_err_hypo         = []
    my_mean_avg_hypo_healthy    = []
    my_err_hypo_healthy         = []
    
    ### All states
    for my_ind in range(0, len(concatenated), 2):
        # Get the pair of concatenated elements
        C_array = concatenated[my_ind]
        H_array = concatenated[my_ind + 1]
    
        mean_control, min_control, max_control, error_control = ut.confidence_interval(C_array)
        mean_hypo,    min_hypo,    max_hypo,    error_hypo = ut.confidence_interval(H_array)
        my_mean_avg_control.append(mean_control)
        my_err_control.append(error_control)
        my_mean_avg_hypo.append(mean_hypo)
        my_err_hypo.append(error_hypo)
    
    cohen_d_values, visibility_values, Eorig_values, p_values, cliffs_d_values = ut.process_all_pairs(concatenated)
    
    N_values = []
    # Iterate through the list in pairs
    for i in range(0, len(concatenated), 2):
        # Sum the lengths of consecutive pairs (concatenated[i] and concatenated[i + 1])
        length_sum = len(concatenated[i]) + len(concatenated[i + 1])
        N_values.append(length_sum)
        
    ### Healthy
    for my_ind in range(0, len(healthy_concatenated), 2):
        # Get the pair of concatenated elements
        C_array = healthy_concatenated[my_ind]
        H_array = healthy_concatenated[my_ind + 1]
    
        mean_control, min_control, max_control, error_control = ut.confidence_interval(C_array)
        mean_hypo,    min_hypo,    max_hypo,    error_hypo = ut.confidence_interval(H_array)
        my_mean_avg_control_healthy.append(mean_control)
        my_err_control_healthy.append(error_control)
        my_mean_avg_hypo_healthy.append(mean_hypo)
        my_err_hypo_healthy.append(error_hypo)
    
    cohen_d_values_healthy, visibility_values_healthy, Eorig_values_healthy, p_values_healthy, cliffs_d_values_healthy = ut.process_all_pairs(healthy_concatenated)
    
    N_values_healthy = []
    # Iterate through the list in pairs
    for i in range(0, len(healthy_concatenated), 2):
        # Sum the lengths of consecutive pairs (concatenated[i] and concatenated[i + 1])
        length_sum = len(healthy_concatenated[i]) + len(healthy_concatenated[i + 1])
        N_values_healthy.append(length_sum)
        
    # Average effect value for experiment conditions only, discarding positive controls -- note that even_ and odd_ arrays already discard positive controls
    flattened_even = np.concatenate([arr.flatten() for arr in even_arrays])
    flattened_odd = np.concatenate([arr.flatten() for arr in odd_arrays])
    flattened_even_pc = np.concatenate([arr.flatten() for arr in even_arrays_pc])
    flattened_odd_pc = np.concatenate([arr.flatten() for arr in odd_arrays_pc])
    
    onept_mean_control, onept_min_control, onept_max_control, onept_error_control = ut.confidence_interval(flattened_even)
    onept_mean_hypo, onept_min_hypo, onept_max_hypo, onept_error_hypo = ut.confidence_interval(flattened_odd)
    onept_mean_control_pc, onept_min_control_pc, onept_max_control_pc, onept_error_control_pc = ut.confidence_interval(flattened_even_pc)
    onept_mean_hypo_pc, onept_min_hypo_pc, onept_max_hypo_pc, onept_error_hypo_pc = ut.confidence_interval(flattened_odd_pc)
    
    healthy_flattened_even = np.concatenate([arr.flatten() for arr in healthy_even_arrays])
    healthy_flattened_odd = np.concatenate([arr.flatten() for arr in healthy_odd_arrays])
    healthy_flattened_even_pc = np.concatenate([arr.flatten() for arr in healthy_even_arrays_pc])
    healthy_flattened_odd_pc = np.concatenate([arr.flatten() for arr in healthy_odd_arrays_pc])
    
    onept_mean_control_h, onept_min_control_h, onept_max_control_H, onept_error_control_h = ut.confidence_interval(healthy_flattened_even)
    onept_mean_hypo_h, onept_min_hypo_h, onept_max_hypo_H, onept_error_hypo_h = ut.confidence_interval(healthy_flattened_odd)
    onept_mean_control_h_pc, onept_min_control_h_pc, onept_max_control_h_pc, onept_error_control_h_pc = ut.confidence_interval(healthy_flattened_even_pc)
    onept_mean_hypo_h_pc, onept_min_hypo_h_pc, onept_max_hypo_h_pc, onept_error_hypo_h_pc = ut.confidence_interval(healthy_flattened_odd_pc)
    
    cohen_d_avg, visibility_avg, Eorig_avg, p_value_avg, cliffs_avg = ut.stats_and_plot(flattened_even, flattened_odd)
    cohen_d_avg_healthy, visibility_avg_healthy, Eorig_avg_healthy, p_value_avg_healthy, cliffs_avg_healthy = ut.stats_and_plot(healthy_flattened_even, healthy_flattened_odd)
    cohen_d_avg_pc, visibility_avg_pc, Eorig_avg_pc, p_value_avg_pc, cliffs_avg_pc = ut.stats_and_plot(flattened_even_pc, flattened_odd_pc)
    cohen_d_avg_healthy_pc, visibility_avg_healthy_pc, Eorig_avg_healthy_pc, p_value_avg_healthy_pc, cliffs_avg_healthy_pc = ut.stats_and_plot(healthy_flattened_even_pc, healthy_flattened_odd_pc)
        
    ### Split arrays
    cohen_d_array,    cohen_d_array_pc    = ut.split_arrays(cohen_d_values, PositiveControls)
    visibility_array, visibility_array_pc = ut.split_arrays(visibility_values, PositiveControls)
    Eorig_array,      Eorig_array_pc      = ut.split_arrays(Eorig_values, PositiveControls)
    p_value_array,    p_value_array_pc    = ut.split_arrays(p_values, PositiveControls)
    cliffs_d,         cliffs_d_pc         = ut.split_arrays(cliffs_d_values, PositiveControls)
    n_array,          n_array_pc          = ut.split_arrays(N_values, PositiveControls)
    mean_avg_control, mean_avg_control_pc = ut.split_arrays(my_mean_avg_control, PositiveControls)
    err_control,      err_control_pc      = ut.split_arrays(my_err_control, PositiveControls)
    mean_avg_hypo ,   mean_avg_hypo_pc    = ut.split_arrays(my_mean_avg_hypo , PositiveControls)
    err_hypo,         err_hypo_pc         = ut.split_arrays(my_err_hypo, PositiveControls)
    
    
    cohen_d_array_healthy,    cohen_d_array_healthy_pc    = ut.split_arrays(cohen_d_values_healthy, PositiveControls)
    visibility_array_healthy, visibility_array_healthy_pc = ut.split_arrays(visibility_values_healthy, PositiveControls)
    Eorig_array_healthy,      Eorig_array_healthy_pc      = ut.split_arrays(Eorig_values_healthy, PositiveControls)
    p_value_array_healthy,    p_value_array_healthy_pc    = ut.split_arrays(p_values_healthy, PositiveControls)
    cliffs_d_healthy,         cliffs_d_healthy_pc         = ut.split_arrays(cliffs_d_values_healthy, PositiveControls)
    n_array_healthy,          n_array_healthy_pc          = ut.split_arrays(N_values_healthy, PositiveControls)
    mean_avg_control_healthy, mean_avg_control_healthy_pc = ut.split_arrays(my_mean_avg_control_healthy, PositiveControls)
    err_control_healthy,      err_control_healthy_pc      = ut.split_arrays(my_err_control_healthy, PositiveControls)
    mean_avg_hypo_healthy,    mean_avg_hypo_healthy_pc    = ut.split_arrays(my_mean_avg_hypo_healthy, PositiveControls)
    err_hypo_healthy,         err_hypo_healthy_pc         = ut.split_arrays(my_err_hypo_healthy, PositiveControls)
    
    
    # Determine the y-limits based on both plots
    all_means = np.concatenate([mean_avg_control, mean_avg_hypo, mean_avg_control_healthy, mean_avg_hypo_healthy])
    all_cis = np.concatenate([err_control, err_hypo, err_control_healthy, err_hypo_healthy])

    # Calculate ylims with some 5% padding
    y_min = 0.95* min(all_means - all_cis) #was 0.95
    y_max = 1.05* max(all_means + all_cis) #was 1.05
    
    # Adjust batches_to_analyze based on PositiveControls
    if PositiveControls > 0:
        batches_labels = batches_to_analyze[:-PositiveControls]
    else:
        batches_labels = batches_to_analyze
        
    if my_units[idx] != '':
        plot_title = f'c) Day {my_day} $\cdot$ ' + tests[idx] + f' ({my_units[idx]}) $\cdot$ all states\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0'
        plot_title_h = f'c) Day {my_day} $\cdot$ ' + tests[idx] + f' ({my_units[idx]}) $\cdot$ only healthy'
    else:
        plot_title = f'c) Day {my_day} $\cdot$ ' + tests[idx] + ' $\cdot$ all states\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0'
        plot_title_h = f'c) Day {my_day} $\cdot$ ' + tests[idx] + ' $\cdot$ only healthy'
    
    fig7a, axs7a = plt.subplots()
    ut.plot_barplot(fig7a, axs7a, f'{tests[idx]}', 
                mean_avg_control, mean_avg_hypo, 
                err_control, err_hypo,
                
                batches_labels, plot_title, font_title, font_text, 
                SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, 
                
                onept_mean_control, onept_mean_hypo,
                onept_error_control, onept_error_hypo, 
                
                mean_avg_control_pc, mean_avg_hypo_pc,
                err_control_pc, err_hypo_pc,
                
                onept_mean_control_pc, onept_mean_hypo_pc,
                onept_error_control_pc, onept_error_hypo_pc, 
                
                n_array, n_array_pc,
                y_min=y_min, y_max=y_max)

    fig7b, axs7b = plt.subplots()
    ut.plot_barplot(fig7b, axs7b,f'{tests[idx]}', 
                mean_avg_control_healthy, mean_avg_hypo_healthy, 
                err_control_healthy, err_hypo_healthy,
                
                batches_labels, plot_title_h, font_title, font_text, 
                SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, 
                
                onept_mean_control_h, onept_mean_hypo_h,
                onept_error_control_h, onept_error_hypo_h, 
                
                mean_avg_control_healthy_pc, mean_avg_hypo_healthy_pc,
                err_control_healthy_pc, err_hypo_healthy_pc,
                
                onept_mean_control_h_pc, onept_mean_hypo_h_pc,
                onept_error_control_h_pc, onept_error_hypo_h_pc, 
                
                n_array_healthy, n_array_healthy_pc,
                ishealthy=True, y_min=y_min, y_max=y_max)
    
    # Calculate max and min:
    my_ymax1 = max(np.max(cohen_d_array), np.max(cohen_d_array_healthy), np.max(cliffs_d), np.max(cliffs_d_healthy))
    my_ymin1 = 0
    
    my_ymax2 = max(np.max(p_values), np.max(p_values_healthy))
    my_ymin2 = min(np.min(p_values), np.min(p_values_healthy), np.min(p_value_avg), np.min(p_value_avg_healthy))
    
    my_ymax3 = max(np.max(visibility_array), np.max(Eorig_array), np.max(visibility_array_healthy), np.max(Eorig_array_healthy))
    my_ymin3 = 0
    
    # Define some padding (optional) to make the plots look better
    # normal scale padding
    y_min1 = 0
    padding = 0.1 * (my_ymax1 - my_ymin1)
    y_max1 = my_ymax1 + padding
    
    padding = 0.1 * (my_ymax3 - my_ymin1)
    y_max3 = my_ymax3 + padding
    
    # log scale padding
    padding_factor = 20.0  # Padding factor for log plot # was 5.0
    y_min2 = my_ymin2 / padding_factor
    y_max2 = my_ymax2 * padding_factor
    
    if my_units[idx] != '':
        plot_title = f'e) Day {my_day} $\cdot$ ' + tests[idx] + f' ({my_units[idx]}) $\cdot$ all states\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0'
        plot_title_h = f'e) Day {my_day} $\cdot$ ' + tests[idx] + f' ({my_units[idx]}) $\cdot$ only healthy'
    else:
        plot_title = f'e) Day {my_day} $\cdot$ ' + tests[idx] + ' $\cdot$ all states'
        plot_title_h = f'e) Day {my_day} $\cdot$ ' + tests[idx] + ' $\cdot$ only healthy'
        
    fig8a, (axs8a, axs8a2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ut.plot_stats1(fig8a, axs8a, axs8a2,
                      cohen_d_array, visibility_array, Eorig_array, p_value_array, cliffs_d,
                      batches_labels, 
                      plot_title, font_title, font_text, 
                      SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, 
                      y_min1, y_max1, y_min2, y_max2, 
                      cohen_d_avg, visibility_avg, Eorig_avg, p_value_avg, cliffs_avg, n_array,
                      cohen_d_array_pc, visibility_array_pc, Eorig_array_pc, p_value_array_pc, cliffs_d_pc, n_array_pc,
                      cohen_d_avg_pc, visibility_avg_pc, Eorig_avg_pc, p_value_avg_pc, cliffs_avg_pc)
    
    fig8b, (axs8b, axs8b2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ut.plot_stats1(fig8b, axs8b, axs8b2, 
                      cohen_d_array_healthy, visibility_array_healthy, Eorig_array_healthy, p_value_array_healthy, cliffs_d_healthy,
                      batches_labels, 
                      plot_title_h, font_title, font_text, 
                      SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, 
                      y_min1, y_max1, y_min2, y_max2, 
                      cohen_d_avg_healthy, visibility_avg_healthy, Eorig_avg_healthy, p_value_avg_healthy, cliffs_avg_healthy, n_array_healthy,
                      cohen_d_array_healthy_pc, visibility_array_healthy_pc, Eorig_array_healthy_pc, p_value_array_healthy_pc, cliffs_d_healthy_pc, n_array_healthy_pc,
                      cohen_d_avg_healthy_pc, visibility_avg_healthy_pc, Eorig_avg_healthy_pc, p_value_avg_healthy_pc, cliffs_avg_healthy_pc,
                      ishealthy=True)

    if my_units[idx] != '':
        plot_title = f'f) Day {my_day} $\cdot$ ' + tests[idx] + f' ({my_units[idx]}) $\cdot$ all states\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0'
        plot_title_h = f'f) Day {my_day} $\cdot$ ' + tests[idx] + f' ({my_units[idx]}) $\cdot$ only healthy'
    else:
        plot_title = f'f) Day {my_day} $\cdot$ ' + tests[idx] + ' $\cdot$ all states'
        plot_title_h = f'f) Day {my_day} $\cdot$ ' + tests[idx] + ' $\cdot$ only healthy'
    fig9a, (axs9a, axs9a2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ut.plot_stats2(fig9a, axs9a, axs9a2, 
                      cohen_d_array, visibility_array, Eorig_array, p_value_array, cliffs_d,
                      batches_labels, 
                      plot_title, font_title, font_text, 
                      SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, 
                      y_min1, y_max3, y_min2, y_max2, 
                      cohen_d_avg, visibility_avg, Eorig_avg, p_value_avg, cliffs_avg, n_array,
                      cohen_d_array_pc, visibility_array_pc, Eorig_array_pc, p_value_array_pc, cliffs_d_pc, n_array_pc,
                      cohen_d_avg_pc, visibility_avg_pc, Eorig_avg_pc, p_value_avg_pc, cliffs_avg_pc)
    
    fig9b, (axs9b, axs9ab) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ut.plot_stats2(fig9b, axs9b, axs9ab, 
                      cohen_d_array_healthy, visibility_array_healthy, Eorig_array_healthy, p_value_array_healthy, cliffs_d_healthy,
                      batches_labels, 
                      plot_title_h, font_title, font_text, 
                      SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, 
                      y_min1, y_max3, y_min2, y_max2, 
                      cohen_d_avg_healthy, visibility_avg_healthy, Eorig_avg_healthy, p_value_avg_healthy, cliffs_avg_healthy, n_array_healthy,
                      cohen_d_array_healthy_pc, visibility_array_healthy_pc, Eorig_array_healthy_pc, p_value_array_healthy_pc, cliffs_d_healthy_pc, n_array_healthy_pc,
                      cohen_d_avg_healthy_pc, visibility_avg_healthy_pc, Eorig_avg_healthy_pc, p_value_avg_healthy_pc, cliffs_avg_healthy_pc,
                      ishealthy=True)

    even = [concatenated[i] for i in range(0, len(concatenated), 2)]
    odd = [concatenated[i] for i in range(1, len(concatenated), 2)]
    even_arrays, even_arrays_pc = ut.split_arrays(even, PositiveControls)
    odd_arrays,  odd_arrays_pc  = ut.split_arrays(odd, PositiveControls)
    
    #Flatten and concatenate avg of experiment conditions
    even_flattened = np.concatenate(even_arrays)
    odd_flattened = np.concatenate(odd_arrays)
    even_arrays.append(even_flattened)
    odd_arrays.append(odd_flattened)

    # Use extend to add the positive controls to the existing lists
    even_arrays.extend(even_arrays_pc)
    odd_arrays.extend(odd_arrays_pc)
    
    # Add positive control flattened array -- new 4 lines
    even_flattened_pc = np.concatenate(even_arrays_pc)
    odd_flattened_pc = np.concatenate(odd_arrays_pc)
    even_arrays.append(even_flattened_pc)
    odd_arrays.append(odd_flattened_pc)
    
    even_h = [healthy_concatenated[i] for i in range(0, len(healthy_concatenated), 2)]
    odd_h = [healthy_concatenated[i] for i in range(1, len(healthy_concatenated), 2)]
    even_arrays_h, even_arrays_h_pc = ut.split_arrays(even_h, PositiveControls)
    odd_arrays_h,  odd_arrays_h_pc  = ut.split_arrays(odd_h, PositiveControls)
    
    #Flatten and concatenate avg of experiment conditions
    even_flattened_h = np.concatenate(even_arrays_h)
    odd_flattened_h = np.concatenate(odd_arrays_h)
    even_arrays_h.append(even_flattened_h)
    odd_arrays_h.append(odd_flattened_h)

    # Use extend to add the positive controls to the existing lists
    even_arrays_h.extend(even_arrays_h_pc)
    odd_arrays_h.extend(odd_arrays_h_pc)
    
    # Add positive control flattened array -- new 4 lines
    even_flattened_healthy_pc = np.concatenate(even_arrays_h_pc)
    odd_flattened_healthy_pc = np.concatenate(odd_arrays_h_pc)
    even_arrays_h.append(even_flattened_healthy_pc)
    odd_arrays_h.append(odd_flattened_healthy_pc)
    
    if my_units[idx] != '':
        plot_title = f'd) Day {my_day} $\cdot$ ' + tests[idx] + f' ({my_units[idx]}) $\cdot$ all states\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0'
        plot_title_h = f'd) Day {my_day} $\cdot$ ' + tests[idx] + f' ({my_units[idx]}) $\cdot$ only healthy'
    else:
        plot_title = f'd) Day {my_day} $\cdot$ ' + tests[idx] + ' $\cdot$ all states\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0'
        plot_title_h = f'd) Day {my_day} $\cdot$ ' + tests[idx] + ' $\cdot$ only healthy'
    
    # Bootstrapping
    nbootstrap = 1000
    
    # Combine all the bootstrap differences for min and max calculation
    all_bootstrapped_diffs = []

    for control_data, condition_data in [(even_arrays, odd_arrays), (even_arrays_h, odd_arrays_h)]:
        for control, condition in zip(control_data, condition_data):
            bootstrapped_diffs = []
            for _ in range(nbootstrap):
                resampled_control = resample(control)
                resampled_condition = resample(condition)
                # Calculate percentage difference from control
                median_diff = (np.median(resampled_condition) - np.median(resampled_control)) / np.median(resampled_control) * 100
                mean_diff = (np.mean(resampled_condition) - np.mean(resampled_control)) / np.mean(resampled_control) * 100
                bootstrapped_diffs.extend([median_diff, mean_diff])
            all_bootstrapped_diffs.extend(bootstrapped_diffs)

    # Determine global min and max for y-axis
    # give some "reducing" padding -- given that there are outliers, these limits are overestimated
    y_min = 1.0* np.min(all_bootstrapped_diffs) # was 1.0
    y_max = 0.8* np.max(all_bootstrapped_diffs) # was 0.8

    fig10a, axs10a = plt.subplots()
    ut.bootstrap_median(fig10a, axs10a, even_arrays, odd_arrays, nbootstrap,
                                      batches_labels, plot_title, font_title, font_text, 
                                      SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE,
                                      n_array, n_array_pc,
                                      y_min, y_max)
    
    fig10b, axs10b = plt.subplots()
    ut.bootstrap_median(fig10b, axs10b, even_arrays_h, odd_arrays_h, nbootstrap,
                                      batches_labels, plot_title_h, font_title, font_text, 
                                      SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE,
                                      n_array_healthy, n_array_healthy_pc,
                                      y_min, y_max, ishealthy=True)
    
###############################################################################

    output_path = base_path + day_path + f'Results/D{my_day}_{tests[idx]}/'

    if save:        
        fig_numbers = plt.get_fignums()
    
        fig1a.savefig(output_path + 'Fig1a.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
    
        fig1b.savefig(output_path + 'Fig1b.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
    
        fig2a.savefig(output_path + 'Fig2a.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
    
        fig2b.savefig(output_path + 'Fig2b.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
        
        fig3a.savefig(output_path + 'Fig3a.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
    
        fig3b.savefig(output_path + 'Fig3b.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
    
        fig4a.savefig(output_path + 'Fig4a.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
    
        fig4b.savefig(output_path + 'Fig4b.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
    
        fig5a.savefig(output_path + 'Fig5a.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
    
        fig5b.savefig(output_path + 'Fig5b.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
    
        fig6a.savefig(output_path + 'Fig6a.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
    
        fig6b.savefig(output_path + 'Fig6b.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
    
        fig7a.savefig(output_path + 'Fig7a.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
    
        fig7b.savefig(output_path + 'Fig7b.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
    
        fig8a.savefig(output_path + 'Fig8a.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
    
        fig8b.savefig(output_path + 'Fig8b.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF

        fig9a.savefig(output_path + 'Fig9a.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
    
        fig9b.savefig(output_path + 'Fig9b.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF

        fig10a.savefig(output_path + 'Fig10a.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
    
        fig10b.savefig(output_path + 'Fig10b.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
        
        plt.close('all')
