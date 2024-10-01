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
    
    ### Note that even_ and odd_ arrays only have experiment conditions, no positive controls!!!
    even_arrays, odd_arrays, stickcolors, even_arrays_pc, odd_arrays_pc, stickcolors_pc = ut.process_arrays(concatenated, stickcolors, PositiveControls)
    healthy_even_arrays, healthy_odd_arrays, healthy_stickcolors, healthy_even_arrays_pc, healthy_odd_arrays_pc, healthy_stickcolors_pc = ut.process_arrays(healthy_concatenated, healthy_stickcolors, PositiveControls)

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
        plot_title = f'Day {my_day} $\cdot$ ' + tests[idx] + f' ({my_units[idx]}) $\cdot$ all states\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0'
    else:
        plot_title = f'Day {my_day} $\cdot$ ' + tests[idx] + ' $\cdot$ all states\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0'
    
    if my_units[idx] != '':
        plot_title_h = f'Day {my_day} $\cdot$ ' + tests[idx] + f' ({my_units[idx]}) $\cdot$ only healthy'
    else:
        plot_title_h = f'Day {my_day} $\cdot$ ' + tests[idx] + ' $\cdot$ only healthy'
   
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
    
    vmin, vmax = ut.get_global_min_max(tukey_interaction_ARTANOVA, tukey_interaction_ARTANOVA_h)

    my_str = ('ART-A. stat. cond. = '       + "{:.1f}".format(f_stat_array[0]) + f' $\cdot$ p value = {p_ARTANOVA_array[0]:.2e}\n' 
            + 'ART-A. stat. batch = '       + "{:.1f}".format(f_stat_array[1]) + f' $\cdot$ p value = {p_ARTANOVA_array[1]:.2e}\n'
            + 'ART-A. stat. cond.:batch = ' + "{:.1f}".format(f_stat_array[2]) + f' $\cdot$ p value = {p_ARTANOVA_array[2]:.2e}')
    fig5c, axs5c = plt.subplots(figsize=(10, 8))
    ut.plot_tukey_ARTanova_BM(fig5c, axs5c, tukey_interaction_ARTANOVA, plot_title, font_title, font_text, 
                            SMALL_SIZE, 0.74*MEDIUM_SIZE, 0.7*BIGGER_SIZE, my_str, vmin, vmax,numbatch=len(batches_to_analyze))

    my_str = ('ART-A. stat. cond. = '       + "{:.1f}".format(f_stat_array_h[0]) + f' $\cdot$ p value = {p_ARTANOVA_array_h[0]:.2e}\n' 
            + 'ART-A. stat. batch = '       + "{:.1f}".format(f_stat_array_h[1]) + f' $\cdot$ p value = {p_ARTANOVA_array_h[1]:.2e}\n'
            + 'ART-A. stat. cond.:batch = ' + "{:.1f}".format(f_stat_array_h[2]) + f' $\cdot$ p value = {p_ARTANOVA_array_h[2]:.2e}')
    fig6c, axs6c = plt.subplots(figsize=(10, 8))
    ut.plot_tukey_ARTanova_BM(fig6c, axs6c, tukey_interaction_ARTANOVA_h, plot_title_h, font_title, font_text, 
                            SMALL_SIZE, 0.74*MEDIUM_SIZE, 0.7*BIGGER_SIZE, my_str, vmin, vmax,numbatch=len(batches_to_analyze), ishealthy=True)
    
###############################################################################

    output_path = base_path + day_path + f'Results/BigMatrix/D{my_day}_{tests[idx]}_'

    if save:        
        fig_numbers = plt.get_fignums()
        
        fig5c.savefig(output_path + 'Fig5cBM.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
    
        fig6c.savefig(output_path + 'Fig6cBM.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
        
        plt.close('all')
