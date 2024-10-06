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
import pandas as pd
from scipy.stats import chi2_contingency

###############################################################################
# CHANGE HERE ONLY
###############################################################################

save = True
batches_to_analyze = [1,2,3,4,5,6,7,8,9,10]

PositiveControls = 3 # Last 3 batches are positive control

###############################################################################
###############################################################################

day_path = '2 Experimental overview/'
    
tests = ['Assessed status']
miny = [0]
maxy = [5.5] 
yticks = [[1,2,3,4]]    
my_lines =  []
my_labels = [''] 
my_units = ['']

miny_avg = miny
maxy_avg = maxy

###############################################################################
###############################################################################
# Number of plates per batch; note: index 0 corresponds to batch 1, etc
no_plates = [4, 5, 5, 1, 2, 6, 4, 6, 1, 10]

# Initialize concatenated with empty lists for each batch and condition
concatenated = [[] for _ in range(2 * len(batches_to_analyze))]

for idx in range(len(tests)):
    
    hor_lines  = [] #my_lines[idx]
    hor_labels = [] #my_labels[idx]

    common_filenames = [[], []]
    filtered_status_d3 = [[], []]
    stickcolors = [[], []]

    my_fil = []

    for my_batch in batches_to_analyze: 

        # Read the assessment files, for all plates in the batches
        for my_ind in range(2):
            
            assess_filenames = [[], []]
            d3_status = [[],[]]
            
            for my_plate in range(1, no_plates[my_batch - 1] + 1):
                assess_path = [f"/Users/clarice/Desktop/2 Experimental overview/Assessments/B{my_batch}/B{my_batch}{'C' if my_idx == 0 else 'H'}P{my_plate}_assessment.txt" for my_idx in range(2)]
                c_content = ut.read_file(assess_path[my_ind])
                assess_filenames[my_ind].extend(ut.extract_list(c_content, 'Filenames:'))  # Filenames separate plate assessment txt files
                d3_status[my_ind].extend(ut.extract_list(c_content, 'StatusD3:'))

            # Store d3_status in concatenated
            concatenated[2*(my_batch-1) + my_ind] = ut.replace_D_with_4(d3_status[my_ind])
           
            # Find common filenames
            common_filenames[my_ind]= assess_filenames[my_ind] ##set(modified_filenames[my_ind]).intersection(set(filenames[my_ind]))
        
            # Filter status_d3 based on common filenames
            filtered_status_d3[my_ind] = d3_status[my_ind] #[d3_status[my_ind][modified_filenames[my_ind].index(fname)] for fname in common_filenames[my_ind]]
                    # if there's a mismatch, debug:
                        # print(f"Common filenames: {len(common_filenames[my_ind])}, "
                        #       f"Filtered Status D3: {len(filtered_status_d3[my_ind])}")

            # Map status to colors and save to framecolors
            stickcolors[my_ind].append(ut.map_status_to_color(filtered_status_d3[my_ind]))
            
            # Debugging information
            # print(f"Batch {my_batch}, condition {my_ind} ")            
            # print(f"Length of common_filenames[{my_ind}]: {len(common_filenames[my_ind])}")
            # print(f"Length of concatenated[{2*(my_batch-1) + my_ind}]: {len(concatenated[2*(my_batch-1) + my_ind])}")
            # print(f"Length of filtered_status_d3[{my_ind}]: {len(filtered_status_d3[my_ind])}")
            
                
    # Calculate the sum of the lengths of the even-indexed elements in concatenated
    sum_even_concatenated = sum(len(concatenated[i]) for i in range(0, len(concatenated), 2))
    # Calculate the sum of the lengths of the odd-indexed elements in concatenated
    sum_odd_concatenated = sum(len(concatenated[i]) for i in range(1, len(concatenated), 2))
    # Calculate the sum of the stickcolors
    sum_even_stickcolors = sum(len(stickcolors[0][i]) for i in range(len(stickcolors[0])))
    sum_odd_stickcolors = sum(len(stickcolors[1][i]) for i in range(len(stickcolors[1])))
    # Compare and raise an error if they do not match
    
    if sum_even_concatenated != sum_even_stickcolors:
        raise ValueError(f"Mismatch: sum of even concatenated lengths ({sum_even_concatenated}) != sum of even stickcolors lengths ({sum_even_stickcolors})")
    if sum_odd_concatenated != sum_odd_stickcolors:
        raise ValueError(f"Mismatch: sum of odd concatenated lengths ({sum_odd_concatenated}) != sum of odd stickcolors lengths ({sum_odd_stickcolors})")    

    ###########

    ### Individual violins
    xpos = list(range(2*len(batches_to_analyze))) # number of positions in x axis
    
    conditionlabels = ['B1C','B1H',
                       'B2C','B2H',
                       'B3C','B3H',
                       'B4C','B4H',
                       'B5C','B5H',
                       'B6C','B6H',
                       'B7C','B7H',
                       '+1C','+1H',
                       '+2C','+2H',
                       '+3C','+3H']
    
    fig1a, axs1a = plt.subplots()
    plt.suptitle(tests[idx] + ' by batch', fontsize=BIGGER_SIZE, fontproperties=font_title)  
    ut.plot_violins_bio(fig1a, 
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

    ### Average violin
    fig2a, axs2a = plt.subplots(figsize=(fig1a_width, fig1a_height))
    plt.suptitle(tests[idx], fontsize=BIGGER_SIZE, fontproperties=font_title)
    
    ### Note that even_ and odd_ arrays only have experiment conditions, no positive controls!!!
    even_arrays, odd_arrays, stickcolors, even_arrays_pc, odd_arrays_pc, stickcolors_pc = ut.process_arrays(concatenated, stickcolors, PositiveControls)

    labels = ut.generate_tuple(concatenated, PositiveControls)
    ut.plot_avg_violin_bio(fig2a, 
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
    
    
        # Count occurrences of each state in both populations
    states = [1, 2, 3, 4]
    control_counts = [np.sum(even_arrays == state) for state in states]
    hypo_counts = [np.sum(odd_arrays == state) for state in states]
    
    # Create a contingency table
    contingency_table = pd.DataFrame({
        'Control': control_counts,
        'Hypo': hypo_counts
    }, index=states)
    
    print("Contingency Table:")
    print(contingency_table)
    # Perform Chi-Square Test
    chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table.T)
    
    print("\nChi-Square Test Results:")
    print(f"Chi-Square Statistic: {chi2_stat:.4f}")
    print(f"P-Value: {p_val:.4f}")
    print(f"Degrees of Freedom: {dof}")
    #print(f"Expected Frequencies Table:\n{pd.DataFrame(expected, index=states, columns=['Control', 'Hypo'])}")
    
    control_counts_pc = [np.sum(even_arrays_pc == state) for state in states]
    hypo_counts_pc = [np.sum(odd_arrays_pc == state) for state in states]
    
    # Create a contingency table
    contingency_table_pc = pd.DataFrame({
        'Control': control_counts_pc,
        'Hypo': hypo_counts_pc
    }, index=states)
    
    print("Contingency Table:")
    print(contingency_table_pc)
    # Perform Chi-Square Test
    chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table_pc.T)
    
    print("\nChi-Square Test Results:")
    print(f"Chi-Square Statistic: {chi2_stat:.4f}")
    print(f"P-Value: {p_val:.4f}")
    print(f"Degrees of Freedom: {dof}")
    #print(f"Expected Frequencies Table:\n{pd.DataFrame(expected, index=states, columns=['Control', 'Hypo'])}")
    
    batches_str = '_'.join(map(str, batches_to_analyze))
    save_avg_path = base_path + day_path + f'Results/{tests[idx]}/Batches{batches_str}_{tests[idx]}_Comparison.txt'
    save_avg_path_healthy = base_path + day_path + f'Results/{tests[idx]}/Batches{batches_str}_{tests[idx]}_Comparison_HEALTHY.txt'
    
    if PositiveControls != 0:
        my_str = batches_str[:-2*PositiveControls]
        
    dunn_results_C,   dunn_results_H,   kw_statistic_C,   p_value_kw_C,   kw_statistic_H,   p_value_kw_H,   ARTANOVA,   tukey_interaction_ARTANOVA  = ut.stats_and_save(save_avg_path,
                                                                                                                               concatenated, 
                                                                                                                               even_arrays, odd_arrays, 
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
    plot_title = tests[idx]  
   
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
    
       
    ##### TUKEY INTERACTION FOR ART-ANOVA IS VALID IF ANY OF ART-ANOVA P-VALUES < 0.01
    # Some p-values in Tukey are so close to zero, the function cannot return them. These appear as white squares (with statistically signiticantly different black border)
    # Programmatically extract p-values from the ANOVA table
    p_ARTANOVA = ARTANOVA.loc[["C(Condition)", "C(Batch)", "C(Condition):C(Batch)"], "PR(>F)"]
    # Output the p-values as a numpy array
    p_ARTANOVA_array = p_ARTANOVA.values
    f_stat_array = ARTANOVA['F'].dropna().values
    
    vmin, vmax = ut.get_global_min_max(tukey_interaction_ARTANOVA, tukey_interaction_ARTANOVA) #, tukey_interaction_ARTANOVA_h)

    my_str = ('ART-A. stat. cond. = '       + "{:.1f}".format(f_stat_array[0]) + f' $\cdot$ p value = {p_ARTANOVA_array[0]:.2e}\n' 
            + 'ART-A. stat. batch = '       + "{:.1f}".format(f_stat_array[1]) + f' $\cdot$ p value = {p_ARTANOVA_array[1]:.2e}\n'
            + 'ART-A. stat. cond.:batch = ' + "{:.1f}".format(f_stat_array[2]) + f' $\cdot$ p value = {p_ARTANOVA_array[2]:.2e}')
    
    fig5a, axs5a = plt.subplots(figsize=(6, 4.3))
    fig5b, axs5b = plt.subplots(figsize=(6, 4.3))
    ut.plot_tukey_ARTanova(fig5a, axs5a, fig5b, axs5b, tukey_interaction_ARTANOVA, plot_title, font_title, font_text, 
                            SMALL_SIZE, 0.74*MEDIUM_SIZE, 0.7*BIGGER_SIZE, my_str, vmin, vmax,numbatch=len(batches_to_analyze))
    
    # Batch statistical comparison
    for my_first_ind in range(len(concatenated)):
        for my_second_ind in range(my_first_ind + 1, len(concatenated)):
            save_stats_path_batch = base_path + day_path + f'Results/BatchStats/{tests[idx]}_{conditionlabels[my_first_ind]}_{conditionlabels[my_second_ind]}_Comparison.txt'
            ut.perform_pairwise_tests([concatenated[my_first_ind], concatenated[my_second_ind]], [conditionlabels[my_first_ind], conditionlabels[my_second_ind]], save_stats_path_batch, save=True)
        
###############################################################################

    output_path = base_path + day_path + f'Results/{tests[idx]}/'

    if save:        
        fig_numbers = plt.get_fignums()
    
        fig1a.savefig(output_path + 'Fig1a.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
    
        fig2a.savefig(output_path + 'Fig2a.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
        
        fig3a.savefig(output_path + 'Fig3a.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
    
        fig3b.savefig(output_path + 'Fig3b.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
    
        fig5a.savefig(output_path + 'Fig5a.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
    
        fig5b.savefig(output_path + 'Fig5b.pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)  # Save as PDF
        
        plt.close('all')
