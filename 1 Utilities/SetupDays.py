###############################################################################
# BASE PATH
base_path = '/Users/clarice/Desktop/'
###############################################################################

import sys
sys.path.append(base_path + '1 Utilities/')
import utilities as ut

def Setup_D1():
    
    # Initialize empty lists for tests, variables, etc.
    tests = []
    variables = []
    miny = []
    maxy = []
    yticks = []
    my_lines = []
    my_labels = []
    my_units = []
    
    # Function to add a test and corresponding parameters
    def add_test(test_name, variable_name, min_val, max_val, y_tick_vals, line_vals, label_val, label_unit):
        tests.append(test_name)
        variables.append(variable_name)
        miny.append(min_val)
        maxy.append(max_val)
        yticks.append(y_tick_vals)
        my_lines.append(line_vals)
        my_labels.append(label_val)
        my_units.append(label_unit)
        
    directory_path = base_path + "1 Utilities/Test images/Results/"
    initial_stage = 20
    end_stage = 24
    
    elongs = ut.extract_mean_quantity(directory_path, "Elongations:", initial_stage, end_stage) # Elongation
    roundns = ut.extract_mean_quantity(directory_path, "Roundnesses:", initial_stage, end_stage)      # Roundness
    eccs = ut.extract_mean_quantity(directory_path, "Eccentricities:", initial_stage, end_stage)     # Eccentricity
    ratioAP = ut.extract_mean_quantity(directory_path, "area/perimeter:", initial_stage, end_stage)     # Area/Per
    BB = ut.extract_mean_quantity(directory_path, "BBARs:", initial_stage, end_stage)     # BB
    convs = ut.extract_mean_quantity(directory_path, "Convexities:", initial_stage, end_stage)     # BB
    radii = ut.extract_mean_quantity(directory_path, "Radius of curvatures:", initial_stage, end_stage)     # Radii curvature
    curvs = ut.extract_mean_quantity(directory_path, "Curvatures:", initial_stage, end_stage)     # Curvs
    RMScurvs = ut.extract_mean_quantity(directory_path, "RMS curvatures:", initial_stage, end_stage)     # RMScurvs
    Normcurvs = ut.extract_mean_quantity(directory_path, "Norm curvatures:", initial_stage, end_stage)     # Normcurvs
    
    my_stages = [f'stage {i}' for i in range(initial_stage, end_stage + 1)]
    
    sols = ut.extract_mean_quantity(directory_path, "Solidities:", initial_stage, end_stage)     # Solidity
    my_sols = [sols[0], sols[2], sols[3]]
    my_sols_labels = ['stage 20', 'stages 21, 22', 'stage 23']

    #Check if they increase/decrease as expected
    if not ut.is_increasing(elongs): # Elongation should be increasing
        raise ValueError("Elongation is not strictly increasing!")
    if not ut.is_decreasing(roundns): # Roundness should be decreasing
        raise ValueError("Roundness is not strictly decreasing!")
    if not ut.is_increasing(eccs): # Eccentricity should be increasing
        raise ValueError("Eccentricity is not strictly increasing!")   
    if not ut.is_decreasing(sols): # Roundness should be decreasing
        raise ValueError("Solidity is not strictly decreasing!")
    if not ut.is_increasing(BB): # BB should be increasing
        raise ValueError("BB is not strictly increasing!")   
    # if not ut.is_increasing(curvs): # Curvatures should be increasing
    #     print(curvs)
    #     raise ValueError("Curvatures is not strictly increasing!")   
    # if not ut.is_increasing(RMScurvs): # Curvatures should be increasing
    #     print(RMScurvs)
    #     raise ValueError("RMS curvatures is not strictly increasing!") 
    # if not ut.is_increasing(Normcurvs): # Curvatures should be increasing
    #     print(Normcurvs)
    #     raise ValueError("Norm. curvatures is not strictly increasing!") 
        
    ###########################################################################
    ###########################################################################
    #Fig 2:
        
    #In text:
    #add_test('eccentricity', 'Eccentricities:', 0.25, 2.25, [0.75, 1.0, 1.25], [1.0], '','norm.')
    #add_test('elongation', 'Elongations:',          0, 3, [1.0, 1.5, 2], [1.0], '','norm.')
    #add_test('roundness', 'Roundnesses:', 0.85, 1.1, [0.9, 0.95, 1.00, 1.05], [1.0], '','norm.')  
    #add_test('solidity', 'Solidities:', 0.95, 1.025, [0.975, 1.0], [1.0], '','norm.')
    #add_test('area', 'Areas:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '', 'norm.')
    #add_test('perimeter', 'Perimeters:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '', 'norm.')
    
    
    #add_test('eccentricity', 'Eccentricities:', 0, 1, [0.3, 0.5, 0.7], eccs, my_stages,'')
    #add_test('elongation', 'Elongations:', 0, 0.4, [0.1, 0.2, 0.3], elongs, my_stages,'')
    #add_test('roundness', 'Roundnesses:', 0.7, 1, [0.75, 0.85, 0.95], roundns, my_stages,'')
    #add_test('solidity', 'Solidities:', 0.95, 1.02, [0.96, 0.97, 0.98, 0.99, 1.00], my_sols, my_sols_labels,'')
    
    add_test('major axis', 'Major axes:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '', 'norm.')
    
    ###########################################################################
    ###########################################################################
    
    # Binarization is bad; no test applied on binary test (color, pigmentation) makes sense
    
    #Not relevant:
    # add_test('major axis', 'Major axes:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '', 'norm.')
    # add_test('major axis', 'Major axes:', 500, 1750, [750, 1000, 1250, 1500], [], '', 'px.')
    # add_test('area', 'Areas:', 0, 250000, [50000, 100000, 150000, 200000], [], '', 'px.$^2$')
    # add_test('perimeter', 'Perimeters:', 1000, 5000, [2000, 3000, 4000], [1.0], '', 'px.')
    # add_test('ratio area to perimeter', 'area/perimeter:', 40, 110, [50, 75, 100], ratioAP, my_stages, 'px.')
    # add_test('ratio area to perimeter', 'area/perimeter:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '', 'norm.')
    # add_test('BBAR', 'BBARs:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '','norm.')
    # add_test('BBAR', 'BBARs:', 2, 6.0, [3, 4, 5], BB, my_stages,'')
    # add_test('convexity', 'Convexities:', 0.9, 1.1, [0.95, 1.0, 1.05], [1.0], '', 'norm.')
    # add_test('convexity', 'Convexities:', 0.85, 1.05, [0.9, 0.95, 1.0], convs, my_stages, '')
    # add_test('radii of curvatures', 'Radius of curvatures:', 0.95, 1.05, [0.975, 1.0, 1.025], [1.0], '', 'norm.')
    # add_test('radii of curvatures', 'Radius of curvatures:', 1.3, 1.6, [1.45, 1.50, 1.55], radii, my_stages, 'px.')
    # add_test('curvature', 'Curvatures:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '', 'norm.')
    # add_test('curvature', 'Curvatures:', 400, 1100, [500, 700, 900, 1000], curvs, my_stages, 'px.$^{-1}$')
    # add_test('mean curvatures', 'Mean curvatures:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '', 'norm.')
    # add_test('skew curvatures', 'Skewness curvatures:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '', 'norm.')
    # add_test('kurtosis curvatures', 'Kurtosis curvatures:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '', 'norm.')
    # add_test('RMS curvatures', 'RMS curvatures:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '', 'norm.')
    # add_test('RMS curvatures', 'RMS curvatures:', 0.35, 0.5, [0.4, 0.425, 0.45], RMScurvs, my_stages, 'px.$^{-1}$')
    # add_test('normalized curvatures', 'Norm curvatures:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '', 'norm.')
    # add_test('normalized curvatures', 'Norm curvatures:', 0.20, 0.4, [0.25, 0.3, 0.35], Normcurvs, my_stages, 'px.$^{-1}$')
    # add_test('RGB Y', 'RGB Y:', 0, 0.5, [0, 0.25, 0.5], [], '', '')
    # add_test('CIE Y', 'CIE Y:', 0, 100, [0, 25, 50, 75, 100], [], '', '') 
    # add_test('lab b Y', 'lab b Y:', 0, 50, [0, 25, 50], [], '', '')
    # add_test('HSV Y', 'HSV Y:', 0, 0.5, [0, 0.25, 0.5], [], '', '')

    return tests, variables, miny, maxy, yticks, my_lines, my_labels, my_units

def Setup_D2D3():
    
    # Initialize empty lists for tests, variables, etc.
    tests = []
    variables = []
    miny = []
    maxy = []
    yticks = []
    my_lines = []
    my_labels = []
    my_units = []
    
    # Function to add a test and corresponding parameters
    def add_test(test_name, variable_name, min_val, max_val, y_tick_vals, line_vals, label_val, label_unit):
        tests.append(test_name)
        variables.append(variable_name)
        miny.append(min_val)
        maxy.append(max_val)
        yticks.append(y_tick_vals)
        my_lines.append(line_vals)
        my_labels.append(label_val)
        my_units.append(label_unit)
        
    
    directory_path = base_path + "1 Utilities/Test images/Results/"
    
    ###########################################################################
    ###########################################################################
    # Fig 3:
        
    # In SI for both D2 and D3:
    #add_test('area', 'Areas:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '', 'norm.')
    #add_test('perimeter', 'Perimeters:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '', 'norm.')
    add_test('major axis', 'Major axes:', 0.5, 1.50, [0.75, 1.0, 1.25], [1.0], '', 'norm.')
    
    # In SI for D2:
    #add_test('l*a*b* yellowness', 'lab b Y binary:', 0.5, 1.75, [0.75, 1.0, 1.25], [1.0], '','norm.')
    #add_test('RGB yellowness', 'RGB Y binary:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '','norm.')
    #add_test('Yellowness Index', 'CIE Y binary:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '','norm.')
    #add_test('HSV yellowness', 'HSV Y binary:',0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '','norm.')
    #add_test('l*a*b* yellowness', 'lab b Y binary:', 0, 60, [20, 30, 40], [], '', '')
    #add_test('RGB yellowness', 'RGB Y binary:', 0, 120, [20, 60, 100], [], '', '')
    #add_test('Yellowness Index', 'CIE Y binary:', 0, 100, [25, 50, 75], [], '', '')
    #add_test('HSV yellowness', 'HSV Y binary:', 0, 0.6, [0.2, 0.3, 0.4], [], '', '')
     
    ##### NOT THERE YET
    # For avg violin only:
    #add_test('curvature', 'Curvatures:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '', 'norm.')
    

    ###########################################################################
    ###########################################################################
    # Fig 4:
    
    # Major axis but also perimeter, roundness, elongation
    #add_test('major axis', 'Major axes:', 0.5, 1.75, [0.75, 1.0, 1.25, 1.50], [1.0], '', 'norm.')
    #add_test('perimeter', 'Perimeters:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '', 'norm.')
    #add_test('curvature', 'Curvatures:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '', 'norm.')
    
    #add_test('solidity', 'Solidities:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '','norm.')
    # sols, successful_stages = ut.extract_mean_quantity_skipping(directory_path, "Solidities:", [38, 41])     # Solidity
    # successful_stages = [f"stage {str(s)}" for s in successful_stages]
    # if not ut.is_decreasing(sols): # Solidity should be decreasing 
    #       raise ValueError("Solidity is not strictly decreasing!")
    # add_test('solidity', 'Solidities:', 0.6, 1.025, [0.7, 0.8, 0.9,1.0], sols, successful_stages,'')
    
    #add_test('elongation', 'Elongations:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '','norm.')
    # elongs, successful_stages  = ut.extract_mean_quantity_skipping(directory_path, "Elongations:", [39, 40, 41, 42]) # Elongation
    # successful_stages = [f"stage {str(s)}" for s in successful_stages]
    # if not ut.is_increasing(elongs): # Elongation should be increasing
    #     raise ValueError("Elongation is not strictly increasing!")
    # add_test('elongation', 'Elongations:', 0.5, 0.8, [0.55, 0.65, 0.75], elongs, successful_stages,'')
    
    #add_test('roundness', 'Roundnesses:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '','norm.')
    roundns, successful_stages = ut.extract_mean_quantity_skipping(directory_path, "Roundnesses:", [38, 40, 42])      # Roundness
    successful_stages = [f"stage {str(s)}" for s in successful_stages]
    if not ut.is_decreasing(roundns): # Roundness should be decreasing
        raise ValueError("Roundness is not strictly decreasing!") 
    add_test('roundness', 'Roundnesses:', 0.1, 0.6, [0.25, 0.5], roundns, successful_stages,'') 

    #add_test('area', 'Areas:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '', 'norm.')
    
    ###########################################################################
    ###########################################################################
    # NOT DOING THE BELOW BECAUSE UNFAIR COMPARISONS
    
    # add_test('major axis', 'Major axes:', 500, 1750, [750, 1000, 1250, 1500], [], '', 'px.') # NOT DOING
    
    #curvs, successful_stages   = ut.extract_mean_quantity_skipping(directory_path, "Curvatures:", [33, 35, 37, 38, 39, 40, 41, 42])     # Curvs
    #print(curvs)
    #print(successful_stages) # uclear what should be happening
    # add_test('curvature', 'Curvatures:', 400, 1100, [500, 700, 900, 1000],curvs, successful_stages, 'px.$^{-1}$') #NOT DOING
    
    # add_test('area', 'Areas:', 100000, 400000, [150000, 250000, 350000], [], '', 'px.$^2$')
    # add_test('perimeter', 'Perimeters:', 1000, 5000, [2000, 3000, 4000], [1.0], '', 'px.')

    # ratioAP, successful_stages = ut.extract_mean_quantity_skipping(directory_path, "area/perimeter:", [35, 38, 40, 42])     # Area/Per
    # if not ut.is_decreasing(ratioAP): # Solidity should be decreasing 
    #     raise ValueError("Ratio AP is not strictly decreasing!")
    # add_test('ratio area to perimeter', 'area/perimeter:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '', 'norm.')
    # add_test('ratio area to perimeter', 'area/perimeter:', 40, 110, [50, 75, 100], ratioAP, successful_stages, 'px.')
    
    # roundns, successful_stages = ut.extract_mean_quantity_skipping(directory_path, "Roundnesses:", [38, 40, 41, 42])      # Roundness
    # if not ut.is_decreasing(roundns): # Roundness should be decreasing
    #     raise ValueError("Roundness is not strictly decreasing!") 
    # # add_test('roundness', 'Roundnesses:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '','norm.')
    # add_test('roundness', 'Roundnesses:', 0.1, 0.9, [0.25, 0.5, 0.75], roundns, successful_stages,'') 
        
    # add_test('pigmentation', 'Pigmentations:', -0.25, 2.25, [0.5, 1.0, 1.5], [1.0], '', 'norm.')
    # add_test('pigmentation', 'Pigmentations:', -0.25, 2.25, [0.5, 1.0, 1.5], [1.0], '', '')
    
    # elongs, successful_stages  = ut.extract_mean_quantity_skipping(directory_path, "Elongations:", [39, 40, 41, 42]) # Elongation
    # if not ut.is_increasing(elongs): # Elongation should be increasing
    #     raise ValueError("Elongation is not strictly increasing!")
    # add_test('elongation', 'Elongations:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '','norm.')
    # add_test('elongation', 'Elongations:', 0.45, 0.85, [0.5, 0.6, 0.7, 0.8], elongs, successful_stages,'')
    
    # BB, successful_stages      = ut.extract_mean_quantity_skipping(directory_path, "BBARs:",  [39, 40, 41, 42])     # BB
    # if not ut.is_increasing(BB): # BB should be increasing
    #     raise ValueError("BB is not strictly increasing!")   
    # add_test('BBAR', 'BBARs:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '','norm.')
    # add_test('BBAR', 'BBARs:', 2, 6.0, [3, 4, 5], BB, my_stages,'')
    
    # eccs, successful_stages    = ut.extract_mean_quantity_skipping(directory_path, "Eccentricities:", [38, 39, 40, 41, 42])     # Eccentricity
    # if not ut.is_increasing(eccs): # Eccentricity should be increasing
    #     raise ValueError("Eccentricity is not strictly increasing!")   
    # add_test('eccentricity', 'Eccentricities:', 0.9, 1.1, [0.95, 1.0, 1.05], [1.0], '','norm.')
    # add_test('eccentricity', 'Eccentricities:', 0.90, 1.0, [0.925, 0.95, 0.975, 1.0], eccs, successful_stages,'')
    
    # convs, successful_stages   = ut.extract_mean_quantity_skipping(directory_path, "Convexities:", [38, 39, 41, 42])     # Convexity
    # if not ut.is_decreasing(convs): # Convexity should be decreasing
    #     raise ValueError("Convexity is not strictly decreasing!") 
    # add_test('convexity', 'Convexities:', 0.9, 1.1, [0.95, 1.0, 1.05], [1.0], '', 'norm.')
    # add_test('convexity', 'Convexities:', 0.85, 1.05, [0.9, 0.95, 1.0], convs, successful_stages, '')    

    # sols, successful_stages = ut.extract_mean_quantity_skipping(directory_path, "Solidities:", [38, 40, 41])     # Solidity
    # if not ut.is_decreasing(sols): # Solidity should be decreasing 
    #     raise ValueError("Solidity is not strictly decreasing!")
    # add_test('solidity', 'Solidities:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '','norm.')
    # add_test('solidity', 'Solidities:', 0.6, 1.0, [0.7, 0.8, 0.9], sols, my_stages,'')
   
    #curvs, successful_stages   = ut.extract_mean_quantity_skipping(directory_path, "Curvatures:", [33, 35, 37, 38, 39, 40, 41, 42])     # Curvs
    #print(curvs)
    #print(successful_stages) # uclear what should be happening
    #add_test('curvature', 'Curvatures:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '', 'norm.')
    #add_test('curvature', 'Curvatures:', 400, 1100, [500, 700, 900, 1000],curvs, successful_stages, 'px.$^{-1}$')
    
    #RMScurvs, successful_stages = ut.extract_mean_quantity_skipping(directory_path, "RMS curvatures:", [33, 35, 37, 38, 39, 40, 41, 42])     # RMScurvs
    #print(RMScurvs)
    #print(successful_stages) # uclear what should be happening
    # add_test('RMS curvatures', 'RMS curvatures:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '', 'norm.')
    # add_test('RMS curvatures', 'RMS curvatures:', 0.35, 0.5, [0.4, 0.425, 0.45], [], '', 'px.$^{-1}$')
    
    # Normcurvs, successful_stages = ut.extract_mean_quantity_skipping(directory_path, "Norm curvatures:", [33, 35, 37, 38, 39, 40, 41, 42])     # Normcurvs
    # print(Normcurvs)
    # print(successful_stages) # uclear what should be happening
    # add_test('normalized curvatures', 'Norm curvatures:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '', 'norm.')
    # add_test('normalized curvatures', 'Norm curvatures:', 0.20, 0.4, [0.25, 0.3, 0.35], [], '', 'px.$^{-1}$')

    # add_test('mean curvatures', 'Mean curvatures:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '', 'norm.')
    # add_test('mean curvatures', 'Mean curvatures:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '', '')
    # add_test('skew curvatures', 'Skewness curvatures:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '', 'norm.')
    # add_test('kurtosis curvatures', 'Kurtosis curvatures:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '', 'norm.')
    
    # add_test('RGB Y', 'RGB Y:', 0, 0.5, [0, 0.25, 0.5], [], '', '')
    # add_test('RGB Y', 'RGB Y:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '','norm.')
    # add_test('CIE Y', 'CIE Y:', 0, 75, [0, 25, 50, 75], [], '', '') 
    # add_test('CIE Y', 'CIE Y:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '','norm.')
    # add_test('lab b Y', 'lab b Y:', 0, 50, [0, 25, 50], [], '', '')
    # add_test('lab b Y', 'lab b Y:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '','norm.')
    # add_test('HSV Y', 'HSV Y:', 0, 0.5, [0, 0.25, 0.5], [], '', '')
    # add_test('HSV Y', 'HSV Y:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '','norm.')
    
    # add_test('RGB Y binary', 'RGB Y binary:', 0, 0.5, [0, 0.25, 0.5], [], '', '')
    # add_test('RGB Y binary', 'RGB Y binary:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '','norm.')
    # add_test('CIE Y binary', 'CIE Y binary:', 0, 75, [0, 25, 50, 75], [], '', '')
    # add_test('CIE Y binary', 'CIE Y binary:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '','norm.')
    # add_test('lab b Y binary', 'lab b Y binary:', 0, 50, [0, 25, 50], [], '', '')
    # add_test('lab b Y binary', 'lab b Y binary:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '','norm.')
    # add_test('HSV Y binary', 'HSV Y binary:', 0, 0.5, [0, 0.25, 0.5], [], '', '')
    # add_test('HSV Y binary', 'HSV Y binary:',0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '','norm.')
    
    # Not relevant
    # add_test('max curvatures', 'Max curvatures:', 0, 0.5, [0, 0.25, 0.5], [], '', 'norm.')
    # add_test('curvature std', 'Curvature Stds:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '')
    # add_test('radii of curvatures', 'Radius of curvatures:', 0.95, 1.05, [0.975, 1.0, 1.025], [1.0], '', 'norm.')
    # add_test('radii of curvatures', 'Radius of curvatures:', 1.3, 1.6, [1.45, 1.50, 1.55], [], '', 'px.')
    # add_test('Frechet', 'Frechets:', 0.5, 1.5, [0.75, 1.0, 1.25], [1.0], '', 'norm.')
    # add_test('Frechet', 'Frechets:', 100, 200, [125, 150, 175], [], '', 'px.')
    
    return tests, variables, miny, maxy, yticks, my_lines, my_labels, my_units

def Setup_Delta(my_test):
    
    # Initialize empty lists for tests, variables, etc.
    tests = []
    variables = []
    miny = []
    maxy = []
    yticks = []
    my_lines = []
    my_labels = []
    my_units = []
    
    # Function to add a test and corresponding parameters
    def add_test(test_name, variable_name, min_val, max_val, y_tick_vals, line_vals, label_val, label_unit):
        tests.append(test_name)
        variables.append(variable_name)
        miny.append(min_val)
        maxy.append(max_val)
        yticks.append(y_tick_vals)
        my_lines.append(line_vals)
        my_labels.append(label_val)
        my_units.append(label_unit)
        
    if my_test == 'Relative_difference_nonabs': #bound [0,1]
    
       add_test('major axis', 'Major axes:', -50, 50, [-25, 0, 25], [0.0], '', 'norm.')
       add_test('curvature', 'Curvatures:',  -50, 50, [-25, 0, 25], [0.0], '', 'norm.')
       add_test('convexity', 'Convexities:',  -10, 10, [-5, 0, 5], [0.0], '', 'norm.')
       add_test('solidity', 'Solidities:',     -25, 25, [-10, 0, 10], [0.0], '', 'norm.')
        
        # add_test('major axis', 'Major axes:', -1, 1, [], [], '', '')
        
        # add_test('convexity', 'Convexities:', -0.2, 0.2, [], [], '', '')
        
        # add_test('solidity', 'Solidities:', -1, 1, [], [], '', '')    
     
        # add_test('curvature', 'Curvatures:',  -1, 1, [], [], '', '')
         
         
    elif my_test == 'Visibility': #bound [-100,100]
     
          #add_test('major axis', 'Major axes:', -25, 25, [-10,0,10], [0.0], '', 'norm.')
          #add_test('curvature', 'Curvatures:',  -25, 25, [-10,0,10], [0.0], '','norm.')
          add_test('convexity', 'Convexities:', -7.5,7.5,[-5, 0, 5], [0.0], '','norm.')
          #add_test('solidity', 'Solidities:',    -15, 15, [-10,0,10], [0.0], '', 'norm.')
          
          # add_test('major axis', 'Major axes:', -10,10, [], [], '', '')
         
          # add_test('convexity', 'Convexities:', -10,20, [], [], '', '')
          
          # add_test('solidity', 'Solidities:', -10,10, [], [], '', '')
          
          # add_test('curvature', 'Curvatures:',  -10,10, [], [], '', '')
     
    elif my_test == 'Z_score_comparison':
        
        
        
        # add_test('major axis', 'Major axes:',  -4, 6.0, [-1, 0, 1], [0.0], '','norm.')
        # add_test('curvature', 'Curvatures:',   -4, 6.0, [-1, 0, 1], [0.0], '','norm.')
        # add_test('convexity', 'Convexities:',  -4, 6.0, [-1, 0, 1], [0.0], '','norm.')
        # add_test('solidity', 'Solidities:',     -4, 6.0, [-1, 0, 1], [0.0], '','norm.')
        
        #add_test('major axis', 'Major axes:', -3, 5, [-2, -1, 0, 1, 2, 3, 4], [], '', '')
        #add_test('convexity', 'Convexities:', -2, 2, [-1, 0 ,1], [], '', '')
        #add_test('solidity', 'Solidities:', -2, 2, [-1, 0, 1], [], '', '')  
        #add_test('curvature', 'Curvatures:',  -2, 2, [-1, 0, 1], [], '', '')
        
        #add_test('roundness', 'Roundnesses:',  -4, 6.0, [-1, 0, 1], [0.0], '','norm.')
        #add_test('area', 'Areas:',  -4, 6.0, [-1, 0, 1], [0.0], '','norm.')
        add_test('perimeter', 'Perimeters:',  -4, 6.0, [-1, 0, 1], [0.0], '','norm.')
        
    elif my_test == 'Percentage_change':
        
        add_test('major axis', 'Major axes:', -50, 50, [-25, 0, 25], [0.0], '', 'norm.')
        add_test('curvature', 'Curvatures:',   -50, 150, [-25, 0, 25, 50, 75, 100, 125], [0.0], '','norm.')
        add_test('convexity', 'Convexities:',  -10, 10, [-5, 0, 5], [0.0], '', 'norm.')
        add_test('solidity', 'Solidities:',     -25, 25, [-10, 0, 10], [0.0], '', 'norm.')
        
        #add_test('major axis', 'Major axes:', -10, 75, [0, 25, 50], [0.0], '', '')
        

        # add_test('convexity', 'Convexities:', 0, 100, [], [], '', '')
       
        # add_test('solidity', 'Solidities:', 0, 100, [], [], '', '')
        
        # add_test('curvature', 'Curvatures:',  0, 100, [], [], '', '')
    
    return tests, variables, miny, maxy, yticks, my_lines, my_labels, my_units 
