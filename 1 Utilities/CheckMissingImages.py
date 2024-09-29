##### needs to be modified so that day 1, only alive at day 1; day 2, only alive at day 2 etc, not going via status d3 at all

import os
import ast

def extract_filenames_from_all_txt_files(directory_path, my_day):
    """
    Extract filenames from all txt files in the given directory where the StatusD3 is 1, 2, or 3.
    Modify the filenames by inserting 'D1', 'D2', or 'D3' after the third character based on `my_day`.
    """
    all_selected_filenames = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            txt_file_path = os.path.join(directory_path, filename)
            with open(txt_file_path, 'r') as file:
                content = file.read()

            # Extracting the 'Filenames:' and 'StatusD3:' sections
            filenames_str = content.split("Filenames:")[1].split("StageD1:")[0].strip()
            status_str = content.split("StatusD3:")[1].split("%")[0].strip()

            # Converting the string representations to actual Python lists
            filenames = ast.literal_eval(filenames_str)
            status_list = ast.literal_eval(status_str)

            # Filter filenames based on the StatusD3 values
            selected_filenames = [fname for fname, status in zip(filenames, status_list) if status in [1, 2, 3]]

            if my_day == 1:
                day_str = 'D1'
            elif my_day == 2:
                day_str = 'D2'
            elif my_day == 3:
                day_str = 'D3'
            else:
                raise ValueError("Need to choose a day: 1, 2 or 3 only.")

            # Modify the filenames by inserting 'Dn' after the 3rd character
            modified_filenames = [fname[:3] + f'{day_str}' + fname[3:] for fname in selected_filenames]

            # Append modified filenames to the overall list
            all_selected_filenames.extend(modified_filenames)

    return all_selected_filenames

def check_missing_files(txt_directory, png_directory1, png_directory2, my_day):
    """
    Check for missing files by comparing the filenames extracted from txt files
    with the png files in the given directories.
    Also, check for extra PNG files that are not required.
    """
    # Extract filenames from all txt files
    extracted_filenames = extract_filenames_from_all_txt_files(txt_directory, my_day)

    # Get a list of all PNG files in both directories, excluding system files
    png_files_dir1 = set(f for f in os.listdir(png_directory1) if f.endswith('.png'))
    png_files_dir2 = set(f for f in os.listdir(png_directory2) if f.endswith('.png'))

    # Combine the sets from both directories
    all_png_files = png_files_dir1.union(png_files_dir2)

    # Check which extracted filenames are missing in the png directories
    missing_files = [fname + ".png" for fname in extracted_filenames if fname + ".png" not in all_png_files]

    # Check for extra files in the PNG directories that are not needed
    extra_files = [fname for fname in all_png_files if fname[:-4] not in extracted_filenames]

    return missing_files, extra_files

###############################################################################
###############################################################################

# Batch 1: D1, D2, D3 all present
# Batch 2: D1,     D3
# Batch 3: D1, D2,
# Batch 4: D1
# Batch 5:   , D3
# Batch 6: 
# Batch 7: D1, D2, D3

my_batches = [8] #[1,2,3,4,5,6,7,8] 

for my_batch in my_batches:
    
    for my_day in [3]: #[1,2,3]:

        if my_day == 1:
            day_path = '3 D1 quantification'
        elif my_day == 2:
            day_path = '4 D2 quantification'
        elif my_day == 3:
            day_path = '5 D3 quantification'
        else:
            raise ValueError("Need to choose a day: 1, 2 or 3 only.")

        txt_folder = f"/Users/clarice/Desktop/2 Experimental overview/Assessments/B{my_batch}"
        png_folder1 = f"/Users/clarice/Desktop/{day_path}/B{my_batch}/B{my_batch}C"
        png_folder2 = f"/Users/clarice/Desktop/{day_path}/B{my_batch}/B{my_batch}H"

        missing_files, extra_files = check_missing_files(txt_folder, png_folder1, png_folder2, my_day)

        if missing_files:
            print("Missing PNG files:")
            for file in missing_files:
                print(file)
            raise ValueError(f"FIX MISSING FILES for batch {my_batch} in day {my_day}!!!")
        else:
            print(f"All files are present in batch {my_batch} in day {my_day}.")
            
        # if extra_files:
        #     print("Extra PNG files:")
        #     for file in extra_files:
        #         print(file)
        #     raise ValueError(f"FIX EXTRA FILES for batch {my_batch} in day {my_day}!!!")
        # else:
        #     print(f"No extra files in batch {my_batch} in day {my_day}.")   

# def extract_filenames_from_all_txt_files(directory_path, my_day):
#     """
#     Extract filenames from all txt files in the given directory where the StatusD3 is 1, 2, or 3.
#     Modify the filenames by inserting 'D2' after the third character.
#     """
#     all_selected_filenames = []

#     for filename in os.listdir(directory_path):
#         if filename.endswith(".txt"):
#             txt_file_path = os.path.join(directory_path, filename)
#             with open(txt_file_path, 'r') as file:
#                 content = file.read()

#             # Extracting the 'Filenames:' and 'StatusD3:' sections
#             filenames_str = content.split("Filenames:")[1].split("StageD1:")[0].strip()
#             status_str = content.split("StatusD3:")[1].split("%")[0].strip()

#             # Converting the string representations to actual Python lists
#             filenames = ast.literal_eval(filenames_str)
#             status_list = ast.literal_eval(status_str)

#             # Filter filenames based on the StatusD3 values
#             selected_filenames = [fname for fname, status in zip(filenames, status_list) if status in [1, 2, 3]]

#             if my_day == 1:
#                 day_str = 'D1'
#             elif my_day == 2:
#                 day_str = 'D2'
#             elif my_day == 3:
#                 day_str = 'D3'
#             else:
#                 raise ValueError("Need to choose a day: 1, 2 or 3 only.")
#             # Modify the filenames by inserting 'Dn' after the 3rd character
#             modified_filenames = [fname[:3] + f'{day_str}' + fname[3:] for fname in selected_filenames]

#             # Append modified filenames to the overall list
#             all_selected_filenames.extend(modified_filenames)

#     return all_selected_filenames

# def check_missing_files(txt_directory, png_directory1, png_directory2, my_day):
#     """
#     Check for missing files by comparing the filenames extracted from txt files
#     with the png files in the given directories.
#     """
#     # Extract filenames from all txt files
#     extracted_filenames = extract_filenames_from_all_txt_files(txt_directory, my_day)

#     # Get a list of all PNG files in both directories
#     png_files_dir1 = set(os.listdir(png_directory1))
#     png_files_dir2 = set(os.listdir(png_directory2))

#     # Combine the sets from both directories
#     all_png_files = png_files_dir1.union(png_files_dir2)

#     # Check which extracted filenames are missing in the png directories
#     missing_files = [fname for fname in extracted_filenames if fname + ".png" not in all_png_files]

#     return missing_files