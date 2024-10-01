import os

def get_image_names(folder_path):
    """Get a set of image file names from the specified folder (excluding file extensions)."""
    return {os.path.splitext(file)[0] for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))}

def compare_folders(folder1, folder2):
    """Compare image names from two folders and print the differences."""
    images_folder1 = get_image_names(folder1)
    images_folder2 = get_image_names(folder2)

    only_in_folder1 = images_folder1 - images_folder2
    only_in_folder2 = images_folder2 - images_folder1

    if only_in_folder1:
        print("Images only in folder 1:")
        for image in only_in_folder1:
            print(image)
    else:
        print("No unique images in folder 1.")

    if only_in_folder2:
        print("\nImages only in folder 2:")
        for image in only_in_folder2:
            print(image)
    else:
        print("No unique images in folder 2.")



folder1 = '/Users/clarice/Desktop/5 D3 quantification/B6/B6C'
folder2 = '/Users/clarice/Desktop/5 D3 quantification/B6OLD/B6C'

compare_folders(folder1, folder2)
