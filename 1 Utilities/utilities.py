# utilities.py

import os
import numpy as np
import ast
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties
import seaborn as sns
from PIL import Image
from skimage.io import imread, imsave
from skimage.transform import rotate
from skimage.util import img_as_bool
from skimage.measure import label, regionprops, find_contours
from skimage.draw import line
from skimage.morphology import convex_hull_image
from skimage.color import rgb2gray
from scipy.stats import ttest_ind, mannwhitneyu
from scipy.spatial import ConvexHull
from skimage.filters import threshold_otsu
from skimage.color import rgb2lab
import cv2
from pdf2image import convert_from_path
import scipy.stats as stats
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
from statsmodels.formula.api import ols
import statsmodels.api as sm
import statsmodels.stats.multicomp as multi
import pingouin as pg
from sklearn.utils import resample
from scikit_posthocs import posthoc_dunn
from scipy.stats import kstest, f_oneway
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D  # Import Line2D for custom legend
from matplotlib.legend_handler import HandlerTuple
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import rankdata
from scipy.stats import chi2
from mpmath import mp, mpf, gammainc
import statsmodels.stats.multicomp as mc

from scipy.spatial.distance import directed_hausdorff
from skimage.draw import line
from skimage.draw import polygon_perimeter
from skimage import color
from skimage.color import rgb2lab, rgb2hsv, xyz2lab, rgb2xyz
from skimage import measure
from scipy.stats import skew, kurtosis
from scipy.stats import chi2_contingency, fisher_exact, kruskal, mannwhitneyu, multinomial#, cohen_kappa_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.patches import ConnectionPatch


def apply_mask_to_image(image, mask):
    """Apply the mask to the image, setting non-masked pixels to black."""
    np_image = np.array(image)
    
    # Set non-white (non-masked) pixels to black
    np_image[~mask] = [0, 0, 0, 0]  # Set RGB and alpha channels to 0 for masked-out pixels
    
    # Convert back to an Image object
    masked_image = Image.fromarray(np_image)
    
    return masked_image

def binarize_image(image, threshold):
    """Binarize an image."""
    grayscale = image.convert('L')  # Convert to grayscale
    binarized = grayscale.point(lambda p: p > threshold and 255)  # Apply threshold
    return binarized

def replace_unquoted_d_e(content):
    """Replace unquoted D and E with 'D' and 'E' in the list strings."""
    lines = content.splitlines()
    for i in range(len(lines)):
        if 'D' in lines[i]:
            lines[i] = lines[i].replace('D', "'D'")
        if 'E' in lines[i]:
            lines[i] = lines[i].replace('E', "'E'")
    return '\n'.join(lines)

def trim_assess_lists(filenames_c, filenames_assess, assess_lists):
    """Trim the lists in assess_lists to only include entries with filenames in filenames_c."""
    indices_to_keep = [i for i, fname in enumerate(filenames_assess) if fname in filenames_c]
    trimmed_lists = {key: [value[i] for i in indices_to_keep] for key, value in assess_lists.items()}
    return trimmed_lists

def map_colors(path, path_assess):
    # Read files
    name_content = read_file(path)
    assess_content = read_file(path_assess)
    
    # Extract Filenames lists
    name_filenames = extract_list(name_content, 'Filenames')
    assess_filenames = extract_list(assess_content, 'Filenames')
    
    # Compare the lists
    if name_filenames == assess_filenames:
        # Extract StatusD3 list
        status_d3_list = extract_list(assess_content, 'StatusD3')
        
        # Map status to colors
        colors = map_status_to_color(status_d3_list)
    else:
        raise ValueError("The 'Filenames' lists are not identical for" + path)
        
    return colors

def read_and_process(path, assess_path):
    # Read the files
    file_content = read_file(path)
    file_content_assess = read_file(assess_path)
    
    # Extract the lists
    filenames = extract_list(file_content, 'Filenames:')
    filenames_assess = extract_list(file_content_assess, 'Filenames:')
    stageD1_assess = extract_list(file_content_assess, 'StageD1:')
    stageD3_assess = extract_list(file_content_assess, 'StageD3:')
    statusD3_assess = extract_list(file_content_assess, 'StatusD3:')
    
    # Trim the assess lists
    assess_lists = {'StageD1': stageD1_assess, 'StageD3': stageD3_assess, 'StatusD3': statusD3_assess}
    trimmed_assess_lists = trim_assess_lists(filenames, filenames_assess, assess_lists)
    
    # Debugging print statements
    print(f"Length of trimmed filenames: {len(trimmed_assess_lists['StageD1'])}")
    print(f"Length of trimmed statusD3: {len(trimmed_assess_lists['StatusD3'])}")
    
    # Map statuses to colors
    statusD3_trimmed = trimmed_assess_lists['StatusD3']
    return map_status_to_color(statusD3_trimmed)

def set_stick_colors(BC_path, BC_assess_path, BH_path, BH_assess_path):
    stick_colors = [read_and_process(BC_path, BC_assess_path), read_and_process(BH_path, BH_assess_path)]
    return stick_colors

# Function to draw a thick line on the image
def draw_thick_line(image, rr, cc, color=[255, 105, 180, 255], thickness=10):
    for d in range(-thickness // 2, thickness // 2 + 1):
        rr_clipped = np.clip(rr + d, 0, image.shape[0] - 1)
        cc_clipped = np.clip(cc + d, 0, image.shape[1] - 1)
        image[rr_clipped, cc_clipped, :] = color  # Ensure to apply color correctly in RGBA format

# Calculate the endpoints for the major and minor axes
def calculate_endpoints(centroid, angle, length):
    y0, x0 = centroid
    delta_x = np.cos(angle) * length / 2
    delta_y = np.sin(angle) * length / 2
    return [(int(y0 - delta_y), int(x0 - delta_x)), (int(y0 + delta_y), int(x0 + delta_x))]

# Function to adjust the position of each axis
def adjust_position(ax, left, bottom, width, height):
    pos = ax.get_position()
    pos.x0 = left
    pos.x1 = left + width
    pos.y0 = bottom
    pos.y1 = bottom + height
    ax.set_position(pos)
    
def calculate_axis_endpoints(centroid, orientation, length):
    """Calculate axis endpoints given centroid, orientation, and length."""
    y0, x0 = centroid
    delta_x = np.cos(orientation) * length / 2
    delta_y = np.sin(orientation) * length / 2
    return (int(y0 - delta_y), int(x0 - delta_x)), (int(y0 + delta_y), int(x0 + delta_x))   

def read_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def extract_list(content, list_name):
    start_index = content.find(list_name)
    if start_index == -1:
        print(f"{list_name} not found in the file.")
        return None
    start_index = content.find('[', start_index)
    end_index = content.find(']', start_index) + 1
    if start_index == -1 or end_index == -1:
        print(f"Error finding the list {list_name} in the content.")
        return None
    list_str = content[start_index:end_index]
    try:
        return ast.literal_eval(list_str)
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing list {list_name}: {e}")
        print(f"List content: {list_str}")
        return None
    
def extract_array(file_path, NameArray):
    """
    Extract values from a specified section in a text file and return them as a numpy array.
    
    Parameters:
    file_path (str): The path to the text file.
    NameArray (str): The name of the array to extract.
    
    Returns:
    np.array: A numpy array containing the extracted values.
    """
    # Read the content of the file
    with open(file_path, 'r') as file:
        content = file.readlines()

    # Extract the section
    section_list = []
    recording = False

    for my_line in content:
        if NameArray in my_line:
            recording = True
            continue
        if recording:
            if my_line.strip() == "" or ':' in my_line:  # End of section or start of a new section
                break
            section_list.extend([float(value) for value in my_line.strip().strip('[]').split(', ')])

    # Convert the list to a numpy array
    section_array = np.array(section_list)
    return section_array

def map_status_to_color(status_list):
    """
    Gives the color of the datapoint, given the status of the embryo
    """
    color_map = {'1': 'green', '2': 'orange', '3': 'red', 'D': 'black'}
    return [color_map[str(status)] for status in status_list]

def replace_D_and_calculate_avg_std(filtered_stage):
    # Remove 'D' entries
    numeric_values = [int(value) for value in filtered_stage if value != 'D']
    # Calculate average and standard deviation
    if numeric_values:
        average_value = np.mean(numeric_values)
        std_value = np.std(numeric_values)
        median_value = np.median(numeric_values)
    else:
        average_value = None
        std_value = None
        median_value = None
    return average_value, std_value, median_value

def replace_D(filtered_stages):
    # Remove 'D' entries from each sublist
    numeric_values = [[int(value) for value in stage if value != 'D'] for stage in filtered_stages]
    return numeric_values

def replace_D_list(filtered_stage, rem_idx=False):
    # Find indices of 'D' entries
    removed_indices = [i for i, value in enumerate(filtered_stage) if value == 'D']
    # Remove 'D' entries
    numeric_values = [int(value) for i, value in enumerate(filtered_stage) if value != 'D']
    if rem_idx:
        return numeric_values, removed_indices
    else:
        return numeric_values
    
def remove_indices_from_list(original_list, indices_to_remove):
    return [value for i, value in enumerate(original_list) if i not in indices_to_remove]

def plot_poster(fig, axes, font_title, font_text, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, colorlist, my_batch, my_condition, my_plate, d1_path, d2_path, d3_path, stages, surviving, avg_stages, std_stages, output_path_pdf, output_path_png, save=True, my_dpi=600):
    
    # Adjusting padding to bring subplots closer together
    fig.tight_layout(pad=0.5)
    plt.subplots_adjust(wspace=0.1, hspace=0.3)  # Adjust as needed
    
    for i in range(1, 25):
        if i < 10:
            d1_image_path = os.path.join(d1_path, f'B{my_batch}{my_condition}D1P{my_plate}-0{i}.pdf')
            d2_image_path = os.path.join(d2_path, f'B{my_batch}{my_condition}D2P{my_plate}-0{i}.pdf')
            d3_image_path = os.path.join(d3_path, f'B{my_batch}{my_condition}D3P{my_plate}-0{i}.pdf')
        else:
            d1_image_path = os.path.join(d1_path, f'B{my_batch}{my_condition}D1P{my_plate}-{i}.pdf')
            d2_image_path = os.path.join(d2_path, f'B{my_batch}{my_condition}D2P{my_plate}-{i}.pdf')
            d3_image_path = os.path.join(d3_path, f'B{my_batch}{my_condition}D3P{my_plate}-{i}.pdf')
        
        d1_images = convert_from_path(d1_image_path)
        d2_images = convert_from_path(d2_image_path)
        d3_images = convert_from_path(d3_image_path)
        
        d1_image = d1_images[0]
        d2_image = d2_images[0]
        d3_image = d3_images[0]
        
        row, col = divmod(i-1, 6)
        ax = axes[row, col]
        
        # Create a blank image with the desired dimensions to paste the two images
        combined_image = Image.new('RGB', (778, 1002))
        combined_image.paste(d1_image, (0, 0))
        combined_image.paste(d2_image, (0, 334))
        combined_image.paste(d3_image, (0, 668))
        ax.imshow(combined_image)
        ax.set_title(f'#{i}, stages: ' + str(stages[0][i-1]) + '/' + str(stages[2][i-1]), fontproperties=font_text,color=colorlist[i-1])
        ax.title.set_size(MEDIUM_SIZE)
        ax.axis('off')
        
        # Minimalistic ruler
        ax_ruler = ax.inset_axes([-0.05, -0.05, 1.1, 0.05])  # Adjusted position 
        ax_ruler.set_ylim(0, 1)
        ax_ruler.set_xticks(range(0, 779, 200))
        ax_ruler.set_xticklabels(range(0, 779, 200)) #one tick per 200 pixels
        ax_ruler.tick_params(axis='y', which='both', left=False, labelleft=False, right=False, labelright=False)
        ax_ruler.tick_params(axis='x', which='both', top=False, bottom=True, labeltop=False, labelbottom=True, labelsize=10)
        
        for tick in range(0, 779, 200): #one tick per 200 pixels
            ax_ruler.plot([tick, tick], [0.2, 0.8], color='black', lw=0.8)
        
        ax_ruler.plot([0, 778], [0.5, 0.5], color='black', lw=0.8)
        ax_ruler.axis('off')

    plt.subplots_adjust(top=0.9, bottom=0.05, left=0.05, right=0.95)
    
    # Add a line of text at the bottom of the figure
    footer_text = ('1 div = 200 px $\cdot$ surviving: ' + str(surviving[0]) + ', ' 
                                                        + str(surviving[1]) + ', ' 
                                                        + str(surviving[2]) + '/24 $\cdot$ avg. stages: ' 
                                                        + str(avg_stages[0]) + '$\pm$' + str(std_stages[0]) + ' in day 1, ' 
                                                       # + str(avg_stages[1]) + '$\pm$' + str(std_stages[1]) + ', '  
                                                        + str(avg_stages[2]) + '$\pm$' + str(std_stages[2]) + ' in day 3')
    fig.text(0.5, 0, footer_text, ha='center', fontsize=MEDIUM_SIZE, fontproperties=font_text)
    
    # Save the figure
    if save:
        plt.savefig(output_path_pdf, format='pdf', dpi=my_dpi, bbox_inches='tight')
        plt.savefig(output_path_png, format='png', dpi=my_dpi, bbox_inches='tight')
    
    plt.show()
        
def plot_poster_48(fig, axes, font_title, font_text, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, colorlist, my_batch, my_condition, my_plate, d1_path, d2_path, d3_path, stages, surviving, avg_stages, std_stages, output_path_pdf, output_path_png, save=True, my_dpi=600):
    
    # Adjusting padding to bring subplots closer together
    fig.tight_layout(pad=0.5)
    plt.subplots_adjust(wspace=0.1, hspace=0.3)  # Adjust as needed
    
    for i in range(1, 49): #49
        if i < 10:
            d1_image_path = os.path.join(d1_path, f'B{my_batch}{my_condition}D1P{my_plate}-0{i}.pdf')
            d2_image_path = os.path.join(d2_path, f'B{my_batch}{my_condition}D2P{my_plate}-0{i}.pdf')
            d3_image_path = os.path.join(d3_path, f'B{my_batch}{my_condition}D3P{my_plate}-0{i}.pdf')
        else:
            d1_image_path = os.path.join(d1_path, f'B{my_batch}{my_condition}D1P{my_plate}-{i}.pdf')
            d2_image_path = os.path.join(d2_path, f'B{my_batch}{my_condition}D2P{my_plate}-{i}.pdf')
            d3_image_path = os.path.join(d3_path, f'B{my_batch}{my_condition}D3P{my_plate}-{i}.pdf')
        
        d1_images = convert_from_path(d1_image_path)
        d2_images = convert_from_path(d2_image_path)
        d3_images = convert_from_path(d3_image_path)
        
        d1_image = d1_images[0]
        d2_image = d2_images[0]
        d3_image = d3_images[0]
        
        row, col = divmod(i-1, 6)
        ax = axes[row, col]
        
        # Create a blank image with the desired dimensions to paste the two images
        combined_image = Image.new('RGB', (778, 1002))
        combined_image.paste(d1_image, (0, 0))
        combined_image.paste(d2_image, (0, 334))
        combined_image.paste(d3_image, (0, 668))
        ax.imshow(combined_image)
        ax.set_title(f'#{i}, stages: ' + str(stages[0][i-1]) + '/' + str(stages[2][i-1]), fontproperties=font_text,color=colorlist[i-1])
        ax.title.set_size(MEDIUM_SIZE)
        ax.axis('off')
        
        # Minimalistic ruler
        ax_ruler = ax.inset_axes([-0.05, -0.05, 1.1, 0.05])  # Adjusted position 
        ax_ruler.set_ylim(0, 1)
        ax_ruler.set_xticks(range(0, 779, 200))
        ax_ruler.set_xticklabels(range(0, 779, 200)) #one tick per 200 pixels
        ax_ruler.tick_params(axis='y', which='both', left=False, labelleft=False, right=False, labelright=False)
        ax_ruler.tick_params(axis='x', which='both', top=False, bottom=True, labeltop=False, labelbottom=True, labelsize=10)
        
        for tick in range(0, 779, 200): #one tick per 200 pixels
            ax_ruler.plot([tick, tick], [0.2, 0.8], color='black', lw=0.8)
        
        ax_ruler.plot([0, 778], [0.5, 0.5], color='black', lw=0.8)
        ax_ruler.axis('off')

    plt.subplots_adjust(top=0.95, bottom=0.025, left=0.05, right=0.95)
    
    # Add a line of text at the bottom of the figure
    footer_text = ('1 div = 200 px $\cdot$ surviving: ' + str(surviving[0]) + ', ' 
                                                        + str(surviving[1]) + ', ' 
                                                        + str(surviving[2]) + '/24 $\cdot$ avg. stages: ' 
                                                        + str(avg_stages[0]) + '$\pm$' + str(std_stages[0]) + ' in day 1, ' 
                                                       # + str(avg_stages[1]) + '$\pm$' + str(std_stages[1]) + ', '  
                                                        + str(avg_stages[2]) + '$\pm$' + str(std_stages[2]) + ' in day 3')
    fig.text(0.5, 0, footer_text, ha='center', fontsize=MEDIUM_SIZE, fontproperties=font_text)
    
    # Save the figure
    if save:
        plt.savefig(output_path_pdf, format='pdf', dpi=my_dpi, bbox_inches='tight')
        plt.savefig(output_path_png, format='png', dpi=my_dpi, bbox_inches='tight')
    
    plt.show()
    
def plot_poster_7days(fig, axes, font_title, font_text, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, colorlist, my_batch, my_condition, my_plate, d1_path, d2_path, d3_path, d4_path, d5_path, d6_path, d7_path, stages, surviving, avg_stages, std_stages, output_path_pdf, output_path_png, save=True, my_dpi=600):
    
    # Adjusting padding to bring subplots closer together
    fig.tight_layout(pad=0.5)
    plt.subplots_adjust(wspace=0.1, hspace=0.2)  # Adjust as needed
    
    for i in range(1, 25):
        if i < 10:
            d1_image_path = os.path.join(d1_path, f'B{my_batch}{my_condition}D1P{my_plate}-0{i}.pdf')
            d2_image_path = os.path.join(d2_path, f'B{my_batch}{my_condition}D2P{my_plate}-0{i}.pdf')
            d3_image_path = os.path.join(d3_path, f'B{my_batch}{my_condition}D3P{my_plate}-0{i}.pdf')
            d4_image_path = os.path.join(d4_path, f'B{my_batch}{my_condition}D4P{my_plate}-0{i}.pdf')
            d5_image_path = os.path.join(d5_path, f'B{my_batch}{my_condition}D5P{my_plate}-0{i}.pdf')
            d6_image_path = os.path.join(d6_path, f'B{my_batch}{my_condition}D6P{my_plate}-0{i}.pdf')
            d7_image_path = os.path.join(d7_path, f'B{my_batch}{my_condition}D7P{my_plate}-0{i}.pdf')
        else:
            d1_image_path = os.path.join(d1_path, f'B{my_batch}{my_condition}D1P{my_plate}-{i}.pdf')
            d2_image_path = os.path.join(d2_path, f'B{my_batch}{my_condition}D2P{my_plate}-{i}.pdf')
            d3_image_path = os.path.join(d3_path, f'B{my_batch}{my_condition}D3P{my_plate}-{i}.pdf')
            d4_image_path = os.path.join(d4_path, f'B{my_batch}{my_condition}D4P{my_plate}-{i}.pdf')
            d5_image_path = os.path.join(d5_path, f'B{my_batch}{my_condition}D5P{my_plate}-{i}.pdf')
            d6_image_path = os.path.join(d6_path, f'B{my_batch}{my_condition}D6P{my_plate}-{i}.pdf')
            d7_image_path = os.path.join(d7_path, f'B{my_batch}{my_condition}D7P{my_plate}-{i}.pdf')
        
        d1_images = convert_from_path(d1_image_path)
        d2_images = convert_from_path(d2_image_path)
        d3_images = convert_from_path(d3_image_path)
        d4_images = convert_from_path(d4_image_path)
        d5_images = convert_from_path(d5_image_path)
        d6_images = convert_from_path(d6_image_path)
        d7_images = convert_from_path(d7_image_path)
        
        d1_image = d1_images[0]
        d2_image = d2_images[0]
        d3_image = d3_images[0]
        d4_image = d4_images[0]
        d5_image = d5_images[0]
        d6_image = d6_images[0]
        d7_image = d7_images[0]
        
        row, col = divmod(i-1, 6)
        ax = axes[row, col]
        
        # Create a blank image with the desired dimensions to paste the two images
        combined_image = Image.new('RGB', (778, 2338))
        combined_image.paste(d1_image, (0, 0))
        combined_image.paste(d2_image, (0, 334))
        combined_image.paste(d3_image, (0, 668))
        combined_image.paste(d4_image, (0, 1002))
        combined_image.paste(d5_image, (0, 1336))
        combined_image.paste(d6_image, (0, 1670))
        combined_image.paste(d7_image, (0, 2004))
        ax.imshow(combined_image)
        ax.set_title(f'#{i}, stages D1/D3: ' + str(stages[0][i-1]) + '/' + str(stages[2][i-1]), fontproperties=font_text,color=colorlist[i-1])
        ax.title.set_size(MEDIUM_SIZE)
        ax.axis('off')
        
        # Minimalistic ruler
        ax_ruler = ax.inset_axes([-0.05, -0.05, 1.1, 0.05])  # Adjusted position 
        ax_ruler.set_ylim(0, 1)
        ax_ruler.set_xticks(range(0, 779, 200))
        ax_ruler.set_xticklabels(range(0, 779, 200)) #one tick per 200 pixels
        ax_ruler.tick_params(axis='y', which='both', left=False, labelleft=False, right=False, labelright=False)
        ax_ruler.tick_params(axis='x', which='both', top=False, bottom=True, labeltop=False, labelbottom=True, labelsize=10)
        
        for tick in range(0, 779, 200): #one tick per 200 pixels
            ax_ruler.plot([tick, tick], [0.2, 0.8], color='black', lw=0.8)
        
        ax_ruler.plot([0, 778], [0.5, 0.5], color='black', lw=0.8)
        ax_ruler.axis('off')

    plt.subplots_adjust(top=0.9, bottom=0.05, left=0.05, right=0.95)
    
    # Add a line of text at the bottom of the figure
    footer_text = ('1 div = 200 px $\cdot$ surviving: ' + str(surviving[0]) + ', ' 
                                                        + str(surviving[1]) + ', ' 
                                                        + str(surviving[2]) + '/24 $\cdot$ avg. stages: ' 
                                                        + str(avg_stages[0]) + '$\pm$' + str(std_stages[0]) + ', ' 
                                                        + str(avg_stages[1]) + '$\pm$' + str(std_stages[1]) + ', '  
                                                        + str(avg_stages[2]) + '$\pm$' + str(std_stages[2]))
    fig.text(0.5, 0, footer_text, ha='center', fontsize=MEDIUM_SIZE, fontproperties=font_text)
    
    # Save the figure
    if save:
        plt.savefig(output_path_pdf, format='pdf', dpi=my_dpi, bbox_inches='tight')
        plt.savefig(output_path_png, format='png', dpi=my_dpi, bbox_inches='tight')
    
    plt.show()
    
    
def find_max_dimension(folders):
    """ Finds the maximum dimension among all images in the provided folders. """
    max_dimension = 0
    for folder in folders:
        for image_name in os.listdir(folder):
            if image_name.endswith('.png'):
                image_path = os.path.join(folder, image_name)
                with Image.open(image_path) as img:
                    if img.width > max_dimension:
                        max_dimension = img.width
                    if img.height > max_dimension:
                        max_dimension = img.height
    return max_dimension

def resize_images_to_square(folders, target_size):
    """ Resize all images in the provided folders to a specified square size with transparent background. """
    for folder in folders:
        for image_name in os.listdir(folder):
            if image_name.endswith('.png'):
                image_path = os.path.join(folder, image_name)
                with Image.open(image_path) as img:
                    # Convert to RGBA if not already to ensure transparency is handled
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')

                    # Create a new image with transparent background
                    new_image = Image.new('RGBA', (target_size, target_size), (255, 255, 255, 0))
                    # Calculate the position to paste the resized image
                    x = (target_size - img.width) // 2
                    y = (target_size - img.height) // 2
                    new_image.paste(img, (x, y))

                    # Save the resized image
                    new_image.save(image_path)
                    
def check_image_sizes(folders, my_batch):
    """Check if all images in the provided folders have the same number of pixels."""
    image_size = None
    for folder in folders:
        for image_name in os.listdir(folder):
            if image_name.endswith('.png'):
                image_path = os.path.join(folder, image_name)
                with Image.open(image_path) as img:
                    current_size = img.size
                    if image_size is None:
                        image_size = current_size
                    elif current_size != image_size:
                        raise ValueError(f"Image {image_path} has a different size: {current_size}")
    print(f"All images in Batch {my_batch} have the same size:", image_size)
    return image_size

def calculate_elongation(major_axis, minor_axis):
    """Calculate the elongation of an object given its major and minor axes."""
    return (major_axis - minor_axis) / (major_axis + minor_axis)

def calculate_roundness(area, perimeter):
    """Calculate the roundness score of an object given its area and perimeter."""
    return (4 * np.pi * area) / (perimeter ** 2)

def calculate_AdivP(area, perimeter):
    """Calculate the roundness score of an object given its area and perimeter."""
    return (area) / (perimeter)

def calculate_eccentricity(major_axis, minor_axis):
    """Calculate the eccentricity of an object given its major and minor axes."""
    return np.sqrt(1 - (minor_axis**2 / major_axis**2))

def plot_D1(fig, num_rows, num_cols, num_images1, num_images2, images_with_overlay, plot_title, footer_txt, framecolors, BIGGER_SIZE, MEDIUM_SIZE, font_title, font_text, rotate_indices_C, rotate_indices_H, hull_images=None):
    
    plt.suptitle(plot_title, fontsize=BIGGER_SIZE, fontproperties=font_title)
    gs = gridspec.GridSpec(num_rows, 2 * num_cols, width_ratios=[1] * (2 * num_cols),  wspace=0.2, hspace=0.02) #wspace=0.02

    thick = 20 #was 10
    # Plot images from the first folder
    for idx in range(num_images1):
        ax = plt.subplot(gs[idx // num_cols, idx % num_cols])
        images_with_overlay[0][idx] = add_frame(images_with_overlay[0][idx], framecolors[0][idx], thick) 
        if hull_images:
            ax.imshow(hull_images[0][idx], cmap='Blues')
            ax.imshow(images_with_overlay[0][idx], alpha = 0.3)
        else:
            ax.imshow(images_with_overlay[0][idx])
        ax.axis('off')

    # Plot images from the second folder
    for idx in range(num_images2):
        ax = plt.subplot(gs[idx // num_cols, num_cols + (idx % num_cols)])
        images_with_overlay[1][idx] = add_frame(images_with_overlay[1][idx], framecolors[1][idx], thick) 
        if hull_images:
            ax.imshow(hull_images[1][idx], cmap='Blues')
            ax.imshow(images_with_overlay[1][idx], alpha = 0.3)
        else:
            ax.imshow(images_with_overlay[1][idx])
        ax.axis('off')
        
    # Add a vertical line exactly in the middle with reduced height
    middle_x = 0.5  # Middle of the figure in normalized coordinates
    line_height = 0.85  # Height of the line as a fraction of the figure height
    line_y_start = 0.05 
    ax_line = fig.add_axes([middle_x - 0.002, line_y_start, 0.004, line_height], frameon=False)
    ax_line.axvline(x=0.5, color='black', linestyle='-', linewidth=5)
    ax_line.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.01, hspace=0.01)

    # Add a line of text at the bottom of the figure
    fig.text(0.5, 0, footer_txt, ha='center', fontsize=MEDIUM_SIZE, fontproperties=font_text)
    
def add_frame(data, frame_color_name, frame_width, alpha_threshold=1e-5):
    
    frame_color = mcolors.to_rgba(frame_color_name, alpha=1)
    frame_color = tuple(int(c * 255) for c in frame_color[:3]) + (255,)
    
    # Get image dimensions
    height, width, channels = data.shape
    
    # Replace the outer W pixels with the frame color
    for x in range(width):
        for y in range(height):
            if (x < frame_width or x >= width - frame_width or
                y < frame_width or y >= height - frame_width):
                if data[y, x, 3] < alpha_threshold:  # Check if pixel alpha is near zero
                    data[y, x] = frame_color  # Set to frame color with full opacity
                    # Debugging: print updated pixel information
                    #print(f"Updated pixel at ({y}, {x}) to {frame_color}")
                    
    return data

    
def plot_violins(fig, axs, concatenated, conditionlabels, stick_colors, xpos, miny, maxy, my_yticks, font_title, font_text, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, hor_lines=None, hor_labels=None, p_array=None):
    # Create a violin plot
    parts = sns.violinplot(data=concatenated, color='grey', inner='stick', linewidth=0.5, fill=True, ax=axs)
    plt.setp(parts.collections, alpha=0.2)

    # Draw vertical lines between each pair of Control and Hypo
    for i in range(1, len(conditionlabels) - 1, 2): 
        axs.axvline(x=i + 0.5, color='black', linestyle='--', linewidth=1)

    #Extract the stick lines (lines corresponding to the individual data points)
    stick_lines = [child for child in axs.get_children() if isinstance(child, LineCollection)]

    # Adjust the color of each stick based on the organized stick_colors
    start_idx = 0
    for batch_idx, (control_colors, hypo_colors) in enumerate(zip(stick_colors[0], stick_colors[1])):
        # Each batch has its own control and hypo colors
        batch_colors = [control_colors, hypo_colors]
        for condition_idx, condition_colors in enumerate(batch_colors):
            segments = stick_lines[start_idx].get_segments()

            # Check if the number of segments matches the number of colors
            if len(segments) != len(condition_colors):
                raise ValueError(f"Mismatch between segments ({len(segments)}) and colors ({len(condition_colors)}) in batch {batch_idx + 1}, condition {condition_idx + 1}")

            # Assign colors to each segment
            for segment, my_color in zip(segments, condition_colors):
                axs.add_collection(LineCollection([segment], colors=[my_color], linewidths=0.5))  # Set the color for each segment

            start_idx += 1

    # Plot avg, median, std for each violin
    for my_idx in range(len(xpos)):
        axs.plot(xpos[my_idx], np.mean(concatenated[my_idx]), marker="o", markersize=7, markeredgecolor="k", markerfacecolor="k",zorder=2)
        axs.plot(xpos[my_idx], np.median(concatenated[my_idx]), marker="x", markersize=9, markeredgecolor="k", markerfacecolor="k",zorder=2)
        axs.vlines(xpos[my_idx], np.mean(concatenated[my_idx])-np.std(concatenated[my_idx]), np.mean(concatenated[my_idx])+np.std(concatenated[my_idx]), color='black', lw=2,zorder=1)
        #lw was 1 I just made it 2 CDA 20240928
       
    # Add legend for mean, std, and median
    legend_elements = [
    Line2D([0], [0], marker='o', color='black', label='mean ± std', 
           markerfacecolor='black', markersize=7, linestyle='-', lw=1),  # Dot with line
    Line2D([0], [0], marker='x', color='black', label='median', markersize=7, linestyle='')
        ]

    # Add the legend to the plot
    legend_font = FontProperties(fname=font_text.get_file(), size=MEDIUM_SIZE)
    # legend = axs.legend(handles=legend_elements, loc='lower center', prop=legend_font, frameon=True)
    
    # Assuming legend_elements is already a list of handles
    legend = axs.legend(
        handles=legend_elements,  # Pass the list directly without wrapping it in another list
        loc='upper right',
        bbox_to_anchor=(1, 1.05),  # Adjust these values as needed
        frameon=True,             # Turn off the frame around the legend
        prop=legend_font,          # Adjust font size as needed
        handletextpad=0.5,         # Spacing between the legend symbol and text
        labelspacing=0.5           # Spacing between rows of the legend
    )
    
    # # Customize the frame (facecolor sets the background color)
    legend.get_frame().set_facecolor('white')  # Set background color to white
    legend.get_frame().set_edgecolor('none')   # Remove the edge of the frame (border)
    legend.get_frame().set_alpha(1.0)     

       
    if hor_lines is not None:
         for i, line_pos in enumerate(hor_lines):
             axs.axhline(y=line_pos, color='red', linestyle='--', linewidth=1)
             # Add text next to the horizontal line
             if hor_labels is not None and i < len(hor_labels):
                 axs.text(xpos[-1] + 0.5, line_pos, hor_labels[i], color='red', va='bottom', ha='left', fontsize=SMALL_SIZE,fontproperties=font_text)

    axs.spines[['right', 'top']].set_visible(False)
    axs.set_xlim(-0.5, len(conditionlabels))  # set x axis limits to appropriate values; fixed
    axs.set_ylim(miny, maxy)  # variable limits for each test
    axs.set_yticks(my_yticks)
    
    # Add custom horizontal lines and text based on p_array
    if p_array is not None:
        # Line between the first and third violins
        y_line1 = maxy # maxy - 0.1 * (maxy - miny)
        axs.hlines(y=y_line1, xmin=0, xmax=2, color='black', linestyle='-', linewidth=1)
        axs.text(1, y_line1 + 0.02 * (maxy - miny), p_array[0], color='black', va='bottom', ha='center', fontsize=MEDIUM_SIZE, fontproperties=font_text)

        # Line between the first and second violins
        y_line2 = maxy - 0.15 * (maxy - miny)
        axs.hlines(y=y_line2, xmin=0, xmax=1, color='black', linestyle='-', linewidth=1)
        axs.text(0.5, y_line2 + 0.02 * (maxy - miny), p_array[1], color='black', va='bottom', ha='center', fontsize=MEDIUM_SIZE, fontproperties=font_text)

        # Line between the third and fourth violins
        y_line3 = maxy - 0.15 * (maxy - miny)
        axs.hlines(y=y_line3, xmin=2, xmax=3, color='black', linestyle='-', linewidth=1)
        axs.text(2.5, y_line3 + 0.02 * (maxy - miny), p_array[2], color='black', va='bottom', ha='center', fontsize=MEDIUM_SIZE, fontproperties=font_text)

    # Apply the font properties to the tick labels
    for my_label in axs.get_xticklabels():
        my_label.set_fontproperties(font_text)
    for my_label in axs.get_yticklabels():
        my_label.set_fontproperties(font_text)
    axs.tick_params(labelsize=MEDIUM_SIZE)
    axs.tick_params(bottom=False)
    # setting ticks for x-axis; fixed
    axs.set_xticks([i + 0.5 for i in range(0, len(conditionlabels), 2)])  # Tick positions between the pairs
    reduced_labels = [label[:-1] for label in conditionlabels[::2]]
    string_array_with_newline = [s + '\n' for s in reduced_labels]
    axs.set_xticklabels(string_array_with_newline) 
    
    # Define arrow and text parameters
    arrowprops = dict(facecolor='black', arrowstyle="->", lw=1)
    # Adjust figure-relative coordinates for arrows
    xposc = 0.025  # Adjust the position for C
    xposh = 0.075   # Adjust the position for H
    yposup_text = 0.89
    yposup = 0.95*maxy
    yposdown = 0.85*maxy
    # Add the arrow at x = xposc, pointing down (relative to the figure)
    axs.annotate('', xy=(0, yposdown), xytext=(0, yposup), arrowprops=arrowprops, transform=axs.transAxes) 
    # Add the arrow at x = xposh, pointing down (relative to the figure)
    axs.annotate('', xy=(1, yposdown), xytext=(1, yposup), arrowprops=arrowprops, transform=axs.transAxes)
    # Define the text box properties with a white background and no visible frame
    bbox_props = dict(boxstyle="round,pad=0.25", edgecolor="none", facecolor="white")
    # Add text with a white background (relative to the figure)
    axs.text(xposc, yposup_text + 0.05, 'C', ha='center', fontsize=MEDIUM_SIZE, font=font_text, transform=axs.transAxes, bbox=bbox_props)
    axs.text(xposh, yposup_text + 0.05, 'H', ha='center', fontsize=MEDIUM_SIZE, font=font_text, transform=axs.transAxes, bbox=bbox_props)
    
    
    plt.subplots_adjust(bottom=0, wspace=1.75, top=0.8)
    fig.tight_layout()
    
def perform_pairwise_tests(concatenated, conditionlabels, path, save=False):
    # Initialize lists to hold the results
    ttest_results = []
    mannwhitney_results = []
    kolmogorov_results = []
    shapiro_results = []

    # Iterate over the pairs of conditions
    for i in range(0, len(concatenated), 2):
        # C and H conditions for the current pair
        C_data = concatenated[i]
        H_data = concatenated[i + 1]
        
        # Perform t-test
        t_stat, t_p_value = ttest_ind(C_data, H_data, equal_var=False)
        ttest_results.append((conditionlabels[i], conditionlabels[i+1], t_stat, t_p_value))
        
        # Perform Mann-Whitney U test
        mw_stat, mw_p_value = mannwhitneyu(C_data, H_data, alternative='two-sided')
        mannwhitney_results.append((conditionlabels[i], conditionlabels[i+1], mw_stat, mw_p_value))
        
        # Perform Kolmogorov-Smirnov test: p_value > alpha ~ 0.05: Distribution is normal
        ks_stat_C, ks_p_value_C = stats.kstest(C_data, 'norm')
        ks_stat_H, ks_p_value_H = stats.kstest(H_data, 'norm')
        kolmogorov_results.append((conditionlabels[i], conditionlabels[i+1], ks_p_value_C, ks_p_value_H))

        # Perform Shapiro-Wilk test: p_value > alpha ~ 0.05: Distribution is normal
        if len(C_data) >= 3:
            shapiro_stat_C, shapiro_p_value_C = stats.shapiro(C_data)
            shapiro_stat_H, shapiro_p_value_H = stats.shapiro(H_data)
            shapiro_results.append((conditionlabels[i], conditionlabels[i+1], shapiro_p_value_C, shapiro_p_value_H))
        else:
            shapiro_results.append((conditionlabels[i], conditionlabels[i+1], np.nan, np.nan))
            

        if save:
            # Save the results to text files for each folder
            with open(path, 'w') as f:
                f.write("T test results (T-stat, p-value):\n")
                f.write(str(ttest_results) + "\n")
                f.write("Mann-Whitney test results (U-stat, p-value):\n")
                f.write(str(mannwhitney_results) + "\n")
                f.write("Kolmogorov normality test results (p-values for C and H):\n")
                f.write(str(kolmogorov_results) + "\n")
                f.write("Shapiro normality test results (p-values for C and H):\n")
                f.write(str(shapiro_results) + "\n")
    
    return ttest_results, mannwhitney_results, kolmogorov_results, shapiro_results

def process_and_save_D2_images(folder1, folder2, alpha_threshold=1e-5):
    """Process images from two folders, align and resize them, and save them back to their original directories."""
    images = []
    
    # Load, rotate, and crop images from both folders
    for folder in [folder1, folder2]:
        for filename in os.listdir(folder):
            if filename.endswith('.png'):
                image_path = os.path.join(folder, filename)
                data = imread(image_path)
                
                # Rotate the image so the longest axis is horizontal
                rotated_image = rotate_to_longest_axis(data, alpha_threshold)
                
                # Find the bounding box of the non-transparent region
                bbox = find_bounding_box(rotated_image, alpha_threshold)
                
                # Crop the image to the bounding box
                cropped_image = crop_to_bounding_box(rotated_image, bbox)
                
                # Store the cropped image, its size, and its path
                images.append((cropped_image, filename, folder))
    
    # Determine the minimal common bounding rectangle size
    max_height = max(img.shape[0] for img, _, _ in images)
    max_width = max(img.shape[1] for img, _, _ in images)
    
    # Pad each image to the common size and save to the original directory
    for img, filename, folder in images:
        padded_image = pad_and_align_left(img, max_height, max_width, alpha_threshold)
        
        # Ensure the image is rotated by 90 degrees before saving
        padded_image = np.rot90(padded_image)
        
        # Save the rotated image back to the original directory
        output_path = os.path.join(folder, filename)
        imsave(output_path, padded_image)
        
def find_bounding_box(data, alpha_threshold=1e-5):
    """Find the bounding box of the non-transparent region."""
    non_transparent = data[..., 3] >= alpha_threshold
    labeled_image = label(non_transparent)
    regions = regionprops(labeled_image)
    if not regions:
        return (0, 0, 0, 0)
    
    minr, minc, maxr, maxc = regions[0].bbox
    return (minr, minc, maxr, maxc)

def rotate_to_longest_axis(data, alpha_threshold=1e-5):
    """Rotate the image so that the longest axis is horizontal."""
    non_transparent = data[..., 3] >= alpha_threshold
    labeled_image = label(non_transparent)
    regions = regionprops(labeled_image)
    if not regions:
        return data
    
    region = max(regions, key=lambda r: r.major_axis_length)
    angle = -np.degrees(region.orientation) if region.orientation < 0 else -np.degrees(region.orientation) + 180
    rotated_image = rotate(data, angle, resize=True, preserve_range=True, mode='constant', cval=0).astype(np.uint8)
    
    return rotated_image

def find_first_non_transparent_pixel(data, alpha_threshold=1e-5):
    """Find the column index of the first non-transparent pixel."""
    for x in range(data.shape[1]):
        if np.any(data[:, x, 3] >= alpha_threshold):
            return x
    return 0  # If no non-transparent pixel is found, return 0

def crop_to_bounding_box(data, bbox):
    """Crop the image to the bounding box."""
    minr, minc, maxr, maxc = bbox
    cropped_image = data[minr:maxr, minc:maxc]
    return cropped_image

def pad_and_align_left(data, target_height, target_width, alpha_threshold=1e-5):
    """Pad the image to the target size with transparent pixels and align it flush left."""
    height, width = data.shape[:2]
    
    # Find the first non-transparent pixel and calculate the shift
    first_non_transparent_col = find_first_non_transparent_pixel(data, alpha_threshold)
    left_padding = -first_non_transparent_col
    
    # Padding: flush left alignment and centered vertically
    pad_height_top = (target_height - height) // 2
    pad_height_bottom = target_height - height - pad_height_top
    pad_width_right = target_width - width - left_padding
    
    padded_image = np.pad(data, ((pad_height_top, pad_height_bottom), 
                                 (left_padding, pad_width_right), 
                                 (0, 0)), mode='constant', constant_values=0)
    return padded_image

def process_folder_D2(folder_path):
    """Process all images in a folder, overlay the longest connected line, and return a sorted list of images, binary images, major axis lengths, and filenames."""
    images = []
    binary_images = []
    major_axis_lengths = []  # List to store the major axis lengths
    filenames = []  # List to store filenames
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            data = imread(image_path)
            data_with_overlay, major_axis_length, binary_image = calculate_longest_connected_line(data)
            images.append(data_with_overlay)
            binary_images.append(binary_image)
            major_axis_lengths.append(major_axis_length)  # Store the major axis length
            filenames.append(filename)  # Store the filename
    
    # Sort by major_axis_lengths
    sorted_indices = np.argsort(major_axis_lengths)
    
    sorted_images = [images[i] for i in sorted_indices]
    sorted_binary_images = [binary_images[i] for i in sorted_indices]
    sorted_major_axis_lengths = [major_axis_lengths[i] for i in sorted_indices]
    sorted_filenames = [filenames[i] for i in sorted_indices]
    
    return sorted_images, sorted_binary_images, sorted_major_axis_lengths, sorted_filenames

def calculate_longest_connected_line(data, alpha_threshold=1e-5):
    """Calculate the approximate length of the longest connected line using the convex hull and overlay it on the image."""
    non_transparent = data[..., 3] >= alpha_threshold
    
    # Convert to a binary image (boolean)
    binary_image = img_as_bool(non_transparent)
    
    # Label connected components in the binary image
    labeled_image = label(binary_image)
    regions = regionprops(labeled_image)
    
    if not regions:
        return data, 0, binary_image
    
    # Find the largest region (by area)
    region = max(regions, key=lambda r: r.area)
    coords = region.coords  # Get the coordinates of the region
    
    if len(coords) < 3:
        return data, 0, binary_image  # Not enough points for a hull
    
    # Compute the convex hull of the region
    hull = ConvexHull(coords)
    
    # Find the two points on the convex hull that are farthest apart
    max_distance = 0
    start = end = None
    for i in range(len(hull.vertices)):
        for j in range(i + 1, len(hull.vertices)):
            point1 = coords[hull.vertices[i]]
            point2 = coords[hull.vertices[j]]
            distance = np.linalg.norm(point1 - point2)
            if distance > max_distance:
                max_distance = distance
                start, end = point1, point2
    
    if start is None or end is None:
        return data, 0, binary_image
    
    y1, x1 = start
    y2, x2 = end
    
    # Draw the major axis (longest line) on the image
    rr, cc = line(y1, x1, y2, x2)
    #Two lines below add the line to the colorized picture if wanted
    #overlay_color = [255, 0, 0, 255]  # Red color with full opacity
    #set_color(data, (rr, cc), overlay_color)
    
    # Draw the major axis as "True" pixels on the binary image
    binary_image[rr, cc] = False

    return data, max_distance, binary_image

def plot_images_custom_aspect(folder1_images, folder2_images, aspect_ratio, plot_title, footer_txt, font_title, font_text, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, alpha_threshold=1e-5):
    """Plot images from two folders with first non-transparent pixel flushleft and custom aspect ratio."""
    num_images1 = len(folder1_images)
    num_images2 = len(folder2_images)
    num_rows = max(num_images1, num_images2)
    
    # Define the overall figure size based on the number of rows and aspect ratio
    fig_height = num_rows * 2
    fig_width = aspect_ratio * 5
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    plt.suptitle(plot_title, fontsize=BIGGER_SIZE, fontproperties=font_title)
    gs = plt.GridSpec(num_rows, 2, width_ratios=[1, 1], height_ratios=[1] * num_rows)

    for i in range(num_rows):
        if i < num_images1:
            first_non_transparent = find_first_non_transparent_pixel(folder1_images[i], alpha_threshold)
            ax1 = fig.add_subplot(gs[i, 0])
            ax1.imshow(folder1_images[i], aspect='auto')
            ax1.set_xlim(first_non_transparent, first_non_transparent + folder1_images[i].shape[1])
            ax1.axis('off')
        else:
            print(f'It is finding an empty subplot for left column, i = {i}')
            ax1 = fig.add_subplot(gs[i, 0])
            ax1.axis('off')

        if i < num_images2:
            first_non_transparent = find_first_non_transparent_pixel(folder2_images[i], alpha_threshold)
            ax2 = fig.add_subplot(gs[i, 1])
            ax2.imshow(folder2_images[i], aspect='auto')
            ax2.set_xlim(first_non_transparent, first_non_transparent + folder2_images[i].shape[1])
            ax2.axis('off')
        else:
            print(f'It is finding an empty subplot for right column, i = {i}')
            ax2 = fig.add_subplot(gs[i, 1])
            ax2.axis('off')
    
    # Add a vertical line exactly in the middle with reduced height
    middle_x = 0.475  # Middle of the figure in normalized coordinates
    line_height = 0.85  # Height of the line as a fraction of the figure height
    line_y_start = 0.05
    ax_line = fig.add_axes([middle_x - 0.002, line_y_start, 0.004, line_height], frameon=False)
    ax_line.axvline(x=0.5, color='black', linestyle='--', linewidth=5)
    ax_line.axis('off')
    
    xpos = [0.05, 0.15, 0.25, 0.35, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]
    for x in xpos:
        add_vert_line(fig, x)
    
    # Add a line of text at the bottom of the figure
    fig.text(0.5, 0, footer_txt, ha='center', fontsize=MEDIUM_SIZE, fontproperties=font_text)
    plt.tight_layout(pad=0)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.925, bottom=0.05, wspace=0.0, hspace=0.0)
    
def add_vert_line(fig, middle_x):
    # Add a vertical line exactly in the middle with reduced height
    line_height = 0.85  # Height of the line as a fraction of the figure height
    line_y_start = 0.05
    ax_line = fig.add_axes([middle_x - 0.002, line_y_start, 0.004, line_height], frameon=False)
    ax_line.axvline(x=0.5, color='black', linestyle=":", linewidth=1)
    ax_line.axis('off')
    
def plot_binary_images(folder1_binaries, folder2_binaries, aspect_ratio, plot_title, footer_txt, font_title, font_text, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE):
    """Plot binary images from two folders in the same sequence."""
    num_images1 = len(folder1_binaries)
    num_images2 = len(folder2_binaries)
    num_rows = max(num_images1, num_images2)
    
    # Define the overall figure size based on the number of rows and aspect ratio
    fig_height = num_rows * 2
    fig_width = aspect_ratio * 5
    
    fig2 = plt.figure(figsize=(fig_width, fig_height))
    plt.suptitle(plot_title + ' $\cdot$ Masks with longest line', fontsize=BIGGER_SIZE, fontproperties=font_title)
    gs = plt.GridSpec(num_rows, 2, width_ratios=[1, 1], height_ratios=[1] * num_rows)

    for i in range(num_rows):
        if i < num_images1:
            ax1 = fig2.add_subplot(gs[i, 0])
            ax1.imshow(folder1_binaries[i], cmap='gray', aspect='auto')
            ax1.axis('off')
        else:
            ax1 = fig2.add_subplot(gs[i, 0])
            ax1.axis('off')

        if i < num_images2:
            ax2 = fig2.add_subplot(gs[i, 1])
            ax2.imshow(folder2_binaries[i], cmap='gray', aspect='auto')
            ax2.axis('off')
        else:
            ax2 = fig2.add_subplot(gs[i, 1])
            ax2.axis('off')
            
    fig2.text(0.5, 0, footer_txt, ha='center', fontsize=MEDIUM_SIZE, fontproperties=font_text)
    plt.tight_layout(pad=0)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.925, bottom=0.05, wspace=0.0, hspace=0.0)

def calculate_global_threshold(images):
    """Calculate a global threshold based on the combined histogram of all images."""
#Global Threshold Based on Histogram Analysis
#What it is: Analyze the histograms of all the images combined to find a single threshold value that works reasonably well across the entire set.
#Fairness Consideration: This approach provides a consistent threshold across all images, making comparisons straightforward. However, it might not perform well on all images if there's significant variability.
#Implementation:
#Combine the histograms of all images.
#Choose a threshold value based on the combined histogram, such as a fixed value (e.g., 128) or the midpoint between two peaks in the histogram."
    combined_histogram = np.zeros(256)  # Assuming 8-bit grayscale images
    
    # Loop through each image to build the combined histogram
    for image in images:
        if image.shape[2] == 4:  # Check if the image has an alpha channel
            rgb = image[..., :3]  # Extract the RGB channels
        else:
            rgb = image
        
        grayscale_image = rgb2gray(rgb) * 255  # Convert to 8-bit grayscale
        grayscale_image = grayscale_image.astype(np.uint8)  # Convert to integer type
        histogram, _ = np.histogram(grayscale_image, bins=256, range=(0, 256))
        combined_histogram += histogram
    
    # Plot the combined histogram for visualization if wanted
# =============================================================================
#     plt.figure(figsize=(10, 6))
#     plt.plot(combined_histogram, label='Combined Histogram')
#     plt.xlabel('Pixel Intensity')
#     plt.ylabel('Frequency')
#     plt.title('Combined Histogram of All Images')
#     plt.legend()
#     plt.show()
# =============================================================================

    # Calculate meaningful threshold
    # Option 1: Use the mean of the histogram
    mean_threshold = np.mean(np.nonzero(combined_histogram))
    
    # Option 2: Use the midpoint between peaks (if multiple peaks are present)
    peaks = np.where((combined_histogram[1:-1] > combined_histogram[:-2]) & 
                     (combined_histogram[1:-1] > combined_histogram[2:]))[0] + 1
    
    if len(peaks) > 1:
        midpoint_threshold = (peaks[0] + peaks[-1]) // 2
    else:
        midpoint_threshold = mean_threshold
    
    # Choose the best threshold to use
    chosen_threshold = int(midpoint_threshold)
    
    return chosen_threshold

def apply_fixed_threshold(image, threshold, alpha_threshold=1e-5):
    """Apply a fixed binary threshold to an image while neglecting the transparency."""
    # Separate the RGB and alpha channels
    rgb = image[..., :3]
    alpha = image[..., 3]
    
    # Convert RGB to grayscale
    grayscale = rgb2gray(rgb) * 255  # Convert to 8-bit grayscale (range 0-255)
    
    # Apply the fixed threshold (like Threshold 128 in Photoshop)
    binary_image = grayscale >= threshold
    
    # Mask out transparent pixels based on the alpha channel
    binary_image[alpha < alpha_threshold] = False
    
    return binary_image

def plot_binary_images_custom_aspect(folder1_images, folder2_images, aspect_ratio, plot_title, footer_txt, font_title, font_text, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE):
    """Plot binary images from two folders with first non-transparent pixel flushleft and custom aspect ratio."""
    num_images1 = len(folder1_images)
    num_images2 = len(folder2_images)
    num_rows = max(num_images1, num_images2)
    
    # Define the overall figure size based on the number of rows and aspect ratio
    fig_height = num_rows * 2
    fig_width = aspect_ratio * 5
    
    fig4 = plt.figure(figsize=(fig_width, fig_height))
    plt.suptitle(plot_title + ' $\cdot$ Binarized images', fontsize=BIGGER_SIZE, fontproperties=font_title)
    gs4 = plt.GridSpec(num_rows, 2, width_ratios=[1, 1], height_ratios=[1] * num_rows)

    for i in range(num_rows):
        if i < num_images1:
            ax1 = fig4.add_subplot(gs4[i, 0])
            ax1.imshow(folder1_images[i], cmap='gray', aspect='auto')  # Use 'gray' colormap for binary images
            ax1.axis('off')
        else:
            ax1 = fig4.add_subplot(gs4[i, 0])
            ax1.axis('off')

        if i < num_images2:
            ax2 = fig4.add_subplot(gs4[i, 1])
            ax2.imshow(folder2_images[i], cmap='gray', aspect='auto')  # Use 'gray' colormap for binary images
            ax2.axis('off')
        else:
            ax2 = fig4.add_subplot(gs4[i, 1])
            ax2.axis('off')
    
    # Add a vertical line exactly in the middle with reduced height
    middle_x = 0.475  # Middle of the figure in normalized coordinates
    line_height = 0.85  # Height of the line as a fraction of the figure height
    line_y_start = 0.05
    ax_line = fig4.add_axes([middle_x - 0.002, line_y_start, 0.004, line_height], frameon=False)
    ax_line.axvline(x=0.5, color='black', linestyle='--', linewidth=5)
    ax_line.axis('off')
    
    #If want to add vertical lines for length comparison:
# =============================================================================
#     xpos = [0.05, 0.15, 0.25, 0.35, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]
#     for x in xpos:
#         add_vert_line(fig4, x)
# =============================================================================
    
    # Add a line of text at the bottom of the figure
    fig4.text(0.5, 0, footer_txt, ha='center', fontsize=MEDIUM_SIZE, fontproperties=font_text)
    plt.tight_layout(pad=0)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.925, bottom=0.05, wspace=0.0, hspace=0.0)

def plot_false_features_custom_aspect(folder1_images, folder2_images, aspect_ratio, plot_title, footer_txt, font_title, font_text, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE):
    """Plot binary images from two folders with 'False' features overlaid, with first non-transparent pixel flushleft and custom aspect ratio."""
    num_images1 = len(folder1_images)
    num_images2 = len(folder2_images)
    num_rows = max(num_images1, num_images2)
    
    # Define the overall figure size based on the number of rows and aspect ratio
    fig_height = num_rows * 2
    fig_width = aspect_ratio * 5
    
    fig5 = plt.figure(figsize=(fig_width, fig_height))
    plt.suptitle(plot_title, fontsize=BIGGER_SIZE, fontproperties=font_title)
    gs5 = plt.GridSpec(num_rows, 2, width_ratios=[1, 1], height_ratios=[1] * num_rows)

    areasC = [] # List to store areas of connected features for all images
    areasH = []

    for i in range(num_rows):
        if i < num_images1:
            ax1 = fig5.add_subplot(gs5[i, 0])
            colored_false_features, areas1 = overlay_sorted_false_features(folder1_images[i])
            ax1.imshow(folder1_images[i], cmap='gray', aspect='auto')  # Use 'gray' colormap for binary images
            ax1.imshow(colored_false_features, alpha=0.6, aspect='auto')  # Overlay the colored connected components
            ax1.axis('off')
            areasC.append(areas1)
        else:
            ax1 = fig5.add_subplot(gs5[i, 0])
            ax1.axis('off')

        if i < num_images2:
            ax2 = fig5.add_subplot(gs5[i, 1])
            colored_false_features, areas2 = overlay_sorted_false_features(folder2_images[i])
            ax2.imshow(folder2_images[i], cmap='gray', aspect='auto')  # Use 'gray' colormap for binary images
            ax2.imshow(colored_false_features, alpha=0.6, aspect='auto')  # Overlay the colored connected components
            ax2.axis('off')
            areasH.append(areas1)
        else:
            ax2 = fig5.add_subplot(gs5[i, 1])
            ax2.axis('off')
    
    # Add a vertical line exactly in the middle with reduced height
    middle_x = 0.475  # Middle of the figure in normalized coordinates
    line_height = 0.85  # Height of the line as a fraction of the figure height
    line_y_start = 0.05
    ax_line = fig5.add_axes([middle_x - 0.002, line_y_start, 0.004, line_height], frameon=False)
    ax_line.axvline(x=0.5, color='black', linestyle='--', linewidth=5)
    ax_line.axis('off')
    
    #If want to add vertical lines for length comparison:
# =============================================================================
#     xpos = [0.05, 0.15, 0.25, 0.35, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]
#     for x in xpos:
#         add_vert_line(fig4, x)
# =============================================================================
    
    # Add a line of text at the bottom of the figure
    fig5.text(0.5, 0, footer_txt, ha='center', fontsize=MEDIUM_SIZE, fontproperties=font_text)
    plt.tight_layout(pad=0)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.925, bottom=0.05, wspace=0.0, hspace=0.0)
    
    return areasC, areasH

def overlay_sorted_false_features(image):
    inverted_image = ~image
    labeled_false_features = label(inverted_image, connectivity=2)
    regions = regionprops(labeled_false_features)
        
    # Sort regions by area in decreasing order
    regions_sorted = sorted(regions, key=lambda r: r.area, reverse=True)
        
    # Extract the areas in decreasing order
    areas = [region.area for region in regions_sorted]
        
    # Create a blank RGB image for color overlay
    overlay_image = np.zeros((*image.shape, 3), dtype=np.float32)
        
    # Define the custom color order
    color_order = ['black','red', 'blue', 'green', 'yellow', 'cyan', 'magenta']
    
    # Assign colors to sorted regions
    for idx, region in enumerate(regions_sorted):
        #color_map = plt.colors.to_rgb(color_order[idx % len(color_order)])  #plt.cm.get_cmap('tab20')
        region_color = mcolors.to_rgb(color_order[idx % len(color_order)])   # Get a color from the colormap
        overlay_image[labeled_false_features == region.label] = region_color
    
    return overlay_image, areas

def calculate_mean_yellow_colors(images):
    """Calculate the mean yellow color for a list of images and return images with yellow bands highlighted.
    
    Args:
        images (list): A list of images (as numpy arrays).
        
    Returns:
        tuple: A tuple containing two lists:
               - List of mean yellow RGB values for each image.
               - List of images with yellow bands highlighted.
    """
    mean_yellow_colors = []
    yellow_band_images = []

    # Define the RGB range for yellow
    lower_yellow = np.array([100, 87, 0])   # Lower bound for RGB yellow
    upper_yellow = np.array([255, 255, 150]) # Upper bound for RGB yellow
    # Before, I was using as bands the followinh *HSV* values:
    # Lower Yellow (HSV: [20, 100, 100]) / RGB: [100, 87, 61]
    # Upper Yellow (HSV: [30, 255, 255]) / RGB: [255, 255, 0]
    
    for idx, image in enumerate(images):
        # Check if the image has 4 channels (RGBA) and convert to RGB
        if image.shape[-1] == 4:
            image = image[..., :3]  # Drop the alpha channel
        
        # Create a binary mask for the yellow color based on RGB thresholds
        mask = np.all(image >= lower_yellow, axis=-1) & np.all(image <= upper_yellow, axis=-1)
        
        # Apply the mask to the original image to highlight yellow bands
        yellow_parts = np.zeros_like(image)
        yellow_parts[mask] = image[mask]
        
        # Calculate the mean color in the masked region
        if np.any(mask):
            mean_yellow_color_rgb = np.mean(image[mask], axis=0)
        else:
            mean_yellow_color_rgb = np.array([0, 0, 0])  # No yellow detected
        
        # Append the mean yellow color to the list
        mean_yellow_colors.append(mean_yellow_color_rgb)
        
        # Store the image with yellow bands highlighted
        yellow_band_images.append(yellow_parts)
        
# =============================================================================
#         # Display the original image and the image with the yellow regions highlighted
#         plt.figure(figsize=(10, 5))
#         plt.subplot(1, 2, 1)
#         imshow(image)
#         plt.title(f"Original Image {idx+1}")
#         
#         plt.subplot(1, 2, 2)
#         imshow(yellow_parts)
#         plt.title(f"Yellow Regions Highlighted {idx+1}")
#         
#         plt.show()
# =============================================================================

    return mean_yellow_colors, yellow_band_images

def plot_yellow_bands_custom_aspect(folder1_images, folder2_images, aspect_ratio, plot_title, footer_txt, font_title, font_text, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, threshold=1):
    """Plot images from two folders with first non-black pixel flushleft and custom aspect ratio."""
    num_images1 = len(folder1_images)
    num_images2 = len(folder2_images)
    num_rows = max(num_images1, num_images2)
    
    # Define the overall figure size based on the number of rows and aspect ratio
    fig_height = num_rows * 2
    fig_width = aspect_ratio * 5
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    plt.suptitle(plot_title, fontsize=BIGGER_SIZE, fontproperties=font_title)
    gs = plt.GridSpec(num_rows, 2, width_ratios=[1, 1], height_ratios=[1] * num_rows)

    for i in range(num_rows):
        if i < num_images1:
            first_non_black = find_first_non_black_pixel(folder1_images[i], threshold)
            ax1 = fig.add_subplot(gs[i, 0])
            ax1.imshow(folder1_images[i], aspect='auto')
            ax1.set_xlim(first_non_black, first_non_black + folder1_images[i].shape[1])
            ax1.axis('off')
        else:
            ax1 = fig.add_subplot(gs[i, 0])
            ax1.axis('off')

        if i < num_images2:
            first_non_black = find_first_non_black_pixel(folder2_images[i], threshold)
            ax2 = fig.add_subplot(gs[i, 1])
            ax2.imshow(folder2_images[i], aspect='auto')
            ax2.set_xlim(first_non_black, first_non_black + folder2_images[i].shape[1])
            ax2.axis('off')
        else:
            ax2 = fig.add_subplot(gs[i, 1])
            ax2.axis('off')
    
    # Add a vertical line exactly in the middle with reduced height
    middle_x = 0.475  # Middle of the figure in normalized coordinates
    line_height = 0.85  # Height of the line as a fraction of the figure height
    line_y_start = 0.05
    ax_line = fig.add_axes([middle_x - 0.002, line_y_start, 0.004, line_height], frameon=False)
    ax_line.axvline(x=0.5, color='black', linestyle='--', linewidth=5)
    ax_line.axis('off')
    
# =============================================================================
#     xpos = [0.05, 0.15, 0.25, 0.35, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]
#     for x in xpos:
#         add_vert_line(fig, x)
# =============================================================================
    
    # Add a line of text at the bottom of the figure
    fig.text(0.5, 0, footer_txt, ha='center', fontsize=MEDIUM_SIZE, fontproperties=font_text)
    plt.tight_layout(pad=0)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.925, bottom=0.05, wspace=0.0, hspace=0.0)

def find_first_non_black_pixel(image, threshold=1):
    """Find the first non-black pixel in an RGB image."""
    for x in range(image.shape[1]):
        if np.any(np.mean(image[:, x, :], axis=-1) >= threshold):
            return x
    return 0  # Return 0 if no non-black pixel is found

def ANOVAcomparisons(concatenated):
    
    # ANOVA and Kruskal-Wallis for Controls and Hypo separately
    controls = [concatenated[i] for i in range(0, len(concatenated), 2)]
    hypos = [concatenated[i] for i in range(1, len(concatenated), 2)]
    
    f_statistic_C, p_value_C = stats.f_oneway(*controls)
    kw_statistic_C, p_value_kw_C = stats.kruskal(*controls)
    
    f_statistic_H, p_value_H = stats.f_oneway(*hypos)
    kw_statistic_H, p_value_kw_H = stats.kruskal(*hypos)
    
    my_alpha = 0.01 #The confidence intervals for the mean differences between groups are also based on the 95% confidence level.
    
    # Flatten the list of arrays into a single list of values
    values_C = [item for sublist in controls for item in sublist]
    # Create group labels corresponding to each array in 'a'
    groups_C = []
    for i, group in enumerate(controls):
        #groups_C.extend([f'Batch{i+1}C'] * len(group)) # gives wrong order if batch >= 10
        groups_C.extend([f'Batch{i+1:02}C'] * len(group))
    # Create a DataFrame
    data_C = pd.DataFrame({'value': values_C, 'group': groups_C})
    # Perform Tukey's HSD test
    tukey_C = pairwise_tukeyhsd(endog=data_C['value'], groups=data_C['group'], alpha=my_alpha)
    # Display the results
    #print(tukey_C)    
    
    # Flatten the list of arrays into a single list of values
    values_H = [item for sublist in hypos for item in sublist]
    # Create group labels corresponding to each array in 'a'
    groups_H = []
    for i, group in enumerate(hypos):
        #groups_H.extend([f'Batch{i+1}H'] * len(group))
        groups_H.extend([f'Batch{i+1:02}H'] * len(group))
    # Create a DataFrame
    data_H = pd.DataFrame({'value': values_H, 'group': groups_H})
    # Perform Tukey's HSD test
    tukey_H = pairwise_tukeyhsd(endog=data_H['value'], groups=data_H['group'], alpha=my_alpha)
    # Display the results
    #print(tukey_H)    
    
    # Kruskal-Wallis post-hoc test, Dunn
    dunn_results_C = sp.posthoc_dunn(data_C, val_col='value', group_col='group', p_adjust='bonferroni')
    dunn_results_H = sp.posthoc_dunn(data_H, val_col='value', group_col='group', p_adjust='bonferroni')
    pd.set_option('display.max_columns', None)
    #print(dunn_results_C) 
    #print(dunn_results_H) 
    
    # Running 2-waqy ANOVA
    # Flatten the control and hypo data
    control_values = np.concatenate(controls)
    hypo_values = np.concatenate(hypos)

    # Create a condition label for each data point
    control_labels = ['Control'] * len(control_values)
    hypo_labels = ['Hypo'] * len(hypo_values)

    # Create a correct batch label for each data point
    # Assuming each element of `controls` and `hypos` corresponds to a batch
    #batch_labels_controls = [f'Batch{i + 1}' for i in range(len(controls)) for _ in range(len(controls[i]))]
    #batch_labels_hypos = [f'Batch{i + 1}' for i in range(len(hypos)) for _ in range(len(hypos[i]))]
    batch_labels_controls = [f'Batch{i+1:02}' for i in range(len(controls)) for _ in range(len(controls[i]))]
    batch_labels_hypos = [f'Batch{i+1:02}' for i in range(len(hypos)) for _ in range(len(hypos[i]))]

    # Combine all data into a DataFrame
    data = pd.DataFrame({
        'Value': np.concatenate([control_values, hypo_values]),
        'Condition': control_labels + hypo_labels,
        'Batch': batch_labels_controls + batch_labels_hypos
        })
    
    # Fit the two-way ANOVA model
    model = ols('Value ~ C(Condition) + C(Batch) + C(Condition):C(Batch)', data=data).fit()

    # Perform the ANOVA
    anova_table = sm.stats.anova_lm(model, typ=2)
    #print(anova_table)
    
    # If the two-way ANOVA is significant for Condition, run Tukey HSD
    tukey_hsd_condition = multi.pairwise_tukeyhsd(endog=data['Value'], groups=data['Condition'], alpha=my_alpha)
    #print(tukey_hsd_condition)
    # If the two-way ANOVA is significant for Batch, run Tukey HSD for Batch
    tukey_hsd_batch = multi.pairwise_tukeyhsd(endog=data['Value'], groups=data['Batch'], alpha=my_alpha)
    #print(tukey_hsd_batch)
    
    # String to store all the results
    tukey_interaction_results = "Tukey HSD for interaction of Condition and Batch:\n"
    # If the interaction is significant, you may need to run Tukey HSD for each level of the other factor
    for condition in data['Condition'].unique():
        subset = data[data['Condition'] == condition]
        tukey_hsd_interaction = multi.pairwise_tukeyhsd(endog=subset['Value'], groups=subset['Batch'], alpha=0.01)
        # Add the condition title
        tukey_interaction_results += f"\nTukey HSD for {condition}:\n"
        # Convert the Tukey HSD summary to a string and add to the results string
        tukey_interaction_results += tukey_hsd_interaction.summary().as_text() + "\n"    
        
    ##### ART ANOVA
    # Step 1: Align data manually for each effect
    data['RankedValue'] = rankdata(data['Value'])  # Rank the data globally

    # Step 2: Fit the aligned ANOVA model on ranked data
    model = ols('RankedValue ~ C(Condition) + C(Batch) + C(Condition):C(Batch)', data=data).fit()

    # Step 3: Perform ANOVA on the ranked data
    ARTanova_table = sm.stats.anova_lm(model, typ=2)
    #print(ARTanova_table)
    
    tukey_ARTanova = pairwise_tukeyhsd(data['RankedValue'], data['Condition'],alpha=0.01)
    #print(tukey_ARTanova)
    
    # Create an interaction term for Condition and Batch
    data['Condition_Batch'] = data['Condition'] + "_" + data['Batch']

    # Apply Tukey's HSD for interaction between Condition and Batch
    #tukey_interaction_ARTANOVA = pairwise_tukeyhsd(data['RankedValue'], data['Condition_Batch'], alpha=0.01)
    tukey_interaction_ARTANOVA = perform_tukey_with_precision(data['RankedValue'], data['Condition_Batch'], alpha=0.01)

    # Print the results
    #print(tukey_interaction_ARTANOVA)

    
    # ##### Scheirer-Ray-Hare (SRH) Test: didn't really work
    
        
    return f_statistic_C, p_value_C, tukey_C, kw_statistic_C, p_value_kw_C, dunn_results_C, f_statistic_H, p_value_H, tukey_H, kw_statistic_H, p_value_kw_H, dunn_results_H, anova_table, tukey_hsd_condition, tukey_hsd_batch, tukey_interaction_results, ARTanova_table, tukey_ARTanova, tukey_interaction_ARTANOVA

def perform_tukey_with_precision(data, groups, alpha=0.01):
    """
    Perform Tukey HSD with higher precision for small p-values using MultiComparison.
    
    Parameters:
    - data: The data values to compare (e.g., 'RankedValue').
    - groups: The group labels (e.g., 'Condition_Batch').
    - alpha: The significance level.
    
    Returns:
    TukeyHSDResults: The Tukey HSD results object with high precision p-values.
    """
    # Use MultiComparison from statsmodels to perform Tukey HSD
    comparison = mc.MultiComparison(data, groups)
    tukey_result = comparison.tukeyhsd(alpha=alpha)
    
    # Adjust small p-values to prevent them from being exactly zero
    for i, p_adj in enumerate(tukey_result.pvalues):
        if p_adj == 0:
            tukey_result.pvalues[i] = np.finfo(float).eps  # Replace zero with smallest float value
    
    return tukey_result



def stats_and_save(save_avg_path, concatenated, even_arrays, odd_arrays, batches_str, conditionlabels):
    
    # Save the results to text files for each folder
    with open(save_avg_path, 'w') as f:
        f.write("Avg Control:\n")
        f.write(str(np.mean(even_arrays)) + "\n")
        f.write("Median Control:\n")
        f.write(str(np.median(even_arrays)) + "\n")
        f.write("Std Control:\n")
        f.write(str(np.std(even_arrays)) + "\n\n")
        
        f.write("Avg Hypo:\n")
        f.write(str(np.mean(odd_arrays)) + "\n")
        f.write("Median Hypo:\n")
        f.write(str(np.median(odd_arrays)) + "\n")
        f.write("Std Hypo:\n")
        f.write(str(np.std(odd_arrays)) + "\n\n\n")
        
        f.write("Cohen effect size, pooled std:\n")
        mean1 = np.mean(even_arrays)
        mean2 = np.mean(odd_arrays)
        std1 = np.std(even_arrays, ddof=1)
        std2 = np.std(odd_arrays, ddof=1)
        n1 = len(even_arrays)
        n2 = len(odd_arrays)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        cohen_d = (mean1 - mean2) / pooled_std
        f.write(str(cohen_d) + "\n")      
        f.write("Effect visibility, in percent:\n")
        f.write(str(100.0* (np.mean(even_arrays) - np.mean(odd_arrays))/(np.mean(even_arrays) + np.mean(odd_arrays))) + "\n")
        f.write("True Binhi effect size, in percent:\n")
        sem_even = np.std(even_arrays) / np.sqrt(len(even_arrays))
        sem_odd  = np.std(odd_arrays) / np.sqrt(len(odd_arrays))
        Eorig = 100 * np.abs(np.mean(even_arrays) - np.mean(odd_arrays)) / np.sqrt(np.mean(even_arrays)**2 + sem_even**2 + sem_odd**2)
        f.write(str(Eorig) + "\n")
        
        f.write("Cohen effect size, one std:\n")
        f.write(str((np.mean(even_arrays) - np.mean(odd_arrays))/np.std(np.concatenate((even_arrays,odd_arrays)))) + "\n")
        f.write("Modified Binhi effect size 1, in percent:\n")
        sem_even = np.std(even_arrays) / np.sqrt(len(even_arrays))
        sem_odd  = np.std(odd_arrays) / np.sqrt(len(odd_arrays))
        E = 100 * np.abs(np.mean(even_arrays) - np.mean(odd_arrays)) / np.sqrt(np.mean(even_arrays)**2 + np.mean(odd_arrays)**2 + sem_even**2 + sem_odd**2)
        f.write(str(E) + "\n")
        f.write("Modified Binhi effect size 2, in percent:\n")
        sem_even = np.std(even_arrays) 
        sem_odd  = np.std(odd_arrays) 
        Em = 100 * np.abs(np.mean(even_arrays) - np.mean(odd_arrays)) / np.sqrt(np.mean(even_arrays)**2 + np.mean(odd_arrays)**2 + sem_even**2 + sem_odd**2)
        f.write(str(Em) + "\n")

        
        ttest_results, mannwhitney_results, kolmogorov_results, shapiro_results = perform_pairwise_tests([even_arrays, odd_arrays], ['C', 'H'], save_avg_path, save=False)
        f.write(f"Stats for all {batches_str} batches combined:\n\n")
        f.write("T test results (if normal distributions) (T-stat, p-value):\n")
        for result in ttest_results:
            f.write(str(result) + "\n")  # Add a newline after each entry
        f.write("Mann-Whitney test results (if non-normal distributions) (U-stat, p-value):\n")
        for result in mannwhitney_results:
            f.write(str(result) + "\n")  # Add a newline after each entry
        f.write("Kolmogorov normality test results (p-values for C and H); if p-value > 0.05 or 0.01: distribution is normal:\n")
        for result in kolmogorov_results:
            f.write(str(result) + "\n")  # Add a newline after each entry
        f.write("Shapiro normality test results (p-values for C and H); if p-value > 0.05 or 0.01: distribution is normal:\n")
        for result in shapiro_results:
            f.write(str(result) + "\n")  # Add a newline after each entry
        f.write("\n\n\n")
        
        ttest_results_2, mannwhitney_results_2, kolmogorov_results_2, shapiro_results_2 = perform_pairwise_tests(concatenated, conditionlabels, save_avg_path, save=False)
        f.write("Stats for each batch:\n\n")
        f.write("T test results (T-stat, p-value):\n")
        for result in ttest_results_2:
            f.write(str(result) + "\n")  # Add a newline after each entry
        f.write("Mann-Whitney test results (U-stat, p-value):\n")
        for result in mannwhitney_results_2:
            f.write(str(result) + "\n")  # Add a newline after each entry
        f.write("Kolmogorov normality test results (p-values for C and H); if p-value > 0.05 or 0.01: distribution is normal:\n")
        for result in kolmogorov_results_2:
            f.write(str(result) + "\n")  # Add a newline after each entry
        f.write("Shapiro normality test results (p-values for C and H); if p-value > 0.05 or 0.01: distribution is normal:\n")
        for result in shapiro_results_2:
            f.write(str(result) + "\n")  # Add a newline after each entry
        f.write("\n\n\n")
        
        f_statistic_C, p_value_C, tukey_C, kw_statistic_C, p_value_kw_C, dunn_results_C, f_statistic_H, p_value_H, tukey_H, kw_statistic_H, p_value_kw_H, dunn_results_H, anova_table, tukey_hsd_condition, tukey_hsd_batch, tukey_interaction_results, ARTanova_table, ARTANOVAtukey, tukey_interaction_ARTANOVA = ANOVAcomparisons(concatenated)
        f.write("ANOVA (if normal distribution) Controls (fstat, p-value, Tukey):\n")
        f.write(str(f_statistic_C) + "\n")
        f.write(str(p_value_C) + "\n")
        f.write(str(tukey_C) + "\n")
        f.write('If True, it means that there is a statistically significant difference between the means of the two groups.\n\n')
        
        f.write("Kruskal-Wallis (if non-normal distribution) Controls (statistic, p_value, Dunn p-value):\n")
        f.write(str(kw_statistic_C) + "\n")
        f.write(str(p_value_kw_C) + "\n")
        f.write(str(dunn_results_C.to_string(index=True, header=True)) + "\n")
        f.write('If < 0.05 or 0.01, it means that there is a statistically significant difference between the two groups.\n\n\n\n\n')
        
        f.write("ANOVA (if normal distribution) Hypo (fstat, p_value):\n")
        f.write(str(f_statistic_H) + "\n")
        f.write(str(p_value_H) + "\n")
        f.write(str(tukey_H) + "\n")
        f.write('If True, it means that there is a statistically significant difference between the means of the two groups.\n\n')
        
        f.write("Kruskal-Wallis (if non-normal distribution) Hypos (statistic, p-value, Dunn p-value):\n")
        f.write(str(kw_statistic_H) + "\n")
        f.write(str(p_value_kw_H) + "\n")
        f.write(str(dunn_results_H.to_string(index=True, header=True)) + "\n")
        f.write('If < 0.05 or 0.01, it means that there is a statistically significant difference between the two groups.\n\n\n\n\n')
        
        f.write("2-way ANOVA table:\n")
        f.write(str(anova_table) + "\n")
        f.write("Tukey HSD for Condition (Hypo or Control):\n")
        f.write(str(tukey_hsd_condition) + "\n")
        f.write("Tukey HSD for Batch:\n")
        f.write(str(tukey_hsd_batch) + "\n\n\n")
        f.write(tukey_interaction_results + "\n\n\n")
        
        f.write("ART-ANOVA table:\n")
        f.write(str(ARTanova_table) + "\n") 
        f.write("Tukey for Condition (Hypo or Control) using ART-ANOVA:\n")
        f.write(str(ARTANOVAtukey) + "\n")
        f.write("Tukey Interaction using ART-ANOVA:\n")
        f.write(str(tukey_interaction_ARTANOVA) + "\n")
        
        return dunn_results_C, dunn_results_H, kw_statistic_C, p_value_kw_C, kw_statistic_H, p_value_kw_H, ARTanova_table, tukey_interaction_ARTANOVA
        
def check_single_true(D1, D2, D3):
    # Count the number of True values
    true_count = sum([D1, D2, D3])

    # Check if exactly one is True
    if true_count != 1:
        raise ValueError("Only one of D1, D2, or D3 can be True at a time.")

def extract_mean_quantity(directory_path, string_quantity,initial_stage, end_stage):
    """
    Extract the values associated with the specified string from text files
    in the given directory.
    """
    mean_quantity = []

    for i in range(initial_stage, end_stage+1):  # Loop through Stage18 to Stage25
        filename = f"Stage{i}_analysis.txt"
        file_path = os.path.join(directory_path, filename)
        
        try:
            with open(file_path, 'r') as file:
                file_lines = file.readlines()
                for j, file_line in enumerate(file_lines):
                    if string_quantity in file_line:
                        # Debugging: Print the exact line that is causing issues
                        #print(f"Found line in {filename}: {file_line.strip()}")
                        
                        # Check if the next line contains the value
                        if j + 1 < len(file_lines):
                            next_line = file_lines[j + 1].strip()
                            if next_line:
                                try:
                                    # Remove the brackets and convert to float
                                    value = float(next_line.strip('[]'))
                                    mean_quantity.append(value)
                                    #print(f"Successfully extracted {string_quantity} from {filename}: {value}")
                                except ValueError:
                                    print(f"Warning: Could not convert value to float in file {filename} for line: {next_line}")
                            else:
                                print(f"Warning: Next line is empty in file {filename} for line: {file_line.strip()}")
                        else:
                            print(f"Warning: No following line for value in file {filename} for line: {file_line.strip()}")
                        break
                else:
                    print(f"Warning: Did not find expected quantity '{string_quantity}' in file {filename}")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    return mean_quantity

def is_increasing(arr):
    return all(arr[i] > arr[i - 1] for i in range(1, len(arr)))

def is_decreasing(arr):
    return all(arr[i] < arr[i - 1] for i in range(1, len(arr)))

def stats_and_save_minibatch(save_avg_path, concatenated, even_arrays, odd_arrays, batches_str, conditionlabels):
    
    # Save the results to text files for each folder
    with open(save_avg_path, 'w') as f:
        f.write("Avg Control:\n")
        f.write(str(np.mean(even_arrays)) + "\n")
        f.write("Median Control:\n")
        f.write(str(np.median(even_arrays)) + "\n")
        f.write("Std Control:\n")
        f.write(str(np.std(even_arrays)) + "\n\n")
        
        f.write("Avg Hypo:\n")
        f.write(str(np.mean(odd_arrays)) + "\n")
        f.write("Median Hypo:\n")
        f.write(str(np.median(odd_arrays)) + "\n")
        f.write("Std Hypo:\n")
        f.write(str(np.std(odd_arrays)) + "\n\n\n")
        
        f.write("Cohen effect size, one std:\n")
        f.write(str((np.mean(even_arrays) - np.mean(odd_arrays))/np.std(np.concatenate((even_arrays,odd_arrays)))) + "\n")
        f.write("Cohen effect size, pooled std:\n")
        mean1 = np.mean(even_arrays)
        mean2 = np.mean(odd_arrays)
        std1 = np.std(even_arrays, ddof=1)
        std2 = np.std(odd_arrays, ddof=1)
        n1 = len(even_arrays)
        n2 = len(odd_arrays)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        cohen_d = (mean1 - mean2) / pooled_std
        f.write(str(cohen_d) + "\n")      
        f.write("Effect visibility, in percent:\n")
        f.write(str(100.0* (np.mean(even_arrays) - np.mean(odd_arrays))/(np.mean(even_arrays) + np.mean(odd_arrays))) + "\n")
        f.write("Binhi effect size, in percent:\n")
        sem_even = np.std(even_arrays) / np.sqrt(len(even_arrays))
        sem_odd  = np.std(odd_arrays) / np.sqrt(len(odd_arrays))
        E = 100 * np.abs(np.mean(even_arrays) - np.mean(odd_arrays)) / np.sqrt(np.mean(even_arrays)**2 + np.mean(odd_arrays)**2 + sem_even**2 + sem_odd**2)
        f.write(str(E) + "\n")
        f.write("Modified Binhi effect size, in percent:\n")
        sem_even = np.std(even_arrays) 
        sem_odd  = np.std(odd_arrays) 
        Em = 100 * np.abs(np.mean(even_arrays) - np.mean(odd_arrays)) / np.sqrt(np.mean(even_arrays)**2 + np.mean(odd_arrays)**2 + sem_even**2 + sem_odd**2)
        f.write(str(Em) + "\n")
        f.write("True Binhi effect size, in percent (Clarice believes this formula is wrong; denominator takes mean of control only):\n")
        sem_even = np.std(even_arrays) / np.sqrt(len(even_arrays))
        sem_odd  = np.std(odd_arrays) / np.sqrt(len(odd_arrays))
        Eorig = 100 * np.abs(np.mean(even_arrays) - np.mean(odd_arrays)) / np.sqrt(np.mean(even_arrays)**2 + sem_even**2 + sem_odd**2)
        f.write(str(Eorig) + "\n")
        
        ttest_results, mannwhitney_results, kolmogorov_results, shapiro_results = perform_pairwise_tests([even_arrays, odd_arrays], ['C', 'H'], save_avg_path, save=False)
        f.write(f"Stats for all {batches_str} batches combined:\n\n")
        f.write("T test results (if normal distributions) (T-stat, p-value):\n")
        for result in ttest_results:
            f.write(str(result) + "\n")  # Add a newline after each entry
        f.write("Mann-Whitney test results (if non-normal distributions) (U-stat, p-value):\n")
        for result in mannwhitney_results:
            f.write(str(result) + "\n")  # Add a newline after each entry
        f.write("Kolmogorov normality test results (p-values for C and H); if p-value > 0.05 or 0.01: distribution is normal:\n")
        for result in kolmogorov_results:
            f.write(str(result) + "\n")  # Add a newline after each entry
        f.write("Shapiro normality test results (p-values for C and H); if p-value > 0.05 or 0.01: distribution is normal:\n")
        for result in shapiro_results:
            f.write(str(result) + "\n")  # Add a newline after each entry
        f.write("\n\n\n")
        
        ttest_results_2, mannwhitney_results_2, kolmogorov_results_2, shapiro_results_2 = perform_pairwise_tests(concatenated, conditionlabels, save_avg_path, save=False)
        f.write("Stats for each batch:\n\n")
        f.write("T test results (T-stat, p-value):\n")
        for result in ttest_results_2:
            f.write(str(result) + "\n")  # Add a newline after each entry
        f.write("Mann-Whitney test results (U-stat, p-value):\n")
        for result in mannwhitney_results_2:
            f.write(str(result) + "\n")  # Add a newline after each entry
        f.write("Kolmogorov normality test results (p-values for C and H); if p-value > 0.05 or 0.01: distribution is normal:\n")
        for result in kolmogorov_results_2:
            f.write(str(result) + "\n")  # Add a newline after each entry
        f.write("Shapiro normality test results (p-values for C and H); if p-value > 0.05 or 0.01: distribution is normal:\n")
        for result in shapiro_results_2:
            f.write(str(result) + "\n")  # Add a newline after each entry
        f.write("\n\n\n")
        
        f_statistic_C, p_value_C, tukey_C, kw_statistic_C, p_value_kw_C, dunn_results_C, f_statistic_H, p_value_H, tukey_H, kw_statistic_H, p_value_kw_H, dunn_results_H, anova_table, tukey_hsd_condition, tukey_hsd_batch, tukey_interaction_results = ANOVAcomparisons(concatenated)
        f.write("ANOVA (if normal distribution) Controls (fstat, p-value, Tukey):\n")
        f.write(str(f_statistic_C) + "\n")
        f.write(str(p_value_C) + "\n")
        f.write(str(tukey_C) + "\n")
        f.write('If True, it means that there is a statistically significant difference between the means of the two groups.\n\n')
        
        f.write("Kruskal-Wallis (if non-normal distribution) Controls (statistic, p_value, Dunn p-value):\n")
        f.write(str(kw_statistic_C) + "\n")
        f.write(str(p_value_kw_C) + "\n")
        f.write(str(dunn_results_C.to_string(index=True, header=True)) + "\n")
        f.write('If < 0.05 or 0.01, it means that there is a statistically significant difference between the two groups.\n\n\n\n\n')
        
        f.write("ANOVA (if normal distribution) Hypo (fstat, p_value):\n")
        f.write(str(f_statistic_H) + "\n")
        f.write(str(p_value_H) + "\n")
        f.write(str(tukey_H) + "\n")
        f.write('If True, it means that there is a statistically significant difference between the means of the two groups.\n\n')
        
        f.write("Kruskal-Wallis (if non-normal distribution) Hypos (statistic, p-value, Dunn p-value):\n")
        f.write(str(kw_statistic_H) + "\n")
        f.write(str(p_value_kw_H) + "\n")
        f.write(str(dunn_results_H.to_string(index=True, header=True)) + "\n")
        f.write('If < 0.05 or 0.01, it means that there is a statistically significant difference between the two groups.\n\n\n\n\n')
        
        f.write("2-way ANOVA table:\n")
        f.write(str(anova_table) + "\n")
        f.write("Tukey HSD for Condition (Hypo or Control):\n")
        f.write(str(tukey_hsd_condition) + "\n")
        f.write("Tukey HSD for Batch:\n")
        f.write(str(tukey_hsd_batch) + "\n\n\n")
        f.write(tukey_interaction_results)

        return mannwhitney_results_2, dunn_results_C
    
def stats_and_plot(even_arrays, odd_arrays):
            
        cohen_d =100 * np.abs( pg.compute_effsize(even_arrays, odd_arrays, eftype='cohen') )
        #cohen_d = 100 *       ( pg.compute_effsize(even_arrays, odd_arrays, eftype='cohen') )
        
        visibility = 100 * np.abs((np.mean(even_arrays) - np.mean(odd_arrays)))/(np.mean(even_arrays) + np.mean(odd_arrays))
        #visibility  = 100 *       ((np.mean(even_arrays) - np.mean(odd_arrays)))/(np.mean(even_arrays) + np.mean(odd_arrays))
        
        sem_even = np.std(even_arrays) / np.sqrt(len(even_arrays))
        sem_odd  = np.std(odd_arrays) / np.sqrt(len(odd_arrays))
        Eorig = 100 * np.abs(np.mean(even_arrays) - np.mean(odd_arrays)) / np.sqrt(np.mean(even_arrays)**2 + sem_even**2 + sem_odd**2)
        #Eorig  = 100 *       (np.mean(even_arrays) - np.mean(odd_arrays)) / np.sqrt(np.mean(even_arrays)**2 + sem_even**2 + sem_odd**2)
        
        delta, size = cliffs_delta(even_arrays, odd_arrays)
        cliffs_d = 100 * np.abs(delta)
        #cliffs_d  = 100 *       (delta)
        
        #Test for normality
        # Shapiro
        #shapiro_stat_C, p_value_C = stats.shapiro(even_arrays)
        #shapiro_stat_H, sp_value_H = stats.shapiro(odd_arrays)
        
        # Kolmogorov-Smirnov
        statistic_C, p_value_C = kstest(even_arrays, 'norm', args=(np.mean(even_arrays), np.std(even_arrays)))
        statistic_H, p_value_H= kstest(odd_arrays, 'norm', args=(np.mean(odd_arrays), np.std(odd_arrays)))
        
        my_alpha = 0.01
        if p_value_C  >= my_alpha and p_value_H  >= my_alpha: # Both data are normal, T-test
             t_stat, t_p_value = ttest_ind(even_arrays, odd_arrays, equal_var=False)
             p_value = t_p_value
        else: # One or both data are not normal, Mann-Whitney
            mw_stat, mw_p_value = mannwhitneyu(even_arrays, odd_arrays, alternative='two-sided')
            p_value = mw_p_value
        
        return cohen_d, visibility, Eorig, p_value, cliffs_d
    
def process_all_pairs(concatenated):
    """
    Processes all even-odd pairs in concatenated and returns 5 vectors:
    - cohen_d_values
    - visibility_values
    - Eorig_values
    - p_values
    - cliffs_d_values
    """
    
    cohen_d_values = []
    visibility_values = []
    Eorig_values = []
    p_values = []
    cliffs_d_values = []

    # Iterate over pairs of even and odd arrays
    for i in range(0, len(concatenated), 2):
        even_arrays = concatenated[i]
        odd_arrays = concatenated[i + 1]

        cohen_d, visibility, Eorig, p_value, cliffs_d = stats_and_plot(even_arrays, odd_arrays)

        # Store results in their respective lists
        cohen_d_values.append(cohen_d)
        visibility_values.append(visibility)
        Eorig_values.append(Eorig)
        p_values.append(p_value)
        cliffs_d_values.append(cliffs_d)

    return cohen_d_values, visibility_values, Eorig_values, p_values, cliffs_d_values
    
def process_images_in_folder_yellow(folder_path, lower_yellow, upper_yellow):
    # Get all jpg images in the folder
    images = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    
    # List to store the generated masks
    masks = []

    for image_name in images:
        # Construct the full image path
        image_path = os.path.join(folder_path, image_name)
        
        # Load the image
        image = cv2.imread(image_path)
        
        # Convert the image to RGB (OpenCV loads images in BGR by default)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create a mask that identifies the yellow regions
        mask = cv2.inRange(image_rgb, lower_yellow, upper_yellow)
        
        # Append the mask to the list of masks
        masks.append(mask)
        
        # Plot the mask
        plt.imshow(mask, cmap='gray')
        plt.title(f'Mask for {image_name}')
        plt.show()
        
        # Save the mask as a png file
        mask_name = f"{os.path.splitext(image_name)[0]}_mask.png"
        mask_path = os.path.join(folder_path, mask_name)
        cv2.imwrite(mask_path, mask)

        print(f'Mask saved to {mask_path}')
    
    # Return the list of masks
    return masks

def process_images_in_folder_yellow_new(folder_path):
    lower_yellow = np.array([20, 100, 100])   # Lower bound for HSV yellow
    upper_yellow = np.array([30, 255, 255])   # Upper bound for HSV yellow

    masks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            mask = get_yellow_mask(img, lower_yellow, upper_yellow)
            largest_contour_mask = extract_largest_contour(mask, img)
            masks.append(largest_contour_mask)
            save_path = os.path.join(folder_path, filename.replace(".jpg", "_masked.png"))
            cv2.imwrite(save_path, largest_contour_mask)
            plt.imshow(largest_contour_mask, cmap='gray')
            plt.title(f'Mask for {filename}')
            plt.show()

    return masks

def get_yellow_mask(image, lower_yellow, upper_yellow):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    return mask

def extract_largest_contour(mask, original_image):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
        # Create a new image with the masked area
        result = cv2.bitwise_and(original_image, original_image, mask=mask)
        
        # Get bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Crop the image to this bounding box
        cropped_result = result[y:y+h, x:x+w]
        
        return cropped_result
    else:
        return np.zeros_like(mask)

def find_extra_image(filenames, filtered_status_d3):
    extra_image = None
    em = []
    
    for i, fname_list in enumerate(filenames):
        for fname in fname_list:
            # Check if the file name is in the filtered_status_d3
            found = False
            for status_list in filtered_status_d3:
                if fname in status_list:
                    found = True
                    break
            # If the file is not found in any of the status lists, it might be the extra one
            if not found:
                em.append(fname)
                #extra_image = fname
                break
        #if extra_image:
         #   break

    return em #extra_image


    
def dot_plot(control_means, intervention_means, batches_to_analyze, plot_title, font_title, font_text):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot control and intervention means as dots
    ax.plot(control_means, range(len(control_means)), 'bo', label='Control')
    ax.plot(intervention_means, range(len(intervention_means)), 'ro', label='Intervention')

    # Set labels
    ax.set_yticks(range(len(batches_to_analyze)))
    ax.set_yticklabels(batches_to_analyze)
    ax.set_xlabel('Mean Outcome')
    ax.set_title(plot_title, fontsize=16, fontproperties=font_title)
    ax.legend()

    fig.tight_layout()
    plt.show()

def box_plot(control_data, intervention_data, batches_to_analyze, plot_title, font_title, font_text):
    fig, ax = plt.subplots(figsize=(10, 6))
    combined_data = [control_data[i] + intervention_data[i] for i in range(len(batches_to_analyze))]

    # Plotting the box plot
    ax.boxplot(combined_data, labels=batches_to_analyze, patch_artist=True)

    # Set labels
    ax.set_title(plot_title, fontsize=16, fontproperties=font_title)
    ax.set_ylabel('Values')
    fig.tight_layout()
    plt.show()
    
def boxplot(concatenated, flattened_even, flattened_odd):

    # Combine data into a DataFrame for easy plotting
    data = {
        'Group': [],
        'Batch': [],
        'Outcome': []
        }

    batches = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'all', 'PC1']

    # Concatenate control and intervention data into one structure for plotting
    for i, batch in enumerate(batches[:-2]):
        data['Group'].extend(['Control'] * len(concatenated[2*i]) + ['Intervention'] * len(concatenated[2*i+1]))
        data['Batch'].extend([batch] * (len(concatenated[2*i]) + len(concatenated[2*i+1])))
        data['Outcome'].extend(concatenated[2*i])
        data['Outcome'].extend(concatenated[2*i+1])

    # Add the average data
    data['Group'].extend(['Control'] * len(flattened_even) + ['Intervention'] * len(flattened_odd))
    data['Batch'].extend(['Avg'] * (len(flattened_even) + len(flattened_odd)))
    data['Outcome'].extend(flattened_even)
    data['Outcome'].extend(flattened_odd)

    # Add the positive control data
    data['Group'].extend(['Control'] * 1 + ['Intervention'] * 7)
    data['Batch'].extend(['PC1'] * (1 + 7))
    data['Outcome'].extend(control_means_pc)
    data['Outcome'].extend(intervention_means_pc)

    df = pd.DataFrame(data)

    # Create the box plot
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Batch', y='Outcome', hue='Group', data=df, palette='Set2')

    # Add labels and title
    plt.xlabel('Batches')
    plt.ylabel('Outcome')
    plt.title('Box Plot of Outcome by Batch and Group')
    

def swarm(concatenated, flattened_even, flattened_odd, control_means_pc, intervention_means_pc):
    # Combine data into a DataFrame for easy plotting
    data = {
        'Group': [],
        'Batch': [],
        'Outcome': []
        }

    batches = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'Avg', 'PC1']

    # Concatenate control and intervention data into one structure for plotting
    for i, batch in enumerate(batches[:-2]):
        data['Group'].extend(['Control'] * len(concatenated[2*i]) + ['Intervention'] * len(concatenated[2*i+1]))
        data['Batch'].extend([batch] * (len(concatenated[2*i]) + len(concatenated[2*i+1])))
        data['Outcome'].extend(concatenated[2*i])
        data['Outcome'].extend(concatenated[2*i+1])

    # Add the average data
    data['Group'].extend(['Control'] * len(flattened_even) + ['Intervention'] * len(flattened_odd))
    data['Batch'].extend(['Avg'] * (len(flattened_even) + len(flattened_odd)))
    data['Outcome'].extend(flattened_even)
    data['Outcome'].extend(flattened_odd)

    # Add the positive control data
    data['Group'].extend(['Control'] * len(control_means_pc) + ['Intervention'] * len(intervention_means_pc))
    data['Batch'].extend(['PC1'] * (len(control_means_pc) + len(intervention_means_pc)))
    data['Outcome'].extend(control_means_pc)
    data['Outcome'].extend(intervention_means_pc)

    df = pd.DataFrame(data)

    # Create the swarm plot
    plt.figure(figsize=(12, 8))
    sns.swarmplot(x='Batch', y='Outcome', hue='Group', data=df, dodge=True, palette='Set2')

    # Add labels and title
    plt.xlabel('Batches')
    plt.ylabel('Outcome')
    plt.title('Swarm Plot of Outcome by Batch and Group')
    
def strip(concatenated, flattened_even, flattened_odd, control_means_pc, intervention_means_pc):
    
    # Combine data into a DataFrame for easy plotting
    data = {
        'Group': [],
        'Batch': [],
        'Outcome': []
        }

    batches = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'Avg', 'PC1']

    # Concatenate control and intervention data into one structure for plotting
    for i, batch in enumerate(batches[:-2]):
        data['Group'].extend(['Control'] * len(concatenated[2*i]) + ['Intervention'] * len(concatenated[2*i+1]))
        data['Batch'].extend([batch] * (len(concatenated[2*i]) + len(concatenated[2*i+1])))
        data['Outcome'].extend(concatenated[2*i])
        data['Outcome'].extend(concatenated[2*i+1])

    # Add the average data
    data['Group'].extend(['Control'] * len(flattened_even) + ['Intervention'] * len(flattened_odd))
    data['Batch'].extend(['Avg'] * (len(flattened_even) + len(flattened_odd)))
    data['Outcome'].extend(flattened_even)
    data['Outcome'].extend(flattened_odd)

    # Add the positive control data
    data['Group'].extend(['Control'] * len(control_means_pc) + ['Intervention'] * len(intervention_means_pc))
    data['Batch'].extend(['PC1'] * (len(control_means_pc) + len(intervention_means_pc)))
    data['Outcome'].extend(control_means_pc)
    data['Outcome'].extend(intervention_means_pc)

    df = pd.DataFrame(data)

    # Create the strip plot
    plt.figure(figsize=(12, 8))
    sns.stripplot(x='Batch', y='Outcome', hue='Group', data=df, dodge=True, palette='Set2', jitter=True)

    # Add labels and title
    plt.xlabel('Batches')
    plt.ylabel('Outcome')
    plt.title('Strip Plot of Outcome by Batch and Group')

def confidence_interval(data, confidence=0.95):
    """
    Calculate the confidence interval for a given data array.
    :param data: array-like, data for which to calculate the confidence interval
    :param confidence: float, confidence level (default is 0.95 for 95% confidence interval)
    :return: (mean, lower bound, upper bound) of the confidence interval
    """
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of the mean
    h = sem * stats.t.ppf((1 + confidence) / 2., n-1)  # Margin of error
    return mean, mean - h, mean + h, h

def cliffs_delta(x, y):
    n_x = len(x)
    n_y = len(y)
    x = np.array(x)
    y = np.array(y)
    delta = (np.sum(x[:, None] > y) - np.sum(x[:, None] < y)) / (n_x * n_y)
    
    # Interpretation of the effect size
    abs_delta = np.abs(delta)
    if abs_delta < 0.147:
        size = "negligible"
    elif abs_delta < 0.33:
        size = "small"
    elif abs_delta < 0.474:
        size = "medium"
    else:
        size = "large"
    
    return delta, size


    
def plot_stats1(fig, ax1, ax2, cohen_d_array, visibility_array, Eorig_array, p_value_array, cliffs_d,
                    batches_to_analyze, plot_title, font_title, font_text, 
                    SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, 
                    y_min1, y_max1, y_min2, y_max2, 
                    avg_cohen, avg_visibility, avg_E, avg_p, cliffs_avg, n_array,
                    cohen_d_array_pc=None, visibility_array_pc=None, Eorig_array_pc=None, p_value_array_pc=None, cliffs_d_pc=None, n_array_pc=None,
                    cohen_d_avg_pc=None, visibility_avg_pc=None, Eorig_avg_pc=None, p_value_avg_pc=None, cliffs_avg_pc=None,
                    ishealthy=False):
    
    # Append "avg" and "PC" labels to the x-axis
    endash = "\u2013"
    batch_labels = [f'\n\n\n\nB{i+1} \n (n={n_array[i]})' if (i+1) % 2 == 0 else f'B{i+1} \n (n={n_array[i]})' for i in range(len(n_array))]
    pc_labels =  [f'\n\n\n\n+{i+1} \n (n={n_array_pc[i]})' if (i+1) % 2 == 0 else f'+{i+1} \n (n={n_array_pc[i]})' for i in range(len(n_array_pc))] + [f'\n\n\n\n+1{endash}{len(n_array_pc)} \n (n={np.sum(n_array_pc)})']
    batches_to_analyze = batch_labels + [f'\n\n\n\nB1{endash}{len(n_array)} \n (n={np.sum(n_array)})'] + pc_labels

    plt.suptitle(plot_title, fontsize=BIGGER_SIZE, fontproperties=font_title)
    
    # Color choice based on whether it's healthy data
    my_color = 'blue' if not ishealthy else 'green'
    
    # Plotting the data on the top subplot
    x_positions = range(len(batches_to_analyze) - 1 - len(pc_labels))  # x positions for the batch data
    ax1.plot(x_positions, cohen_d_array, label="Cohen's |d|", marker='D', color='orange', markersize=7, linestyle='None')
    ax1.plot(x_positions, cliffs_d, label="Cliff's |$\Delta$|", marker='^', color='purple', markersize=7, linestyle='None')
    
    # Plot the average values at the "all" position
    ax1.plot(len(batches_to_analyze) - 1 - len(pc_labels), avg_cohen, marker='D', color='orange', markersize=7,  linestyle='None')
    ax1.plot(len(batches_to_analyze) - 1 - len(pc_labels), cliffs_avg, marker='^', color='purple', markersize=7,  linestyle='None')
    
    # Plot the PC values if provided
    if cohen_d_array_pc is not None:
        for i, (cohen_pc, cliffsdpc) in enumerate(zip(cohen_d_array_pc, cliffs_d_pc)):
            x_position = len(batches_to_analyze) - len(pc_labels) + i  # Ensure this is an array with two values if you have 2 PCs
            ax1.plot(x_position, cohen_pc, marker='D', color='orange', markersize=7, linestyle='None')
            ax1.plot(x_position, cliffsdpc, marker='^', color='purple', markersize=7,  linestyle='None')
        ax1.plot(11, cohen_d_avg_pc, marker='D', color='orange', markersize=7, linestyle='None')
        ax1.plot(11, cliffs_avg_pc, marker='^', color='purple', markersize=7,  linestyle='None')
 
    # Horizontal lines for reference -- Cohen
    ax1.axhline(y=20, color='orange', linestyle='-', linewidth=1)
    ax1.axhline(y=50, color='orange', linestyle='-', linewidth=1)
    ax1.axhline(y=80, color='orange', linestyle='-', linewidth=1)
    
    # Horizontal lines for reference -- Cliffs
    ax1.axhline(y=14.7, color='purple', linestyle=':', linewidth=1)
    ax1.axhline(y=33, color='purple', linestyle=':', linewidth=1)
    ax1.axhline(y=47.4, color='purple', linestyle=':', linewidth=1)
    
    ax1.set_ylabel('effect size (%)', fontproperties=font_text, fontsize=MEDIUM_SIZE)
    ax1.set_ylim(y_min1, y_max1)
    
    legend_font = FontProperties(fname=font_text.get_file(), size=MEDIUM_SIZE)
    
    # Assuming legend_elements is already a list of handles
    legend = ax1.legend(
        loc='upper right',
        bbox_to_anchor=(1, 1.05),  # Adjust these values as needed
        frameon=True,             # Turn off the frame around the legend
        prop=legend_font,          # Adjust font size as needed
        handletextpad=0.5,         # Spacing between the legend symbol and text
        labelspacing=0.5           # Spacing between rows of the legend
    )
    
    # Customize the frame (facecolor sets the background color)
    legend.get_frame().set_facecolor('white')  # Set background color to white
    legend.get_frame().set_edgecolor('none')   # Remove the edge of the frame (border)
    legend.get_frame().set_alpha(1.0)     
    
    ax1.grid(True)
    
    # Plotting the data on the bottom subplot
    ax2.plot(x_positions, p_value_array, label='p value', marker='o', color=my_color,markersize=7,  linestyle='None')
    
    # Plot the average p-value at the "all" position
    ax2.plot(len(batches_to_analyze) - 1 - len(pc_labels), avg_p, marker='o', color=my_color, markersize=7,  linestyle='None')

    # Plot the PC p-values if provided
    if p_value_array_pc is not None:
        for i, p_value_pc in enumerate(p_value_array_pc):
            x_position = len(batches_to_analyze) - len(pc_labels) + i  # Ensure this is an array with two values if you have 2 PCs
            ax2.plot(x_position, p_value_pc, marker='o', color=my_color, markersize=7, linestyle='None')
        ax2.plot(11, p_value_avg_pc, marker='o', color=my_color, markersize=7, linestyle='None')
    
    my_alpha = 0.01
    ax2.axhline(y=my_alpha, color='red', linestyle='--', linewidth=1)

    ax2.set_yscale('log')
    
    # Customize the ticks for the log scale
    log_min = np.floor(np.log10(y_min2))
    log_max = np.ceil(np.log10(y_max2))
    step = (log_max - log_min) / 3
    ticks = [10**(log_min + i*step) for i in range(4)]
    #if 10**-2 >= y_min2 and 10**-2 <= y_max2 and 10**-2 not in ticks:
     #   ticks.append(10**-2)
    ticks = sorted(ticks)
    ax2.set_yticks(ticks)
    ax2.set_yticklabels([f'{int(np.log10(tick))}' for tick in ticks])
    
    # Apply font settings to y-tick labels
    for my_label in ax2.get_yticklabels():
        my_label.set_fontproperties(font_text)
        my_label.set_fontsize(MEDIUM_SIZE)

    ax2.set_ylabel('log(p value)', fontsize=MEDIUM_SIZE, fontproperties=font_text)
    #ax2.set_xlabel('Batches', fontproperties=font_text, fontsize=MEDIUM_SIZE)
    ax2.set_ylim(y_min2, y_max2)
    ax2.grid(True)
    
    # Set custom x-ticks to include the "all" and "PC" labels
    ax2.set_xticks(range(len(batches_to_analyze)))
    #ax2.set_xticklabels(batches_to_analyze, rotation=45, ha='center', va='center')
    ax2.set_xticklabels(batches_to_analyze, ha='center', va='center')
    ax2.tick_params(axis='x', pad=15)  # Increase the pad value to move the labels lower
    
    # Modify tick length for odd and even batches
    for i, tick in enumerate(ax2.xaxis.get_major_ticks()):
        if (i + 1) % 2 == 0:  # Odd batches
            tick.tick1line.set_markersize(30)  # Longer ticks
            tick.tick2line.set_markersize(30)  # Longer ticks
    
    # Adjust font properties for tick labels
    for my_label in ax1.get_xticklabels():
        my_label.set_fontproperties(font_text)
        my_label.set_fontsize(MEDIUM_SIZE)
    for my_label in ax1.get_yticklabels():
        my_label.set_fontproperties(font_text)
        my_label.set_fontsize(MEDIUM_SIZE)
    for my_label in ax2.get_xticklabels():
        my_label.set_fontproperties(font_text)
        my_label.set_fontsize(MEDIUM_SIZE)
        
    ax1.axvline(7.5, color='black', linestyle='--', linewidth=1)
    ax2.axvline(7.5, color='black', linestyle='--', linewidth=1)
    
    # Create a ConnectionPatch
    con = ConnectionPatch(xyA=(7.52, 0), xyB=(7.52, 1), coordsA="data", coordsB="data",
                          axesA=ax1, axesB=ax2, color="black", linestyle='--',linewidth=1)
    # Add the ConnectionPatch to the figure
    fig.add_artist(con)
        
    ax1.grid(False)
    ax2.grid(False, axis='x')

    # Hide the top and right spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()

    
def plot_stats2(fig, ax1, ax2, cohen_d_array, visibility_array, Eorig_array, p_value_array, cliffs_d,
                    batches_to_analyze, plot_title, font_title, font_text, 
                    SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, 
                    y_min1, y_max1, y_min2, y_max2, 
                    avg_cohen, avg_visibility, avg_E, avg_p, cliffs_avg, n_array,
                    cohen_d_array_pc=None, visibility_array_pc=None, Eorig_array_pc=None, p_value_array_pc=None, cliffs_d_pc=None, n_array_pc=None,
                    cohen_d_avg_pc=None, visibility_avg_pc=None, Eorig_avg_pc=None, p_value_avg_pc=None, cliffs_avg_pc=None,
                    ishealthy=False):
    
    # Append "avg" and "PC" labels to the x-axis
    endash = "\u2013"
    batch_labels = [f'\n\n\n\nB{i+1} \n (n={n_array[i]})' if (i+1) % 2 == 0 else f'B{i+1} \n (n={n_array[i]})' for i in range(len(n_array))]
    pc_labels =  [f'\n\n\n\n+{i+1} \n (n={n_array_pc[i]})' if (i+1) % 2 == 0 else f'+{i+1} \n (n={n_array_pc[i]})' for i in range(len(n_array_pc))] + [f'\n\n\n\n+1{endash}{len(n_array_pc)} \n (n={np.sum(n_array_pc)})']
    batches_to_analyze = batch_labels + [f'\n\n\n\nB1{endash}{len(n_array)} \n (n={np.sum(n_array)})'] + pc_labels
    
    plt.suptitle(plot_title, fontsize=BIGGER_SIZE, fontproperties=font_title)
    
    # Color choice based on whether it's healthy data
    my_color = 'blue' if not ishealthy else 'green'
    
    # Plotting the data on the top subplot
    x_positions = range(len(batches_to_analyze) - 1 - len(pc_labels))  # x positions for the batch data
    ax1.plot(x_positions, Eorig_array, label="Binhi's |E|", marker='s', color=my_color, markersize=7,  linestyle='None')
    ax1.plot(x_positions, visibility_array, label=r'visibility |$\nu$|', marker='x', color=my_color, markersize=7,  linestyle='None')
    
    # Plot the average values at the "all" position
    ax1.plot(len(batches_to_analyze) - 1 - len(pc_labels), avg_visibility, marker='x', color=my_color, markersize=7,  linestyle='None')
    ax1.plot(len(batches_to_analyze) - 1 - len(pc_labels), avg_E, marker='s', color=my_color, markersize=7,  linestyle='None')
    
    # Plot the PC values if provided
    if cohen_d_array_pc is not None:
        for i, (visibility_pc, Eorig_pc) in enumerate(zip(visibility_array_pc, Eorig_array_pc)):
            x_position = len(batches_to_analyze) - len(pc_labels) + i  # Ensure this is an array with two values if you have 2 PCs
            ax1.plot(x_position, visibility_pc, marker='x', color=my_color, markersize=7,  linestyle='None')
            ax1.plot(x_position, Eorig_pc, marker='s', color=my_color, markersize=7,  linestyle='None')
        ax1.plot(11, visibility_avg_pc, marker='x', color=my_color, markersize=7,  linestyle='None')
        ax1.plot(11, Eorig_avg_pc, marker='s', color=my_color, markersize=7,  linestyle='None')

    
    ax1.set_ylabel('effect size (%)', fontproperties=font_text, fontsize=MEDIUM_SIZE)
    ax1.set_ylim(y_min1, y_max1)
    
    legend_font = FontProperties(fname=font_text.get_file(), size=MEDIUM_SIZE)
    
    # Assuming legend_elements is already a list of handles
    legend = ax1.legend(
        loc='upper right',
        bbox_to_anchor=(1, 1.05),  # Adjust these values as needed
        frameon=True,             # Turn off the frame around the legend
        prop=legend_font,          # Adjust font size as needed
        handletextpad=0.5,         # Spacing between the legend symbol and text
        labelspacing=0.5           # Spacing between rows of the legend
    )
    
    # Customize the frame (facecolor sets the background color)
    legend.get_frame().set_facecolor('white')  # Set background color to white
    legend.get_frame().set_edgecolor('none')   # Remove the edge of the frame (border)
    legend.get_frame().set_alpha(1.0)     
    
    ax1.grid(True)
    
    # Plotting the data on the bottom subplot
    ax2.plot(x_positions, p_value_array, label='p value', marker='o', color=my_color, markersize=7,  linestyle='None')
    
    # Plot the average p-value at the "all" position
    ax2.plot(len(batches_to_analyze) - 1 - len(pc_labels), avg_p, marker='o', color=my_color, markersize=7,  linestyle='None')

    # Plot the PC p-values if provided
    if p_value_array_pc is not None:
        for i, p_value_pc in enumerate(p_value_array_pc):
            x_position = len(batches_to_analyze) - len(pc_labels) + i  # Ensure this is an array with two values if you have 2 PCs
            ax2.plot(x_position, p_value_pc, marker='o', color=my_color, markersize=7,  linestyle='None')
        ax2.plot(11, p_value_avg_pc, marker='o', color=my_color, markersize=7, linestyle='None')
    
    my_alpha = 0.01
    ax2.axhline(y=my_alpha, color='red', linestyle='--', linewidth=1)

    ax2.set_yscale('log')
    
    # Customize the ticks for the log scale
    log_min = np.floor(np.log10(y_min2))
    log_max = np.ceil(np.log10(y_max2))
    step = (log_max - log_min) / 3
    ticks = [10**(log_min + i*step) for i in range(4)]
    #if 10**-2 >= y_min2 and 10**-2 <= y_max2 and 10**-2 not in ticks:
       # ticks.append(10**-2)
    ticks = sorted(ticks)
    ax2.set_yticks(ticks)
    ax2.set_yticklabels([f'{int(np.log10(tick))}' for tick in ticks])
    
    # Apply font settings to y-tick labels
    for my_label in ax2.get_yticklabels():
        my_label.set_fontproperties(font_text)
        my_label.set_fontsize(MEDIUM_SIZE)

    ax2.set_ylabel('log(p value)', fontsize=MEDIUM_SIZE, fontproperties=font_text)
    #ax2.set_xlabel('Batches', fontproperties=font_text, fontsize=MEDIUM_SIZE)
    ax2.set_ylim(y_min2, y_max2)
    ax2.grid(True)
    
    # Set custom x-ticks to include the "all" and "PC" labels
    ax2.set_xticks(range(len(batches_to_analyze)))
    #ax2.set_xticklabels(batches_to_analyze, rotation=45, ha='center', va='center')
    ax2.set_xticklabels(batches_to_analyze, ha='center', va='center')
    ax2.tick_params(axis='x', pad=15)  # Increase the pad value to move the labels lower
    
    # Modify tick length for odd and even batches
    for i, tick in enumerate(ax2.xaxis.get_major_ticks()):
        if (i + 1) % 2 == 0:  # Odd batches
            tick.tick1line.set_markersize(30)  # Longer ticks
            tick.tick2line.set_markersize(30)  # Longer ticks
    
    # Adjust font properties for tick labels
    for my_label in ax1.get_xticklabels():
        my_label.set_fontproperties(font_text)
        my_label.set_fontsize(MEDIUM_SIZE)
    for my_label in ax1.get_yticklabels():
        my_label.set_fontproperties(font_text)
        my_label.set_fontsize(MEDIUM_SIZE)
    for my_label in ax2.get_xticklabels():
        my_label.set_fontproperties(font_text)
        my_label.set_fontsize(MEDIUM_SIZE)
        
    ax1.axvline(7.5, color='black', linestyle='--', linewidth=1)
    ax2.axvline(7.5, color='black', linestyle='--', linewidth=1)
    # Create a ConnectionPatch
    con = ConnectionPatch(xyA=(7.52, 0), xyB=(7.52, 1), coordsA="data", coordsB="data",
                          axesA=ax1, axesB=ax2, color="black", linestyle='--',linewidth=1)
    # Add the ConnectionPatch to the figure
    fig.add_artist(con)
    
    ax1.grid(False)
    ax2.grid(False, axis='x')

    # Hide the top and right spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
def bootstrap_median(fig, ax, control_data, condition_data, n_bootstrap, 
                                   batches_to_analyze_here, plot_title, font_title, font_text, 
                                   SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE,
                                   n_array, n_array_pc, y_min=None, y_max=None, ishealthy=False):
    """
    Plots the bootstrapped confidence intervals and distributions for multiple pairs of control and condition data.
    n_bootstrap: Number of bootstrap samples.
    Idea from: https://thenode.biologists.com/quantification-of-differences-as-alternative-for-p-values/research/
    Plots the percentage difference between the median of the resampled control data 
    and the median of the resampled condition data for each bootstrap iteration. 
    This percentage difference is then plotted along with the 95% confidence intervals 
    derived from these bootstrapped differences.
    """
    if ishealthy:
        my_face_color = 'lightgreen'
        my_color = 'green'
    else:
        my_face_color = 'lightblue'
        my_color = 'blue'
    
    all_cis = []
    all_bootstrapped_diffs = []

    for control, condition in zip(control_data, condition_data):
        bootstrapped_diffs = []
        for _ in range(n_bootstrap):
            resampled_control = resample(control)
            resampled_condition = resample(condition)
            # Calculate percentage difference from control
            diff = (np.median(resampled_condition) - np.median(resampled_control)) / np.median(resampled_control) * 100
            bootstrapped_diffs.append(diff)
        
        # Calculate the confidence interval
        ci = np.percentile(bootstrapped_diffs, [2.5, 50, 97.5])
        all_cis.append(ci)
        all_bootstrapped_diffs.append(bootstrapped_diffs)
    
    # Determine global min and max for y-axis if not provided
    if y_min is None or y_max is None:
        all_flattened = np.concatenate(all_bootstrapped_diffs)
        y_min = all_flattened.min() if y_min is None else y_min
        y_max = all_flattened.max() if y_max is None else y_max
    
    # Plotting the results
    #fig, ax = plt.subplots()
    plt.suptitle(plot_title, fontsize=BIGGER_SIZE, fontproperties=font_title)
    
    for i, (ci, bootstrapped_diffs) in enumerate(zip(all_cis, all_bootstrapped_diffs)):
        position = i + 1
        
        # Create the violin plot manually
        parts = ax.violinplot(bootstrapped_diffs, positions=[position], showmeans=False, showmedians=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(my_face_color)
            pc.set_alpha(0.3)
        
        # Plot the confidence interval
        ci_handle = ax.errorbar(position, ci[1], 
                     yerr=[[ci[1] - ci[0]], 
                           [ci[2] - ci[1]]],
                     fmt='o', markersize=7, color=my_color,label='95% CI' if i == 0 else "")
        
        # Manually add the legend after all points are plotted
        legend_font = FontProperties(fname=font_text.get_file(), size=MEDIUM_SIZE)
        if i == len(all_cis) - 1:  # Ensure we only add the legend after the last point
         # Create a custom handle for the violin plot
             violin_handle = mpatches.Patch(color=my_face_color, alpha=0.3, label='bootstrapped distrib.')
             #ci_handle = plt.Line2D([0], [0], color=my_color, marker='', linestyle='-', label='95% conf. int.')
             dot_line_handle = plt.Line2D([0], [0], color=my_color, marker='o', linestyle='-', label='median diff. (% of C median) ± 95% conf. int.')
             legend = ax.legend(handles=[violin_handle, dot_line_handle], loc='upper right', bbox_to_anchor=(1, 1.05), frameon=True, prop=legend_font, handletextpad=0.5, labelspacing=0.5)
             
    # # Customize the frame (facecolor sets the background color)
    legend.get_frame().set_facecolor('white')  # Set background color to white
    legend.get_frame().set_edgecolor('none')   # Remove the edge of the frame (border)
    legend.get_frame().set_alpha(1.0)     

    # Set y-axis limits to the global min and max
    ax.set_ylim(y_min, y_max)
    
    # Append "avg" and "PC" labels to the x-axis
    endash = "\u2013"
    batch_labels = [f'\n\n\n\nB{i+1} \n (n={n_array[i]})' if (i+1) % 2 == 0 else f'B{i+1} \n (n={n_array[i]})' for i in range(len(n_array))]
    pc_labels =  [f'\n\n\n\n+{i+1} \n (n={n_array_pc[i]})' if (i+1) % 2 == 0 else f'+{i+1} \n (n={n_array_pc[i]})' for i in range(len(n_array_pc))]  + [f'\n\n\n\n+1{endash}{len(n_array_pc)} \n (n={np.sum(n_array_pc)})']

    batches_to_analyze = batch_labels + [f'\n\n\n\nB1{endash}{len(n_array)} \n (n={np.sum(n_array)})'] + pc_labels
    
    ax.axhline(0, color='red', linestyle='--', lw=1)
    ax.set_xticks(range(1, len(batches_to_analyze) + 1))
    #ax.set_xticklabels(batches_to_analyze, rotation=45, ha='center', va='center')
    ax.set_xticklabels(batches_to_analyze, ha='center', va='center')
    ax.tick_params(axis='x', pad=15)  # Increase the pad value to move the labels lower
    
    # Modify tick length for odd and even batches
    for i, tick in enumerate(ax.xaxis.get_major_ticks()):
        if (i + 1) % 2 == 0:  # Odd batches
            tick.tick1line.set_markersize(30)  # Longer ticks
            tick.tick2line.set_markersize(30)  # Longer ticks
            
    ax.axvline(8.5, color='black', linestyle='--', linewidth=1)
    
    #ax.set_ylabel('median diff. from control (%)', fontproperties=font_text, fontsize=MEDIUM_SIZE)
    # Hide the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
     
    for my_label in ax.get_xticklabels():
         my_label.set_fontproperties(font_text)
         my_label.set_fontsize(MEDIUM_SIZE)
    for my_label in ax.get_yticklabels():
         my_label.set_fontproperties(font_text)
         my_label.set_fontsize(MEDIUM_SIZE)

    # Adjust layout
    #fig.tight_layout()
    plt.tight_layout()
    #plt.show()

def plot_barplot(fig, ax, test_name, 
                control_means, intervention_means, 
                control_cis, intervention_cis,
                batches_to_analyze, plot_title, font_title, font_text, 
                SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, 
                avg_control_means, avg_intervention_means,
                onept_error_control, onept_error_hypo, 
                control_means_pc, intervention_means_pc,
                control_cis_pc, intervention_cis_pc,
                avg_control_means_pc, avg_intervention_means_pc,
                onept_error_control_pc, onept_error_hypo_pc, 
                n_array, n_array_pc,
                ishealthy=None, y_min=None, y_max=None):
    
    if ishealthy is not None:
        mycolor = 'green'
    else:
        mycolor = 'blue'
    
    # Append "avg" and "PC" labels to the x-axis
    endash = "\u2013"
    batch_labels = [f'\n\n\n\nB{i+1} \n (n={n_array[i]})' if (i+1) % 2 == 0 else f'B{i+1} \n (n={n_array[i]})' for i in range(len(n_array))]
    pc_labels =  [f'\n\n\n\n+{i+1} \n (n={n_array_pc[i]})' if (i+1) % 2 == 0 else f'+{i+1} \n (n={n_array_pc[i]})' for i in range(len(n_array_pc))] + [f'\n\n\n\n+1{endash}{len(n_array_pc)} \n (n={np.sum(n_array_pc)})']
    batches_to_analyze = batch_labels + [f'\n\n\n\nB1{endash}{len(n_array)} \n (n={np.sum(n_array)})'] + pc_labels 
    
    # Create the plot
    plt.suptitle(plot_title, fontsize=BIGGER_SIZE, fontproperties=font_title)

    # x positions for the batch data
    x = np.array(range(len(n_array)))  # 7 batches
    x_avg = np.array([len(n_array)])  # Position for 'avg' point
    x_pc = np.arange(len(n_array) + 1, len(n_array) + 1 + len(n_array_pc))  # Position for PC1, .,.. PCn
    x_pc_avg = np.array([x_pc[-1] + 1])
    width = 0.15  # Smaller spacing between pairs

    # Replace the bar plots with points and error bars
    ax.errorbar(x - width/2, control_means, yerr=control_cis, fmt='o', label='Control', color=mycolor, alpha=0.5, capsize=0, markersize=7)
    ax.errorbar(x + width/2, intervention_means, yerr=intervention_cis, fmt='s', label='Intervention', color=mycolor, capsize=0, markersize=7)

    # Plot the points and error bars for the average
    ax.errorbar(x_avg - width/2, avg_control_means, yerr=onept_error_control, fmt='o', label='Avg Control', color=mycolor, alpha=0.5, capsize=0, markersize=7)
    ax.errorbar(x_avg + width/2, avg_intervention_means, yerr=onept_error_hypo, fmt='s', label='Avg Intervention', color=mycolor, capsize=0, markersize=7)
    
    # Plot the points and error bars for the positive controls
    ax.errorbar(x_pc - width/2, control_means_pc, yerr=control_cis_pc, fmt='o', label='Control PC', color=mycolor, alpha=0.5, capsize=0, markersize=7)
    ax.errorbar(x_pc + width/2, intervention_means_pc, yerr=intervention_cis_pc, fmt='s', label='Intervention PC', color=mycolor, capsize=0, markersize=7)

     # Plot the points and error bars for the average_pc
    ax.errorbar(x_pc_avg - width/2, avg_control_means_pc, yerr=onept_error_control_pc, fmt='o', label='Avg Control', color=mycolor, alpha=0.5, capsize=0, markersize=7)
    ax.errorbar(x_pc_avg + width/2, avg_intervention_means_pc, yerr=onept_error_hypo_pc, fmt='s', label='Avg Intervention', color=mycolor, capsize=0, markersize=7)
 
    # Set x-ticks and labels
    all_ticks = list(x) + list(x_avg) + list(x_pc) + list(x_pc_avg)
    
    #ax.set_xticklabels(batches_to_analyze, rotation=45, ha='center', va='center')
    ax.set_xticks(all_ticks)
    ax.set_xticklabels(batches_to_analyze, rotation=0, ha='center', va='center')
    ax.tick_params(axis='x', pad=15)  # Increase the pad value to move the labels lower
    
    # Modify tick length for odd and even batches
    for i, tick in enumerate(ax.xaxis.get_major_ticks()):
        if (i + 1) % 2 == 0:  # Odd batches
            tick.tick1line.set_markersize(30)  # Longer ticks
            tick.tick2line.set_markersize(30)  # Longer ticks

    # Add the legend for "Control" and "Hypo" conditions
    custom_legend = [
        plt.Line2D([0], [0], color=mycolor, marker='o', alpha=0.5, lw=2, label='avg. C ± 95% conf. int.'),
        plt.Line2D([0], [0], color=mycolor, marker='s', lw=2, label='avg. H ± 95% conf. int.')
    ]
    legend_font = FontProperties(fname=font_text.get_file(), size=MEDIUM_SIZE)
    # Assuming legend_elements is already a list of handles
    legend = ax.legend(
        handles=custom_legend,  # Pass the list directly without wrapping it in another list
        loc='upper right',
        bbox_to_anchor=(1, 1.05),  # Adjust these values as needed
        frameon=True,             # Turn off the frame around the legend
        prop=legend_font,          # Adjust font size as needed
        handletextpad=0.5,         # Spacing between the legend symbol and text
        labelspacing=0.5           # Spacing between rows of the legend
    )
    
    # # Customize the frame (facecolor sets the background color)
    legend.get_frame().set_facecolor('white')  # Set background color to white
    legend.get_frame().set_edgecolor('none')   # Remove the edge of the frame (border)
    legend.get_frame().set_alpha(1.0)     
    
    # Add a red line at y = 1.0 (will only show if y = 1.0 is within range)
    ax.axhline(y=1.0, color='red', linewidth=1,linestyle='--')
    
    ax.axvline(7.5, color='black', linestyle='--', linewidth=1)

    # Set the y-limits based on the provided or calculated values
    if y_min is not None and y_max is not None:
        ax.set_ylim([y_min, y_max])
     
    # Hide the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
     
    for my_label in ax.get_xticklabels():
          my_label.set_fontproperties(font_text)
          my_label.set_fontsize(MEDIUM_SIZE)
    for my_label in ax.get_yticklabels():
          my_label.set_fontproperties(font_text)
          my_label.set_fontsize(MEDIUM_SIZE)

    # Adjust layout
    plt.tight_layout()
    
def plot_barplot_delta(fig, ax, test_name, 
                control_means, intervention_means, 
                control_cis, intervention_cis,
                batches_to_analyze, plot_title, font_title, font_text, 
                SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, 
                avg_control_means, avg_intervention_means,
                onept_error_control, onept_error_hypo, 
                control_means_pc, intervention_means_pc,
                control_cis_pc, intervention_cis_pc,
                avg_control_means_pc, avg_intervention_means_pc,
                onept_error_control_pc, onept_error_hypo_pc, 
                n_array, n_array_pc,
                ishealthy=None, y_min=None, y_max=None):
    
    if ishealthy is not None:
        mycolor = 'green'
    else:
        mycolor = 'blue'
    
    # Append "avg" and "PC" labels to the x-axis
    endash = "\u2013"
    batch_labels = [f'\n\n\n\nB{i+1} \n (n={n_array[i]})' if (i+1) % 2 == 0 else f'B{i+1} \n (n={n_array[i]})' for i in range(len(n_array))]
    pc_labels =  [f'\n\n\n\n+{i+1} \n (n={n_array_pc[i]})' if (i+1) % 2 == 0 else f'+{i+1} \n (n={n_array_pc[i]})' for i in range(len(n_array_pc))] + [f'\n\n\n\n+1{endash}{len(n_array_pc)} \n (n={np.sum(n_array_pc)})']
    batches_to_analyze = batch_labels + [f'\n\n\n\nB1{endash}{len(n_array)} \n (n={np.sum(n_array)})'] + pc_labels 
    
    # Create the plot
    plt.suptitle(plot_title, fontsize=BIGGER_SIZE, fontproperties=font_title)

    # x positions for the batch data
    x = np.array(range(len(n_array)))  # 7 batches
    x_avg = np.array([len(n_array)])  # Position for 'avg' point
    x_pc = np.arange(len(n_array) + 1, len(n_array) + 1 + len(n_array_pc))  # Position for PC1, .,.. PCn
    x_pc_avg = np.array([x_pc[-1] + 1])
    width = 0.15  # Smaller spacing between pairs

    # Replace the bar plots with points and error bars
    ax.errorbar(x - width/2, control_means, yerr=control_cis, fmt='o', label='Control', color=mycolor, alpha=0.5, capsize=0, markersize=7)
    ax.errorbar(x + width/2, intervention_means, yerr=intervention_cis, fmt='s', label='Intervention', color=mycolor, capsize=0, markersize=7)

    # Plot the points and error bars for the average
    ax.errorbar(x_avg - width/2, avg_control_means, yerr=onept_error_control, fmt='o', label='Avg Control', color=mycolor, alpha=0.5, capsize=0, markersize=7)
    ax.errorbar(x_avg + width/2, avg_intervention_means, yerr=onept_error_hypo, fmt='s', label='Avg Intervention', color=mycolor, capsize=0, markersize=7)
    
    # Plot the points and error bars for the positive controls
    ax.errorbar(x_pc - width/2, control_means_pc, yerr=control_cis_pc, fmt='o', label='Control PC', color=mycolor, alpha=0.5, capsize=0, markersize=7)
    ax.errorbar(x_pc + width/2, intervention_means_pc, yerr=intervention_cis_pc, fmt='s', label='Intervention PC', color=mycolor, capsize=0, markersize=7)

     # Plot the points and error bars for the average_pc
    ax.errorbar(x_pc_avg - width/2, avg_control_means_pc, yerr=onept_error_control_pc, fmt='o', label='Avg Control', color=mycolor, alpha=0.5, capsize=0, markersize=7)
    ax.errorbar(x_pc_avg + width/2, avg_intervention_means_pc, yerr=onept_error_hypo_pc, fmt='s', label='Avg Intervention', color=mycolor, capsize=0, markersize=7)
 
    # Set x-ticks and labels
    all_ticks = list(x) + list(x_avg) + list(x_pc) + list(x_pc_avg)
    
    #ax.set_xticklabels(batches_to_analyze, rotation=45, ha='center', va='center')
    ax.set_xticks(all_ticks)
    ax.set_xticklabels(batches_to_analyze, rotation=0, ha='center', va='center')
    ax.tick_params(axis='x', pad=15)  # Increase the pad value to move the labels lower
    
    # Modify tick length for odd and even batches
    for i, tick in enumerate(ax.xaxis.get_major_ticks()):
        if (i + 1) % 2 == 0:  # Odd batches
            tick.tick1line.set_markersize(30)  # Longer ticks
            tick.tick2line.set_markersize(30)  # Longer ticks

    # Add the legend for "Control" and "Hypo" conditions
    custom_legend = [
        plt.Line2D([0], [0], color=mycolor, marker='o', alpha=0.5, lw=2, label='avg. C ± 95% conf. int.'),
        plt.Line2D([0], [0], color=mycolor, marker='s', lw=2, label='avg. H ± 95% conf. int.')
    ]
    legend_font = FontProperties(fname=font_text.get_file(), size=MEDIUM_SIZE)
    # Assuming legend_elements is already a list of handles
    legend = ax.legend(
        handles=custom_legend,  # Pass the list directly without wrapping it in another list
        loc='upper right',
        bbox_to_anchor=(1, 1.05),  # Adjust these values as needed
        frameon=True,             # Turn off the frame around the legend
        prop=legend_font,          # Adjust font size as needed
        handletextpad=0.5,         # Spacing between the legend symbol and text
        labelspacing=0.5           # Spacing between rows of the legend
    )
    
    # # Customize the frame (facecolor sets the background color)
    legend.get_frame().set_facecolor('white')  # Set background color to white
    legend.get_frame().set_edgecolor('none')   # Remove the edge of the frame (border)
    legend.get_frame().set_alpha(1.0)     
    
    # Add a red line at y = 0
    ax.axhline(y=0.0, color='red', linewidth=1,linestyle='--')
    
    ax.axvline(7.5, color='black', linestyle='--', linewidth=1)

    # Set the y-limits based on the provided or calculated values
    if y_min is not None and y_max is not None:
        ax.set_ylim([y_min, y_max])
     
    # Hide the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
     
    for my_label in ax.get_xticklabels():
          my_label.set_fontproperties(font_text)
          my_label.set_fontsize(MEDIUM_SIZE)
    for my_label in ax.get_yticklabels():
          my_label.set_fontproperties(font_text)
          my_label.set_fontsize(MEDIUM_SIZE)

    # Adjust layout
    plt.tight_layout()
    
def plot_barplot_delta_old(fig, ax, test_name, 
                control_means, intervention_means, 
                control_cis, intervention_cis,
                batches_to_analyze, plot_title, font_title, font_text, 
                SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, 
                avg_control_means, avg_intervention_means,
                control_means_pc, intervention_means_pc,
                control_cis_pc, intervention_cis_pc,
                n_array, n_array_pc,
                ishealthy=None, y_min=None, y_max=None):
    
    if ishealthy is not None:
        mycolor = 'green'
    else:
        mycolor = 'blue'
    
    # Append "avg" and "PC" labels to the x-axis
    batch_labels = [f'\n\n\n\nB{i+1} \n (n={n_array[i]})' if (i+1) % 2 == 0 else f'B{i+1} \n (n={n_array[i]})' for i in range(len(n_array))]
    pc_labels =  [f'\n\n\n\n+{i+1} \n (n={n_array_pc[i]})' if (i+1) % 2 == 0 else f'+{i+1} \n (n={n_array_pc[i]})' for i in range(len(n_array_pc))]
    endash = "\u2013"
    batches_to_analyze = batch_labels + [f'\n\n\n\nB1{endash}{len(n_array)} \n (n={np.sum(n_array)})'] + pc_labels
    
    # Create the plot
    plt.suptitle(plot_title, fontsize=BIGGER_SIZE, fontproperties=font_title)

    # x positions for the batch data
    x = np.array(range(len(n_array)))  # 7 batches
    x_avg = np.array([len(n_array)])  # Position for 'avg' point
    x_pc = np.arange(len(n_array) + 1, len(n_array) + 1 + len(n_array_pc))  # Position for PC1, .,.. PCn
    width = 0.15  # Smaller spacing between pairs

    # Replace the bar plots with points and error bars
    ax.errorbar(x - width/2, control_means, yerr=control_cis, fmt='o', label='Control', color=mycolor, alpha=0.5, capsize=0, markersize=7)
    ax.errorbar(x + width/2, intervention_means, yerr=intervention_cis, fmt='s', label='Intervention', color=mycolor, capsize=0, markersize=7)

    # Plot the points and error bars for the average
    ax.errorbar(x_avg - width/2, [np.mean(avg_control_means)], yerr=[np.mean(control_cis)], fmt='o', label='Avg Control', color=mycolor, alpha=0.5, capsize=0, markersize=7)
    ax.errorbar(x_avg + width/2, [np.mean(avg_intervention_means)], yerr=[np.mean(intervention_cis)], fmt='s', label='Avg Intervention', color=mycolor, capsize=0, markersize=7)

    # Plot the points and error bars for the positive controls
    ax.errorbar(x_pc - width/2, control_means_pc, yerr=control_cis_pc, fmt='o', label='Control PC', color=mycolor, alpha=0.5, capsize=0, markersize=7)
    ax.errorbar(x_pc + width/2, intervention_means_pc, yerr=intervention_cis_pc, fmt='s', label='Intervention PC', color=mycolor, capsize=0, markersize=7)

    # # Add some text for labels, title, and custom x-axis tick labels, etc.
    # if test_name in ['area', 'perimeter', 'major axis', 'eye size', 'pigmentation size', 'curvature', 'curvature std', 'Frechet']:
    #     ax.set_ylabel('mean ' + test_name + ' (norm.)', fontproperties=font_text, fontsize=MEDIUM_SIZE)
    # else:
    #     ax.set_ylabel('mean ' + test_name, fontproperties=font_text, fontsize=MEDIUM_SIZE)
    
 
    # Set x-ticks and labels
    all_ticks = list(x) + list(x_avg) + list(x_pc)
    ax.set_xticks(all_ticks)
    
    #ax.set_xticklabels(batches_to_analyze, rotation=45, ha='center', va='center')
    ax.set_xticklabels(batches_to_analyze, rotation=0, ha='center', va='center')
    ax.tick_params(axis='x', pad=15)  # Increase the pad value to move the labels lower
    
    # Modify tick length for odd and even batches
    for i, tick in enumerate(ax.xaxis.get_major_ticks()):
        if (i + 1) % 2 == 0:  # Odd batches
            tick.tick1line.set_markersize(30)  # Longer ticks
            tick.tick2line.set_markersize(30)  # Longer ticks

    # Add the legend for "Control" and "Hypo" conditions
    custom_legend = [
        plt.Line2D([0], [0], color=mycolor, marker='o', alpha=0.5, lw=2, label='avg. control ± 95% conf. int.'),
        plt.Line2D([0], [0], color=mycolor, marker='s', lw=2, label='avg. hypo ± 95% conf. int.')
    ]
    legend_font = FontProperties(fname=font_text.get_file(), size=MEDIUM_SIZE)
    #ax.legend(handles=custom_legend, loc='upper right', prop=legend_font, frameon=False)
    
    # Assuming legend_elements is already a list of handles
    legend = ax.legend(
        handles=custom_legend,  # Pass the list directly without wrapping it in another list
        loc='upper right',
        bbox_to_anchor=(1, 1.05),  # Adjust these values as needed
        frameon=True,             # Turn off the frame around the legend
        prop=legend_font,          # Adjust font size as needed
        handletextpad=0.5,         # Spacing between the legend symbol and text
        labelspacing=0.5           # Spacing between rows of the legend
    )
    
    # # Customize the frame (facecolor sets the background color)
    legend.get_frame().set_facecolor('white')  # Set background color to white
    legend.get_frame().set_edgecolor('none')   # Remove the edge of the frame (border)
    legend.get_frame().set_alpha(1.0)     
    
    # Add a red line at y = 0
    ax.axhline(y=0.0, color='red', linewidth=1,linestyle='--')
    
    ax.axvline(7.5, color='black', linestyle='--', linewidth=1)

    # Set the y-limits based on the provided or calculated values
    if y_min is not None and y_max is not None:
        ax.set_ylim([y_min, y_max])
     
    # Hide the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
     
    for my_label in ax.get_xticklabels():
          my_label.set_fontproperties(font_text)
          my_label.set_fontsize(MEDIUM_SIZE)
    for my_label in ax.get_yticklabels():
          my_label.set_fontproperties(font_text)
          my_label.set_fontsize(MEDIUM_SIZE)

    # Adjust layout
    plt.tight_layout()
    #fig.tight_layout()


def rename_labels(conditionlabels, PositiveControls):
    # If PositiveControls > 0, rename the last items as PC
    if PositiveControls > 0:
        for i in range(PositiveControls):
            conditionlabels[-(PositiveControls*2) + i*2] = f'+{i+1}C'
            conditionlabels[-(PositiveControls*2) + i*2 + 1] = f'+{i+1}H'
    
    return conditionlabels


def process_arrays_old(concatenated, stickcolors, positivecontrols):
    """
    This function processes arrays based on the value of positivecontrols.
    
    Parameters:
    - concatenated: List of arrays to be processed
    - stickcolors: List of colors to be adjusted
    - positivecontrols: Integer indicating how many of the last even and odd arrays to skip
    
    Returns:
    - even_arrays: Concatenated even-numbered arrays
    - odd_arrays: Concatenated odd-numbered arrays
    - stickcolors: Adjusted stickcolors for even and odd arrays
    """

    # Ensure that positivecontrols is not larger than the number of even/odd arrays available
    max_len = len(concatenated) // 2  # Maximum number of even or odd arrays
    if positivecontrols > max_len:
        raise ValueError(f"positivecontrols cannot be greater than {max_len} based on array length.")

    # Concatenate arrays, skipping the last 'positivecontrols' arrays for both even and odd
    even_arrays = np.concatenate([concatenated[i] for i in range(0, len(concatenated) - 2 * positivecontrols, 2)])
    odd_arrays = np.concatenate([concatenated[i] for i in range(1, len(concatenated) - 2 * positivecontrols, 2)])

    # Flatten stickcolors
    stickcolors[0] = [color for sublist in stickcolors[0] for color in sublist]
    stickcolors[1] = [color for sublist in stickcolors[1] for color in sublist]

    # Adjust stickcolors to match the number of concatenated arrays
    stickcolors[0] = stickcolors[0][:len(even_arrays)]
    stickcolors[1] = stickcolors[1][:len(odd_arrays)]

    # Ensure that the length of stickcolors matches the concatenated arrays
    if len(stickcolors[0]) != len(even_arrays):
        raise ValueError("Mismatch between the number of colors in stickcolors[0] and the data in even_arrays")
    if len(stickcolors[1]) != len(odd_arrays):
        raise ValueError("Mismatch between the number of colors in stickcolors[1] and the data in odd_arrays")

    return even_arrays, odd_arrays, stickcolors

def generate_tuple(concatenated, positivecontrols):
    """
    Generates a tuple of the form ('B1--xC', 'B1--xH') where x is the length of the concatenated list
    minus the positivecontrols.
    
    Parameters:
    - concatenated: List of arrays
    - positivecontrols: Integer indicating how many controls to subtract from the total length
    
    Returns:
    - A tuple ('B1--xC', 'B1--xH') where x is calculated as len(concatenated) - positivecontrols
    """
    # Calculate x as the length of concatenated minus positivecontrols
    x = int((len(concatenated) / 2) - positivecontrols)
    
    # Use an en dash or em dash in the string
    endash = "\u2013"  # En dash
    
    # Return the tuple in the required format
    return [f'B1{endash}{x}C', f'B1{endash}{x}H']

def process_concatenated(concatenated):
    """
    Processes concatenated arrays in pairs (even and odd).
    
    Parameters:
    - concatenated: List of arrays
    
    Returns:
    - p_values: List of p-values (length will be half of concatenated)
    - test_types: List of test types ('T' for t-test, 'MW' for Mann-Whitney)
    """
    p_values = []  # To store p-values
    test_types = []  # To store whether it was a T-test or MW test
    my_alpha = 0.01  # Significance level for normality test

    # Iterate over pairs of even and odd arrays
    for i in range(0, len(concatenated), 2):
        even_array = concatenated[i]
        odd_array = concatenated[i + 1]

        # Check normality for even and odd arrays using Kolmogorov-Smirnov test
        stat_C, p_value_C = kstest(even_array, 'norm', args=(np.mean(even_array), np.std(even_array)))
        stat_H, p_value_H = kstest(odd_array, 'norm', args=(np.mean(odd_array), np.std(odd_array)))

        # Perform the appropriate test based on normality results
        if p_value_C >= my_alpha and p_value_H >= my_alpha:  # Both arrays are normal, do T-test
            t_stat, t_p_value = ttest_ind(even_array, odd_array, equal_var=False)
            p_values.append(t_p_value)
            test_types.append('T')
        else:  # One or both arrays are not normal, do Mann-Whitney U test
            mw_stat, mw_p_value = mannwhitneyu(even_array, odd_array, alternative='two-sided')
            p_values.append(mw_p_value)
            test_types.append('MW')

    return np.array(p_values), test_types

def split_arrays(p_values, PositiveControls):
    """
    Splits the p_values list into two arrays:
    - p_value_array: All elements except the last PositiveControls elements
    - p_value_array_pc: The last PositiveControls elements
    
    Parameters:
    - p_values: List of p-values
    - PositiveControls: Number of elements to include in p_value_array_pc
    
    Returns:
    - p_value_array: List containing p_values minus the last PositiveControls elements
    - p_value_array_pc: List containing the last PositiveControls elements
    """
    if PositiveControls == 0:
        # No split needed, p_value_array is the same as p_values and p_value_array_pc is empty
        p_value_array = p_values
        p_value_array_pc = []
    else:
        # Split the list based on PositiveControls
        p_value_array = p_values[:-PositiveControls]  # All except the last PositiveControls elements
        p_value_array_pc = p_values[-PositiveControls:]  # The last PositiveControls elements
    
    return p_value_array, p_value_array_pc

def plot_dunn_test_matrix(fig, ax, dunn_results, title, font_title, font_text, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, my_str, vmin, vmax, ishypo=False, ishealthy=False):
    """
    Plots both the upper half of the diagonal for the Dunn test result matrix with log10 scale and shared colorbar range.
    """
    # Extract the upper triangle of the matrix, excluding the diagonal
    dunn_matrix = np.triu(dunn_results, k=1)

    # Apply log10 transformation to the Dunn matrix
    with np.errstate(divide='ignore'):  # Ignore warnings about log10(0)
        dunn_log = np.log10(dunn_matrix)
        dunn_log[dunn_log == -np.inf] = np.nan  # Replace -inf with NaN for plotting purposes

    if ishealthy:
        my_cmap = 'Greens_r'
    else:
        my_cmap = 'Blues_r'

    # Plot log-transformed p-values with shared vmin and vmax
    #fig, ax = plt.subplots()
    sns.heatmap(
        dunn_log, annot=True, fmt='.1f', cmap=my_cmap, cbar_kws={'label': 'log(Dunn p value)', 'shrink': .5}, 
        square=True, mask=np.isnan(dunn_log), ax=ax, annot_kws={"fontsize": SMALL_SIZE, "fontproperties": font_text},
        vmin=vmin, vmax=vmax  # Set the shared colorbar range
    )
    
    # Set color bar and add a line at log(p value) == -2
    cbar = ax.collections[0].colorbar
    cbar.ax.axhline(y=-2, color='black', linewidth=1)
    cbar.set_label('log(Dunn p value)', rotation=270, labelpad=20, fontsize=MEDIUM_SIZE, fontproperties=font_text)
    cbar.ax.yaxis.set_tick_params(labelsize=MEDIUM_SIZE)
    for mylabel in cbar.ax.get_yticklabels():
        mylabel.set_fontproperties(font_text)

    # Add black frames around squares where the p-value is < 0.01
    for i in range(dunn_matrix.shape[0]):
        for j in range(i + 1, dunn_matrix.shape[1]):
            if dunn_matrix[i, j] < 0.01 and not np.isnan(dunn_matrix[i, j]):
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2))  # Increased linewidth for visibility

    # Ensure the limits include the full heatmap (to avoid clipping of rightmost squares)
    ax.set_xlim(0, dunn_matrix.shape[1])
    ax.set_ylim(dunn_matrix.shape[0], 0)  # Reverse order for proper display
    
    # Customize plot labels and title
    if ishypo:
        xticks_labels = [f"B{i+1}H" for i in range(1, dunn_matrix.shape[0] - 3)] + ['+1H', '+2H', '+3H'] 
        yticks_labels = [f"B{i}H" for i in range(1, dunn_matrix.shape[0] - 2)] + ['+1H', '+2H']
    else:
        xticks_labels = [f"B{i+1}C" for i in range(1, dunn_matrix.shape[0] -3)] + ['+1C', '+2C', '+3C']
        yticks_labels = [f"B{i}C" for i in range(1, dunn_matrix.shape[0] -2)] + ['+1C', '+2C']

    ax.set_xticks(np.arange(len(xticks_labels)) + 1.5)
    ax.set_yticks(np.arange(dunn_matrix.shape[0] - 1) + 0.5)
    
    ax.set_xticklabels(xticks_labels)#, fontsize=MEDIUM_SIZE, fontproperties=font_text)
    ax.set_yticklabels(yticks_labels, rotation=0) #, fontsize=MEDIUM_SIZE, fontproperties=font_text, rotation=0)
    for my_label in ax.get_xticklabels():
        my_label.set_fontproperties(font_text)
    for my_label in ax.get_yticklabels():
        my_label.set_fontproperties(font_text)
    ax.tick_params(labelsize=MEDIUM_SIZE)
    
    ax.tick_params(axis='both', which='both', length=0)  # Remove ticks
    
    # Adding the text box below the colorbar (adjusting position as needed)
    fig.text(0.5, 0.87, my_str, fontsize=MEDIUM_SIZE, fontproperties=font_text, 
             bbox=dict(facecolor='white', edgecolor='none'), ha='center')

    plt.suptitle(title, fontsize=BIGGER_SIZE, fontproperties=font_title, x=0.5, y=0.98, ha='center')
    plt.subplots_adjust(left=0.1, right=0.83, top=0.825) #right was 0.85 originally 
    
    #plt.show()

def plot_tukey_ARTanova(fig1, ax1, fig2, ax2, tukey_interaction_ARTANOVA, title, font_title, font_text, 
                        SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, my_str, vmin, vmax, numbatch, ishealthy=False):
    """
    Plots the Tukey ART-ANOVA posthoc test results for three cases:
    1. Control to Control
    2. Hypo to Hypo
    
    All comparisons are plotted based on the log10(p-adj) values between groups.
    """
    # Extract the relevant data from the Tukey object
    results = tukey_interaction_ARTANOVA.summary()

    # Filter the results for Control-Control, Hypo-Hypo, and Control-Hypo comparisons
    control_control_pairs = []
    hypo_hypo_pairs = []
    control_hypo_pairs = []

    # Mapping function to convert Tukey labels to plot labels
    def map_labels(group):
        if "Control" in group:
            group = group.replace("Control_Batch", "B") + "C"
        elif "Hypo" in group:
            group = group.replace("Hypo_Batch", "B") + "H"
        
        # Ensure zero padding
        if "B" in group:
            # Add zero padding to the batch numbers (e.g., B1 -> B01)
            parts = group.split("B")
            if len(parts) == 2 and len(parts[1]) > 0:
                batch_num = parts[1][:-1]  # Extract the batch number (ignores last char which is 'C' or 'H')
                suffix = parts[1][-1]  # 'C' or 'H'
                group = f"B{int(batch_num):02d}{suffix}"  # Pad with zero
        return group

    # Loop over Tukey results to separate into the three categories
    for row in results.data[1:]:  # Skip header
        group1, group2, meandiff, p_adj, lower, upper, reject = row
        group1_mapped = map_labels(group1)
        group2_mapped = map_labels(group2)
        
        # Debugging prints to check label mappings
        #print(f"Processing pair: {group1_mapped} vs {group2_mapped}, p_adj: {p_adj}")
        
        if "C" in group1_mapped and "C" in group2_mapped:
            control_control_pairs.append((group1_mapped, group2_mapped, p_adj))
        elif "H" in group1_mapped and "H" in group2_mapped:
            hypo_hypo_pairs.append((group1_mapped, group2_mapped, p_adj))
        elif ("C" in group1_mapped and "H" in group2_mapped) or ("H" in group1_mapped and "C" in group2_mapped):
            control_hypo_pairs.append((group1_mapped, group2_mapped, p_adj))

    # Convert the extracted pairs into matrices for plotting (log10(p-adj))
    def create_upper_diagonal_matrix(pairs, labels):
        size = len(labels)
        matrix = np.full((size, size), np.nan)  # Initialize with NaNs
        label_idx = {label: i for i, label in enumerate(labels)}  # Map labels to indices

        for group1, group2, p_adj in pairs:
            #print(f"Inserting {group1} vs {group2} at indices: {label_idx.get(group1, 'NA')} vs {label_idx.get(group2, 'NA')}")
            # Check if both group1 and group2 are in label_idx
            if group1 in label_idx and group2 in label_idx:
                i, j = label_idx[group1], label_idx[group2]
                if i < j:  # Only fill upper diagonal
                    matrix[i, j] = np.log10(p_adj)
            else:
                print(f"Warning: One of the groups {group1} or {group2} not found in labels.")

        return matrix

    # Customize plot labels and title -leave padding here
    hypo_labels = [f"B{i:02}H" for i in range(1,numbatch+1)]  # Zero-padded hypo labels
    control_labels = [f"B{i:02}C" for i in range(1, numbatch+1)]  # Zero-padded control labels
    
    # Create the matrices with only upper diagonal elements
    control_control_matrix = create_upper_diagonal_matrix(control_control_pairs, control_labels)
    hypo_hypo_matrix = create_upper_diagonal_matrix(hypo_hypo_pairs, hypo_labels)
    
    # Plot each matrix with formatting similar to the provided function
    def plot_matrix(matrix, title, labels, cmap, vmin, vmax, my_str, my_fig, my_ax, ishypo=False):
        sns.heatmap(
            matrix, annot=True, fmt='.1f', cmap=cmap, cbar_kws={'label': 'log(p-adj)', 'shrink': .5}, 
            square=True, mask=np.isnan(matrix), ax=my_ax, annot_kws={"fontsize": SMALL_SIZE, "fontproperties": font_text},
            vmin=vmin, vmax=vmax
        )
        
        # Set color bar and formatting
        cbar = my_ax.collections[0].colorbar
        cbar.ax.axhline(y=-2, color='black', linewidth=1)
        cbar.set_label('log(p adj.)', rotation=270, labelpad=20, fontsize=MEDIUM_SIZE, fontproperties=font_text)
        cbar.ax.yaxis.set_tick_params(labelsize=MEDIUM_SIZE)
        for mylabel in cbar.ax.get_yticklabels():
            mylabel.set_fontproperties(font_text)

        # Add black frames around squares where p-adj < 0.01
        for i in range(matrix.shape[0]):
            for j in range(i + 1, matrix.shape[1]):  # Only upper diagonal squares
                if matrix[i, j] < -2 and not np.isnan(matrix[i, j]):  # log10(p-adj) < -2
                    my_ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2))

        # Set labels
        my_ax.set_xticks(np.arange(len(labels)) + 0.5)
        my_ax.set_yticks(np.arange(len(labels)) + 0.5)
        
        
        # Set labels
        # Remove the first element from x-tick labels (B1) and replace with an empty string
        if ishypo:
            xticks_labels = [f"B{i}H" for i in range(1, numbatch+1)] #labels.copy()  # Copy the original labels
            yticks_labels = [f"B{i}H" for i in range(1, numbatch+1)] 
        else:
            xticks_labels = [f"B{i}C" for i in range(1, numbatch+1)] #labels.copy()  # Copy the original labels
            yticks_labels = [f"B{i}C" for i in range(1, numbatch+1)] 
        
        xticks_labels[0] = ''  # Replace the first x-tick with an empty string
        if ishypo:
            xticks_labels[-3] = '+1H' 
            xticks_labels[-2] = '+2H'
            xticks_labels[-1] = '+3H' 
        else:
            xticks_labels[-3] = '+1C'
            xticks_labels[-2] = '+2C'
            xticks_labels[-1] = '+3C'
        # Remove the last element from y-tick labels (PC1C) and replace with an empty string
        yticks_labels[-1] = ''  # Replace the last y-tick with an empty string
        if ishypo:
            yticks_labels[-3] = '+1H'
            yticks_labels[-2] = '+2H'
        else:
            yticks_labels[-3] = '+1C'
            yticks_labels[-2] = '+2C'
        
        my_ax.set_xticklabels(xticks_labels, fontsize=MEDIUM_SIZE, fontproperties=font_text)
        my_ax.set_yticklabels(yticks_labels, fontsize=MEDIUM_SIZE, fontproperties=font_text, rotation=0)
        
        my_ax.tick_params(axis='both', which='both', length=0)  # Remove ticks
        
        # Set color bar and add a line at log(p value) == -2
        cbar = my_ax.collections[0].colorbar
        cbar.ax.axhline(y=-2, color='black', linewidth=1)
        cbar.set_label('log(p value adj.)', rotation=270, labelpad=20, fontsize=MEDIUM_SIZE, fontproperties=font_text)
        cbar.ax.yaxis.set_tick_params(labelsize=MEDIUM_SIZE)
        for mylabel in cbar.ax.get_yticklabels():
            mylabel.set_fontproperties(font_text)
        
        # Add text box and title
        my_fig.text(0.5, 0.79, my_str, fontsize=SMALL_SIZE, fontproperties=font_text, #was 0.7775
                 bbox=dict(facecolor='white', edgecolor='none'), ha='center')

        my_fig.suptitle(title, fontsize=BIGGER_SIZE, fontproperties=font_title, x=0.5, y=0.98, ha='center')
        my_fig.subplots_adjust(left=0.1, right=0.83, top=0.75) #was 0.85
        
        #plt.show()

    # Plot Control-to-Control
    if ishealthy:
        my_cmap = 'Greens_r'
    else:
        my_cmap = 'Blues_r'
    
    plot_matrix(control_control_matrix, title, control_labels, my_cmap, vmin, vmax, my_str, fig1, ax1)
    # Plot Hypo-to-Hypo
    plot_matrix(hypo_hypo_matrix, title, hypo_labels, my_cmap, vmin, vmax, my_str, fig2, ax2, ishypo=True)

def adjust_tukey_p_values(tukey_result):
    """
    Adjust the p-values in the Tukey HSD result to handle very small values (0)
    by replacing them with scientific notation values, while keeping the original object structure.
    
    Parameters:
    tukey_result (TukeyHSDResults): The result from `pairwise_tukeyhsd`.
    
    Returns:
    TukeyHSDResults: The same TukeyHSDResults object, but with adjusted p-values.
    """
    # Modify the p-values in-place in the TukeyHSDResults object
    for i, p_adj in enumerate(tukey_result.pvalues):
        if p_adj == 0:
            # Replace 0 with the smallest representable float
            tukey_result.pvalues[i] = np.finfo(float).eps
    
    return tukey_result

def get_global_min_max(tukey_result1, tukey_result2):
    """
    Extract log10(p-adj) values from two Tukey HSD results and find the global min and max.
    
    Returns:
    - vmin: The global minimum log10(p-adj).
    - vmax: The global maximum log10(p-adj).
    """
    def extract_log_padj(tukey_result):
        # Get the Tukey HSD summary
        results = tukey_result.summary()

        log_padj_values = []

        # Extract the p-adj values and apply log10 transformation
        for row in results.data[1:]:  # Skip the header
            group1, group2, meandiff, p_adj, lower, upper, reject = row
            if p_adj > 0:  # To avoid log10(0)
                log_padj_values.append(np.log10(p_adj))
        
        return np.array(log_padj_values)
    
    # Get log10(p-adj) values from both Tukey results
    log_padj_1 = extract_log_padj(tukey_result1)
    log_padj_2 = extract_log_padj(tukey_result2)
    
    # Calculate global min and max
    global_min = np.nanmin(np.concatenate([log_padj_1, log_padj_2]))
    global_max = np.nanmax(np.concatenate([log_padj_1, log_padj_2]))
    
    return global_min, global_max
                   


    



def find_first_non_transparent_pixel2(image, alpha_threshold=1e-5):
    """Find the x-coordinate of the first non-transparent pixel in the image."""
    alpha_channel = image[..., 3]  # Assuming the 4th channel is the alpha
    non_transparent_cols = np.where(np.max(alpha_channel, axis=0) > alpha_threshold)[0]
    if len(non_transparent_cols) > 0:
        return non_transparent_cols[0]
    return 0  # Default to 0 if no non-transparent pixel found


def find_first_non_transparent_pixel2_new(image, alpha_threshold=1e-5):
    # Since we are working with binarized images (3 channels, no alpha), 
    # consider any non-zero pixel as "non-transparent"
    grayscale_image = np.mean(image, axis=-1)  # Convert the 3-channel RGB image to grayscale
    non_transparent = np.any(grayscale_image > alpha_threshold, axis=0)  # Find non-transparent columns
    
    # Find the first non-transparent column
    first_non_transparent = np.argmax(non_transparent)
    
    return first_non_transparent


    
def plot_D2_vertical(fig, num_images1, num_images2, images_with_overlay, plot_title, footer_txt, framecolors, BIGGER_SIZE, MEDIUM_SIZE, font_title, font_text, hull_images=None, alpha_threshold=1e-5):
    # Set the title
    plt.suptitle(plot_title, fontsize=BIGGER_SIZE, fontproperties=font_title)

    # Calculate the number of rows based on the larger of num_images1 and num_images2
    num_rows = max(num_images1, num_images2)

    # Set up GridSpec with 2 columns (Control on the left, Hypo on the right) and dynamically calculated rows
    gs = gridspec.GridSpec(num_rows, 2, width_ratios=[1, 1], wspace=0.2, hspace=0.02)

    # Function to shift the image content to the left
    def shift_image_left(image, alpha_threshold=1e-5):
        # Find the first non-transparent pixel
        first_non_transparent = find_first_non_transparent_pixel2(image, alpha_threshold)
        if first_non_transparent == 0:
            return image  # No need to shift if it's already aligned

        # Shift the image data to the left
        shifted_image = np.zeros_like(image)
        width = image.shape[1]
        shifted_image[:, :width - first_non_transparent] = image[:, first_non_transparent:]

        return shifted_image

    # Plot images from the first folder (Control, left column)
    thick = 20 #was 10
    for idx in range(num_images1):
        ax = plt.subplot(gs[idx, 0])  # Always plot in the first column (Control)
        shifted_image = shift_image_left(images_with_overlay[0][idx])
        images_with_overlay[0][idx] = add_frame(shifted_image, framecolors[0][idx], thick)  # Add frame

        if hull_images:
            shifted_hull_image = shift_image_left(hull_images[0][idx])
            ax.imshow(shifted_hull_image, cmap='Blues')
            ax.imshow(images_with_overlay[0][idx], alpha=0.3)
        else:
            ax.imshow(images_with_overlay[0][idx])

        # Apply a colored frame around the image using `framecolors`
        my_color = framecolors[0][idx]
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(my_color)
            spine.set_linewidth(8)  # Increased width for visibility
        
        ax.axis('off')

    # Plot images from the second folder (Hypo, right column)
    for idx in range(num_images2):
        ax = plt.subplot(gs[idx, 1])  # Always plot in the second column (Hypo)
        shifted_image = shift_image_left(images_with_overlay[1][idx])
        images_with_overlay[1][idx] = add_frame(shifted_image, framecolors[1][idx], thick)  # Add frame

        if hull_images:
            shifted_hull_image = shift_image_left(hull_images[1][idx])
            ax.imshow(shifted_hull_image, cmap='Blues')
            ax.imshow(images_with_overlay[1][idx], alpha=0.3)
        else:
            ax.imshow(images_with_overlay[1][idx])

        # Apply a colored frame around the image using `framecolors`
        color = framecolors[1][idx]
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(8)  # Increased width for visibility
        
        ax.axis('off')

    # Add a vertical line exactly in the middle between the two columns
    middle_x = 0.5  # Middle of the figure in normalized coordinates
    line_height = 0.85  # Height of the line as a fraction of the figure height
    line_y_start = 0.05
    ax_line = fig.add_axes([middle_x - 0.002, line_y_start, 0.004, line_height], frameon=False)
    ax_line.axvline(x=0.5, color='black', linestyle='-', linewidth=5)
    ax_line.axis('off')

    # Adjust layout
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.01, hspace=0.01)

    # Add a line of text at the bottom of the figure
    fig.text(0.5, 0, footer_txt, ha='center', fontsize=MEDIUM_SIZE, fontproperties=font_text)

    # Force drawing update
    plt.draw()



def process_image_NEW(image_path, target_size=None):
    # Process each image to find regions, draw major and minor axes, and ensure major axis is horizontal
    image = Image.open(image_path).convert("RGBA")
    data = np.array(image)

    non_transparent = data[..., 3] > 0
    if not np.any(non_transparent):
        return 0, 0, 0, 0, image  # No non-transparent pixels found

    labeled_image = label(non_transparent)
    regions = regionprops(labeled_image)
    if not regions:
        return np.sum(non_transparent), 0, 0, 0, data  # No regions found

    region = max(regions, key=lambda r: r.area)
    orientation = region.orientation
    major_length = region.major_axis_length
    minor_length = region.minor_axis_length
    
    # Calculate elongation
    elongation = calculate_elongation(major_length, minor_length)
    # Calculate roundness
    roundness = calculate_roundness(np.sum(non_transparent), region.perimeter)
    # Calculate eccentricity
    eccentricity = calculate_eccentricity(major_length, minor_length)
    # Calculate area/perimeter
    AdivP = calculate_AdivP(np.sum(non_transparent), region.perimeter)
    
    # Rotate image to make the major axis horizontal
    rotation_angle = -np.degrees(orientation) + 90  # Rotate additional 90 degrees to make major axis horizontal
    rotated_image = rotate(data, rotation_angle, resize=True, mode='edge')
    
    # Save a copy of the rotated image before drawing overlays (this is the image without overlay)
    rotated_image_no_overlay = np.copy(rotated_image)
    
    alpha_channel = rotated_image[:, :, 3]
    binary_image = alpha_channel > 0
    area = np.sum(binary_image)
    
    convex_hull = convex_hull_image(binary_image)
    convex_hull_area = convex_hull.sum()
    solidity = area / convex_hull_area
    
    object_perimeter = measure.perimeter(binary_image)  # Perimeter of the object
    # Calculate the percentage difference
    #percentage_difference = np.abs(object_perimeter - region.perimeter) / region.perimeter * 100
    # Check if the percentage difference is greater than 5%
    #if percentage_difference > 5:
     #   raise ValueError(f"Perimeters differ by more than 5%. "
      #                   f"measure.perimeter: {object_perimeter}, "
       #                  f"region.perimeter: {region.perimeter}")
    
    convex_hull_perimeter = measure.perimeter(convex_hull)  # Perimeter of the convex hull
    # Calculate convexity (convex hull perimeter / object perimeter)
    convexity = convex_hull_perimeter / object_perimeter

    # Redraw overlays on the rotated image
    rotated_non_transparent = rotated_image[..., 3] > 0
    rotated_labeled_image = label(rotated_non_transparent)
    rotated_regions = regionprops(rotated_labeled_image)      
    if rotated_regions:
        rotated_region = max(rotated_regions, key=lambda r: r.area)
        new_centroid = rotated_region.centroid
    
        # Draw perimeter
        contours = find_contours(rotated_labeled_image, level=0.5)
        for contour in contours:
            rr, cc = contour[:, 0], contour[:, 1]
            draw_thick_line(rotated_image, rr.astype(int), cc.astype(int), [0, 0, 0, 255], 10)  # Black for the perimeter

        # Draw axes
        major_axis_endpoints = calculate_endpoints(new_centroid, 0, rotated_region.major_axis_length)  # Major axis horizontal
        minor_axis_endpoints = calculate_endpoints(new_centroid, np.pi / 2, rotated_region.minor_axis_length)  # Minor axis vertical

        rr, cc = line(*major_axis_endpoints[0], *major_axis_endpoints[1])
        draw_thick_line(rotated_image, rr, cc, [0, 0, 0, 255], 10)  # Black for major axis

        rr, cc = line(*minor_axis_endpoints[0], *minor_axis_endpoints[1])
        draw_thick_line(rotated_image, rr, cc, [0, 0, 0, 255], 10)  # Black for minor axis

        # Calculate Total Curvature and Standard Deviation of Curvature
        curvatures = calculate_curvature(contours[0])  # Assuming single largest contour
        total_curvature = np.sum(np.abs(curvatures))
        curvature_std = np.std(curvatures)
        mean_curvature = np.mean(np.abs(curvatures))
        max_curvature = np.max(np.abs(curvatures))
        skewness_curvature = skew(curvatures)
        kurtosis_curvature = kurtosis(curvatures)
        rms_curvature = np.sqrt(np.mean(curvatures**2))
        
        # Correct reference for contour in arc length calculation
        arc_length = np.sum(np.linalg.norm(np.diff(contours[0], axis=0), axis=1))
        normalized_curvature = total_curvature / arc_length

        # Handle divisions by zero in radius of curvature calculation
        with np.errstate(divide='ignore', invalid='ignore'):
            radius_of_curvature = np.where(curvatures != 0, 1 / np.abs(curvatures), np.inf)
        # Filter out infinite values from radius_of_curvature
        finite_radius_of_curvature = radius_of_curvature[np.isfinite(radius_of_curvature)]
        # Compute the mean radius of curvature (ignoring infinite values)
        mean_radius_of_curvature = np.mean(finite_radius_of_curvature) if len(finite_radius_of_curvature) > 0 else np.nan
        
        # Bounding Box Aspect Ratio
        minr, minc, maxr, maxc = region.bbox
        bounding_box_aspect_ratio = (maxc - minc) / (maxr - minr)
        
        # Fréchet Distance (or Directed Hausdorff)
        contour_coords = contours[0]
        convex_hull_coords = np.column_stack(np.nonzero(convex_hull))
        frechet_distance = calculate_frechet_distance(contour_coords, convex_hull_coords)

        return np.sum(rotated_non_transparent), rotated_region.perimeter, rotated_region.minor_axis_length, rotated_region.major_axis_length, convex_hull_area, elongation, roundness, eccentricity, solidity, rotated_image, convex_hull, total_curvature, curvature_std, bounding_box_aspect_ratio, frechet_distance, rotated_image_no_overlay, AdivP, convexity, mean_curvature, max_curvature, skewness_curvature, kurtosis_curvature, rms_curvature, normalized_curvature, mean_radius_of_curvature
    
    return area, region.perimeter, region.minor_axis_length, region.major_axis_length, convex_hull_area, elongation, roundness, eccentricity, solidity, rotated_image, convex_hull, 0, 0, 0, 0, rotated_image_no_overlay, AdivP, convexity, mean_curvature, max_curvature, skewness_curvature, kurtosis_curvature, rms_curvature, normalized_curvature, mean_radius_of_curvature

# Helper functions for curvature, Fréchet distance, etc.
def calculate_curvature(contour):
    # You can approximate curvature from the contour using second derivatives or difference methods
    diff = np.diff(contour, axis=0)
    second_diff = np.diff(diff, axis=0)
    curvature = np.linalg.norm(second_diff, axis=1)
    return curvature

def calculate_frechet_distance(contour1, contour2):
    # Use Hausdorff as an approximation of Fréchet distance
    return max(directed_hausdorff(contour1, contour2)[0], directed_hausdorff(contour2, contour1)[0])

def sort_by_property(filenames, images_with_overlay, hull_images, framecolors, property_array, images_without_overlay=None, bin_images=None, increasing=True):
    """
    Sorts the images and related data based on the given property array.
    
    Parameters:
    - filenames: A list of filenames to be sorted.
    - images_with_overlay: A list of images to be sorted.
    - hull_images: A list of hull images to be sorted.
    - framecolors: A list of framecolors to be sorted.
    - property_array: The property array by which sorting is performed (e.g., solidities, elongations, etc.).
    - increasing: A boolean indicating if sorting should be in increasing order (default is True).

    Returns:
    - sorted_filenames: The sorted filenames.
    - sorted_property: The sorted property array.
    - sorted_images_with_overlay: The sorted images.
    - sorted_hull_images: The sorted hull images.
    - sorted_framecolors: The sorted framecolors.
    """
    # Determine the sorting order: ascending or descending
    sorted_indices = np.argsort(property_array) if increasing else np.argsort(property_array)[::-1]
    
    # Sort all the associated lists using the sorted indices
    sorted_filenames = [filenames[i] for i in sorted_indices]
    sorted_property = [property_array[i] for i in sorted_indices]
    sorted_images_with_overlay = [images_with_overlay[i] for i in sorted_indices]
    sorted_hull_images = [hull_images[i] for i in sorted_indices]
    sorted_framecolors = [framecolors[i] for i in sorted_indices]
    if images_without_overlay is not None:
        sorted_images_without_overlay = [images_without_overlay[i] for i in sorted_indices]
    if bin_images is not None:
        sorted_bin_images = [bin_images[i] for i in sorted_indices]
    
    
    if images_without_overlay is None:
        return sorted_filenames, sorted_property, sorted_images_with_overlay, sorted_hull_images, sorted_framecolors
    else:
        if bin_images is None:
            return sorted_filenames, sorted_property, sorted_images_with_overlay, sorted_hull_images, sorted_framecolors, sorted_images_without_overlay
        else:
            return sorted_filenames, sorted_property, sorted_images_with_overlay, sorted_hull_images, sorted_framecolors, sorted_images_without_overlay, sorted_bin_images


def binarize_image_non_transparent(image, alpha_threshold=1e-5):
    # Binarize based on non-transparent pixels (where alpha > 0)
    alpha_channel = image[..., 3]  # Extract the alpha channel
    binary_image = (alpha_channel > alpha_threshold).astype(np.uint8) * 255  # Convert to 0 or 255 for binary
    
    # Convert the 2D binary image into a 3D image by stacking the binary values into three channels (RGB)
    binary_image_3d = np.stack([binary_image] * 3, axis=-1)  # Create a grayscale 3-channel image
    return binary_image_3d


def binarize_images_non_transparent(images, alpha_threshold=1e-5):
    binarized_images = []
    for image_list in images:
        binarized_image_list = [binarize_image_non_transparent(image, alpha_threshold) for image in image_list]
        binarized_images.append(binarized_image_list)
    return binarized_images

def plot_D2_vertical_binary(fig, num_images1, num_images2, images_with_overlay, plot_title, footer_txt, framecolors, BIGGER_SIZE, MEDIUM_SIZE, font_title, font_text):
    """Plot binarized images in two columns (Control and Hypo) with shifted content."""
    # Set the title
    plt.suptitle(plot_title, fontsize=BIGGER_SIZE, fontproperties=font_title)

    # Calculate the number of rows based on the larger of num_images1 and num_images2
    num_rows = max(num_images1, num_images2)

    # Set up GridSpec with 2 columns (Control on the left, Hypo on the right) and dynamically calculated rows
    gs = gridspec.GridSpec(num_rows, 2, width_ratios=[1, 1], wspace=0.2, hspace=0.02)

    # Function to shift the image content to the left
    def shift_image_left(image):
        """Shift non-transparent content of the image to the left."""
        # Convert to grayscale
        alpha_channel = image[..., 3]
        non_transparent_columns = np.any(alpha_channel > 0, axis=0)
        first_non_transparent = np.argmax(non_transparent_columns)
        if first_non_transparent == 0:
            return image  # No need to shift if already aligned

        # Shift the image to the left
        shifted_image = np.zeros_like(image)
        width = image.shape[1]
        shifted_image[:, :width - first_non_transparent] = image[:, first_non_transparent:]

        return shifted_image

    # Plot images from the first folder (Control, left column)
    for idx in range(num_images1):
        ax = plt.subplot(gs[idx, 0])  # Always plot in the first column (Control)
        shifted_image = shift_image_left(images_with_overlay[0][idx])
        ax.imshow(shifted_image)
        ax.axis('off')

    # Plot images from the second folder (Hypo, right column)
    for idx in range(num_images2):
        ax = plt.subplot(gs[idx, 1])  # Always plot in the second column (Hypo)
        shifted_image = shift_image_left(images_with_overlay[1][idx])
        ax.imshow(shifted_image)
        ax.axis('off')

    # Add a vertical line exactly in the middle between the two columns
    middle_x = 0.5  # Middle of the figure in normalized coordinates
    line_height = 0.85  # Height of the line as a fraction of the figure height
    line_y_start = 0.05
    ax_line = fig.add_axes([middle_x - 0.002, line_y_start, 0.004, line_height], frameon=False)
    ax_line.axvline(x=0.5, color='black', linestyle='-', linewidth=5)
    ax_line.axis('off')

    # Adjust layout
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.01, hspace=0.01)

    # Add a line of text at the bottom of the figure
    fig.text(0.5, 0, footer_txt, ha='center', fontsize=MEDIUM_SIZE, fontproperties=font_text)

    # Force drawing update
    plt.draw()


def add_frame_binary(image, framecolor, thickness):
    # If the image has more than 2 dimensions (e.g., RGB or grayscale stacked), convert to 2D
    if image.ndim == 3:
        image = image[..., 0]  # Extract a single channel (since it's binary, all channels are the same)
    
    # Ensure the image is now 2D
    height, width = image.shape

    # Create a new image with the frame
    framed_image = np.zeros((height + 2 * thickness, width + 2 * thickness), dtype=np.uint8)

    # Map framecolor to grayscale if it's a string
    color_map = {
        'black': 0,
        'white': 255,
        'gray': 127,
        'red': 76,    # Red's luminance in grayscale
        'green': 150, # Green's luminance in grayscale
        'blue': 29,   # Blue's luminance in grayscale
        'orange': 191 # Orange's luminance (approximation)
    }
    
    if isinstance(framecolor, str):
        framecolor = color_map.get(framecolor.lower(), 127)  # Default to gray if unknown color

    # Fill the frame area with the frame color
    framed_image[:, :] = framecolor

    # Place the original binary image in the center of the framed image
    framed_image[thickness:thickness + height, thickness:thickness + width] = image

    return framed_image

def binarize_imageNEW(image, threshold):
    """Binarize an image with transparency.
    - Transparent pixels become red.
    - Pixels below the threshold become white.
    - Pixels above the threshold become black.
    """
    # Separate the RGBA channels
    rgba_image = np.array(image)
    red, green, blue, alpha = rgba_image[..., 0], rgba_image[..., 1], rgba_image[..., 2], rgba_image[..., 3]
    
    # Create a new output image with RGBA channels
    output_image = np.zeros_like(rgba_image)

    # Find non-transparent pixels (alpha > 0)
    non_transparent_mask = alpha > 0

    # Convert the RGB to grayscale using the luminance formula
    grayscale_image = 0.299 * red + 0.587 * green + 0.114 * blue

    # Apply the threshold: Pixels below become white, above become black
    below_threshold_mask = grayscale_image < threshold
    above_threshold_mask = grayscale_image >= threshold

    # Assign black and white based on the threshold
    output_image[non_transparent_mask & below_threshold_mask] = [0, 0, 0, 255]        # Black
    output_image[non_transparent_mask & above_threshold_mask] = [255, 255, 255, 255]  # White

    # Transparent pixels get assigned red
    output_image[~non_transparent_mask] = [255, 0, 0, 255]  # Red for transparency

    return output_image

def calculate_global_threshold2(images):
    """Calculate a global threshold based on the combined histogram of all non-transparent pixels in the images."""
    combined_histogram = np.zeros(256)  # Assuming 8-bit grayscale images
    
    # Loop through each image to build the combined histogram
    for image in images:
        if image.shape[2] == 4:  # Check if the image has an alpha channel
            rgb = image[..., :3]    # Extract the RGB channels
            alpha_channel = image[..., 3]  # Extract the alpha channel
            
            # Only process non-transparent pixels (alpha > 0)
            non_transparent_mask = alpha_channel > 0
            non_transparent_rgb = rgb[non_transparent_mask]
            
            if non_transparent_rgb.size == 0:
                continue  # Skip if no non-transparent pixels
            
            # Convert the RGB to grayscale
            grayscale_image = rgb2gray(non_transparent_rgb) * 255  # Convert to 8-bit grayscale
            grayscale_image = grayscale_image.astype(np.uint8)  # Convert to integer type
            
            # Build histogram for the non-transparent pixels
            histogram, _ = np.histogram(grayscale_image, bins=256, range=(0, 256))
            combined_histogram += histogram
        else:
            rgb = image
            grayscale_image = rgb2gray(rgb) * 255  # Convert to 8-bit grayscale
            grayscale_image = grayscale_image.astype(np.uint8)
            histogram, _ = np.histogram(grayscale_image, bins=256, range=(0, 256))
            combined_histogram += histogram
    
    # Plot the combined histogram for visualization if desired
    plt.figure(figsize=(10, 6))
    plt.plot(combined_histogram, label='Combined Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Combined Histogram of Non-Transparent Pixels')
    plt.legend()
    plt.show()

    # Calculate meaningful threshold
    # Option 1: Use the mean of the histogram
    mean_threshold = np.mean(np.nonzero(combined_histogram))
    
    # Option 2: Use the midpoint between peaks (if multiple peaks are present)
    peaks = np.where((combined_histogram[1:-1] > combined_histogram[:-2]) & 
                     (combined_histogram[1:-1] > combined_histogram[2:]))[0] + 1
    
    if len(peaks) > 1:
        midpoint_threshold = (peaks[0] + peaks[-1]) // 2
    else:
        midpoint_threshold = mean_threshold
    
    # Choose the best threshold to use
    chosen_threshold = int(midpoint_threshold)
    
    return chosen_threshold

def calculate_otsu_threshold(images):
    """Calculate a global threshold using Otsu's method."""
    combined_pixels = []
    
    # Loop through each image to gather all non-transparent pixels
    for image in images:
        if image.shape[2] == 4:  # Check if the image has an alpha channel
            rgb = image[..., :3]    # Extract the RGB channels
            alpha_channel = image[..., 3]  # Extract the alpha channel
            non_transparent_mask = alpha_channel > 0
            non_transparent_rgb = rgb[non_transparent_mask]
            
            if non_transparent_rgb.size == 0:
                continue  # Skip if no non-transparent pixels
            
            grayscale_image = rgb2gray(non_transparent_rgb) * 255
            combined_pixels.append(grayscale_image)
    
    # Flatten the list of grayscale pixels and calculate the threshold
    combined_pixels = np.concatenate(combined_pixels)
    otsu_threshold = threshold_otsu(combined_pixels)
    
    return otsu_threshold

def binarize_imageNEW2(image, threshold):
    """Binarize an image with transparency.
    - Transparent pixels become red.
    - Pixels below the threshold become white.
    - Pixels above the threshold become black.
    """
    # Ensure the input image is in uint8 format for PIL processing
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)  # Convert float64 or other types to uint8

    # Convert the numpy array to a PIL Image and handle transparency
    pil_image = Image.fromarray(image).convert("RGBA")
    
    # Convert the PIL image to RGB (ignore the alpha channel for binarization)
    rgb_image = pil_image.convert('RGB')

    # Convert the RGB image back to a NumPy array
    rgb_image_np = np.array(rgb_image)

    # Separate the RGB channels
    red, green, blue = rgb_image_np[..., 0], rgb_image_np[..., 1], rgb_image_np[..., 2]

    # Convert the RGB to grayscale using the luminance formula
    grayscale_image = 0.299 * red + 0.587 * green + 0.114 * blue

    # Create a new output image with RGBA channels (same shape as the original)
    output_image = np.zeros_like(image)

    # Extract the alpha channel from the original image
    alpha = image[..., 3]

    # Find non-transparent pixels (alpha > 0)
    non_transparent_mask = alpha > 0

    # Apply the threshold: Pixels below become white, above become black
    below_threshold_mask = grayscale_image < threshold
    above_threshold_mask = grayscale_image >= threshold

    # Assign black and white based on the threshold
    output_image[non_transparent_mask & below_threshold_mask] = [0, 0, 0, 255]        # Black
    output_image[non_transparent_mask & above_threshold_mask] = [255, 255, 255, 255]  # White

    # Transparent pixels get assigned red
    output_image[~non_transparent_mask] = [255, 0, 0, 255]  # Red for transparency

    return output_image





def mask_image_by_binary(binary_image, color_image):
    """
    Mask the color image by the binary image. Only keep regions in the color image
    where the binary image is white (True or 1).
    """
    # Ensure binary image is boolean
    binary_mask = binary_image > 0

    # Apply the binary mask to the color image (masking only RGB channels, not alpha)
    masked_image = np.zeros_like(color_image)
    masked_image[binary_mask] = color_image[binary_mask]

    return masked_image





def quantify_yellowness(images_with_overlay):
    num_images1 = len(images_with_overlay[0])
    num_images2 = len(images_with_overlay[1])

    # Arrays to store results
    rgb_yellowness = [[], []]
    yellowness_index_cie = [[], []]
    lab_b_yellowness = [[], []]
    hsv_yellowness = [[], []]

    # Function to compute yellowness in different ways
    def process_image(image, folder_idx, image_idx):
        # Ensure the image data is in uint8 format
        image_uint8 = (image * 255).astype(np.uint8) if image.dtype != np.uint8 else image
        
        # Convert the numpy array to a PIL Image and handle transparency
        pil_image = Image.fromarray(image_uint8).convert("RGBA")
        data = np.array(pil_image)

        # Separate the RGB and alpha channels
        rgb_image = data[..., :3]  # RGB part
        alpha_channel = data[..., 3]  # Alpha (transparency) part

        # Create a non-transparent mask based on the alpha channel
        non_transparent_mask = alpha_channel > 0
        if not np.any(non_transparent_mask):
            return None, None, None, None

        # Apply the non-transparent mask to the RGB image
        masked_rgb_image = np.zeros_like(rgb_image)
        masked_rgb_image[non_transparent_mask] = rgb_image[non_transparent_mask]

        rgb_image_non_transparent = masked_rgb_image[non_transparent_mask] / 255.0  # Normalize RGB values
        rgb_non_norm = masked_rgb_image[non_transparent_mask]

        # 1) RGB Yellowness (Y = (R + G) / 2 - B)
        # RGB normalized
        #R = rgb_image_non_transparent[:, 0]
        #G = rgb_image_non_transparent[:, 1]
        #B = rgb_image_non_transparent[:, 2]
        # Not RGB normalized
        R = rgb_non_norm[:, 0]
        G = rgb_non_norm[:, 1]
        B = rgb_non_norm[:, 2]
        rgb_yellowness_value = np.mean((R + G) / 2 - B)

        # 2) Yellowness Index (YI) for D65 Illuminant in CIE XYZ
        xyz_image = rgb2xyz(rgb_image_non_transparent) # RGB normalized
        X = xyz_image[:, 0]
        Y = xyz_image[:, 1]
        Z = xyz_image[:, 2]
        
        # Handle zero values in Y to avoid division by zero
        valid_Y_mask = Y > 0
        if np.any(valid_Y_mask):
            yellowness_index = 100 * ((1.28 * X[valid_Y_mask]) - (1.06 * Z[valid_Y_mask])) / Y[valid_Y_mask]
            yellowness_index_value = np.mean(yellowness_index)
        else:
            yellowness_index_value = 0  # No valid Y values

        # 3) Lab b* channel (blue-yellow axis)
        lab_image = rgb2lab(rgb_image_non_transparent) # RGB normalized
        b_channel = lab_image[:, 2]  # b* channel
        lab_b_yellowness_value = np.mean(b_channel)

        # 4) HSV Yellowness (Hue between 45° and 75°)
        hsv_image = rgb2hsv(rgb_image_non_transparent) # RGB normalized
        hue = hsv_image[:, 0] * 360  # Hue in degrees
        saturation = hsv_image[:, 1]
        value = hsv_image[:, 2]

        # Filter for yellow hue (45° to 75°)
        yellow_mask = (hue >= 45) & (hue <= 75)
        if np.any(yellow_mask):
            hue_yellow_mean = np.mean(saturation[yellow_mask] * value[yellow_mask])
        else:
            hue_yellow_mean = 0  # No yellow found

        # Debugging: Plot the mask and the non-transparent image part
        plt.figure(figsize=(12, 4))

        # Plot the RGB image with the applied mask
        plt.subplot(1, 2, 1)
        plt.imshow(masked_rgb_image)
        plt.title(f"RGB Image with Mask (Image {image_idx} - Folder {folder_idx})")

        # Plot the non-transparent mask
        plt.subplot(1, 2, 2)
        plt.imshow(non_transparent_mask, cmap='gray')
        plt.title(f"Non-Transparent Mask for Image {image_idx} (Folder {folder_idx})")
        plt.show()

        return rgb_yellowness_value, yellowness_index_value, lab_b_yellowness_value, hue_yellow_mean

    # Process each image in both folders (Control and Hypo)
    for idx in range(num_images1):
        print(f"Processing Control image {idx}")
        results = process_image(images_with_overlay[0][idx], 0, idx)
        if results:
            rgb_yellowness[0].append(results[0])
            yellowness_index_cie[0].append(results[1])
            lab_b_yellowness[0].append(results[2])
            hsv_yellowness[0].append(results[3])

    for idx in range(num_images2):
        print(f"Processing Hypo image {idx}")
        results = process_image(images_with_overlay[1][idx], 1, idx)
        if results:
            rgb_yellowness[1].append(results[0])
            yellowness_index_cie[1].append(results[1])
            lab_b_yellowness[1].append(results[2])
            hsv_yellowness[1].append(results[3])

    return rgb_yellowness, yellowness_index_cie, lab_b_yellowness, hsv_yellowness

def quantify_yellowness_masked(binarized_images, color_images):
    num_images = len(binarized_images)

    # Arrays to store results
    rgb_yellowness = []
    yellowness_index_cie = []
    lab_b_yellowness = []
    hsv_yellowness = []

    # Function to compute yellowness in different ways
    def process_image(binary_image, color_image, image_idx):
        # Ensure the color image data is in uint8 format
        image_uint8 = (color_image * 255).astype(np.uint8) if color_image.dtype != np.uint8 else color_image
        
        # Convert the numpy array to a PIL Image and handle transparency
        pil_image = Image.fromarray(image_uint8).convert("RGBA")
        data = np.array(pil_image)

        # Separate the RGB and alpha channels
        rgb_image = data[..., :3]  # RGB part
        alpha_channel = data[..., 3]  # Alpha (transparency) part

        # Ensure the binary image is 2D (height, width)
        if binary_image.ndim == 3:
            binary_image = binary_image[..., 0]  # If it's a 3D image, reduce it to 2D

        # Use the binarized image as a mask, where white (True) pixels are the regions of interest
        mask = binary_image > 0
        non_transparent_mask = alpha_channel > 0
        masked_region = mask & non_transparent_mask  # Mask with both binarized image and non-transparent region

        if not np.any(masked_region):
            return None, None, None, None

        # Apply the mask to the RGB image
        masked_rgb_image = np.zeros_like(rgb_image)
        masked_rgb_image[masked_region] = rgb_image[masked_region]

        rgb_image_non_transparent = masked_rgb_image[masked_region] / 255.0  # Normalize RGB values
        rgb_image_not_norm = masked_rgb_image[masked_region] / 1.0  # NOT Normalized RGB values

        # 1) RGB Yellowness (Y = (R + G) / 2 - B)
        # RGB normalized
        #R = rgb_image_non_transparent[:, 0]
        #G = rgb_image_non_transparent[:, 1]
        #B = rgb_image_non_transparent[:, 2]
        # RGB not normalized
        R = rgb_image_not_norm[:, 0]
        G = rgb_image_not_norm[:, 1]
        B = rgb_image_not_norm[:, 2]
        rgb_yellowness_value = np.mean((R + G) / 2 - B)

        # 2) Yellowness Index (YI) for D65 Illuminant in CIE XYZ
        # The below function expects RGB values to be normalized
        xyz_image = rgb2xyz(rgb_image_non_transparent) #RGB normalized
        X = xyz_image[:, 0]
        Y = xyz_image[:, 1]
        Z = xyz_image[:, 2]

        # Handle zero values in Y to avoid division by zero
        valid_Y_mask = Y > 0
        if np.any(valid_Y_mask):
            yellowness_index = 100 * ((1.28 * X[valid_Y_mask]) - (1.06 * Z[valid_Y_mask])) / Y[valid_Y_mask]
            yellowness_index_value = np.mean(yellowness_index)
        else:
            yellowness_index_value = 0  # No valid Y values

        # 3) Lab b* channel (blue-yellow axis)
        # The below function expects RGB values to be normalized
        lab_image = rgb2lab(rgb_image_non_transparent) # RGB normalized
        b_channel = lab_image[:, 2]  # b* channel
        lab_b_yellowness_value = np.mean(b_channel)

        # 4) HSV Yellowness (Hue between 45° and 75°)
        # The below function expects RGB values to be normalized
        hsv_image = rgb2hsv(rgb_image_non_transparent) # RGB normalized
        hue = hsv_image[:, 0] * 360  # Hue in degrees
        saturation = hsv_image[:, 1]
        value = hsv_image[:, 2]

        # Filter for yellow hue (45° to 75°)
        yellow_mask = (hue >= 45) & (hue <= 75)
        if np.any(yellow_mask):
            hue_yellow_mean = np.mean(saturation[yellow_mask] * value[yellow_mask])
        else:
            hue_yellow_mean = 0  # No yellow found

        # Debugging: Plot the mask and the non-transparent image part
        plt.figure(figsize=(12, 4))

        # Plot the RGB image with the applied mask
        plt.subplot(1, 2, 1)
        plt.imshow(masked_rgb_image)
        plt.title(f"RGB Image with Mask (Image {image_idx})")

        # Plot the binarized mask
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.title(f"Binarized Mask for Image {image_idx}")
        plt.show()

        return rgb_yellowness_value, yellowness_index_value, lab_b_yellowness_value, hue_yellow_mean

    # Process each image
    for idx in range(num_images):
        print(f"Processing Image {idx}")
        results = process_image(binarized_images[idx], color_images[idx], idx)
        if results:
            rgb_yellowness.append(results[0])
            yellowness_index_cie.append(results[1])
            lab_b_yellowness.append(results[2])
            hsv_yellowness.append(results[3])

    return rgb_yellowness, yellowness_index_cie, lab_b_yellowness, hsv_yellowness


def extract_array_for_common_files(stats_path, variable_name, common_filenames):
    """
    Extracts the test values for only the common filenames.
    
    Args:
    - stats_path: The path to the analysis file.
    - variable_name: The variable name to extract (e.g., 'Areas:', 'Perimeters:', etc.).
    - common_filenames: A set of filenames that are common between Day 2 and Day 3.
    
    Returns:
    - np.array: An array of test values corresponding to the common filenames.
    """
    
    # Read the analysis file content
    file_content = read_file(stats_path)
    
    # Extract the list of filenames from the file
    all_filenames = extract_list(file_content, 'Filenames:')
    
    # Extract the corresponding test values for the variable
    all_values = extract_array(stats_path, variable_name)
    
    # Initialize an empty list to hold the values for common filenames
    filtered_values = []
    
    # Loop through the filenames and add the corresponding values if the filename is in the common list
    for filename, value in zip(all_filenames, all_values):
        if filename in common_filenames:
            filtered_values.append(value)
    
    # Return the filtered values as a numpy array
    return np.array(filtered_values)
    


def process_arrays(concatenated, stickcolors, positivecontrols):
    """
    This function processes arrays based on the value of positivecontrols.
    
    Parameters:
    - concatenated: List of arrays to be processed
    - stickcolors: List of colors to be adjusted
    - positivecontrols: Integer indicating how many of the last even and odd arrays to skip
    
    Returns:
    - even_arrays: Concatenated even-numbered arrays (excluding the positive controls)
    - odd_arrays: Concatenated odd-numbered arrays (excluding the positive controls)
    - stickcolors: Adjusted stickcolors for even and odd arrays (excluding the positive controls)
    - even_arrays_pc: Concatenated even-numbered arrays for positive controls
    - odd_arrays_pc: Concatenated odd-numbered arrays for positive controls
    - stickcolors_pc: Stickcolors for the positive control arrays
    """

    # Ensure that positivecontrols is not larger than the number of even/odd arrays available
    max_len = len(concatenated) // 2  # Maximum number of even or odd arrays
    if positivecontrols > max_len:
        raise ValueError(f"positivecontrols cannot be greater than {max_len} based on array length.")

    # Concatenate arrays, skipping the last 'positivecontrols' arrays for both even and odd
    even_arrays = np.concatenate([concatenated[i] for i in range(0, len(concatenated) - 2 * positivecontrols, 2)])
    odd_arrays = np.concatenate([concatenated[i] for i in range(1, len(concatenated) - 2 * positivecontrols, 2)])
    
    # Arrays for positive controls (the last positivecontrols arrays)
    even_arrays_pc = np.concatenate([concatenated[i] for i in range(len(concatenated) - 2 * positivecontrols, len(concatenated), 2)])
    odd_arrays_pc = np.concatenate([concatenated[i] for i in range(len(concatenated) - 2 * positivecontrols + 1, len(concatenated), 2)])

    # Flatten stickcolors
    stickcolors[0] = [my_color for sublist in stickcolors[0] for my_color in sublist]
    stickcolors[1] = [my_color for sublist in stickcolors[1] for my_color in sublist]

    # Adjust stickcolors to match the number of concatenated arrays (excluding positive controls)
    stickcolors[0], stickcolors_pc_0 = stickcolors[0][:len(even_arrays)], stickcolors[0][len(even_arrays):]
    stickcolors[1], stickcolors_pc_1 = stickcolors[1][:len(odd_arrays)], stickcolors[1][len(odd_arrays):]

    stickcolors_pc = [stickcolors_pc_0, stickcolors_pc_1]

    # Ensure that the length of stickcolors matches the concatenated arrays
    if len(stickcolors[0]) != len(even_arrays):
        raise ValueError("Mismatch between the number of colors in stickcolors[0] and the data in even_arrays")
    if len(stickcolors[1]) != len(odd_arrays):
        raise ValueError("Mismatch between the number of colors in stickcolors[1] and the data in odd_arrays")

    return even_arrays, odd_arrays, stickcolors, even_arrays_pc, odd_arrays_pc, stickcolors_pc

def plot_avg_violin(fig2, axs2, even_arrays, odd_arrays, stick_colors, labels, miny_avg, maxy_avg, my_yticks, font_title, font_text, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, hor_lines=None, hor_labels=None, even_arrays_pc=None, odd_arrays_pc=None, stick_colors_pc=None, PositiveControls=0):
    
    ######### Experiment violins
    parts2 = sns.violinplot(data=[even_arrays, odd_arrays], color='grey', inner='stick', linewidth=0.5, fill=True, ax=axs2)
    plt.setp(parts2.collections, alpha=0.2)
    
    # Get LineCollections for experiment violins only (first two violins)
    experiment_lines = [child for child in axs2.get_children() if isinstance(child, LineCollection)][:2]
    
    # Adjust stick colors for experiment violins
    for idx, color_list in enumerate(stick_colors):
        segments = experiment_lines[idx].get_segments()
        for segment, my_color in zip(segments, color_list):
            segment_line = LineCollection([segment], colors=[my_color], linewidths=0.5)
            axs2.add_collection(segment_line)
    
    ######### Positive Control violins (stickcolors_pc handling)
    if even_arrays_pc is not None:
        # Create an x-variable for placing the positive control violins at x positions 2 and 3
        x_pc = [2] * len(even_arrays_pc) + [3] * len(odd_arrays_pc)
        data_pc = np.concatenate([even_arrays_pc, odd_arrays_pc])
        hue_pc = ['even'] * len(even_arrays_pc) + ['odd'] * len(odd_arrays_pc)
        
        # Plot the positive control violins at positions 2 and 3
        parts3 = sns.violinplot(x=x_pc, y=data_pc, hue=hue_pc, color='grey', inner='stick', linewidth=0.5, fill=True, ax=axs2)
        plt.setp(parts3.collections, alpha=0.2)
        
        # Get the LineCollections only for positive control violins (last two violins)
        pc_lines = [child for child in axs2.get_children() if isinstance(child, LineCollection)][-2:]
        
        # Adjust stick colors for positive control violins
        for idx, color_list in enumerate(stick_colors_pc):
            segments = pc_lines[idx].get_segments()
            for segment, my_color in zip(segments, color_list):
                segment_line = LineCollection([segment], colors=[my_color], linewidths=0.5)
                axs2.add_collection(segment_line)
    
    # Plot avg, median, std for experiment data (as before)
    avg_C, median_C, std_C = np.mean(even_arrays), np.median(even_arrays), np.std(even_arrays)
    avg_H, median_H, std_H = np.mean(odd_arrays), np.median(odd_arrays), np.std(odd_arrays)
    
    axs2.plot(0, avg_C, "ok", markersize=7, zorder=2)
    axs2.plot(0, median_C, "xk", markersize=9, zorder=2)
    axs2.vlines(0, avg_C - std_C, avg_C + std_C, color='black', lw=2, zorder=3) #lw was 1 I just made it 2 CDA 20240928

    axs2.plot(1, avg_H, "ok", markersize=7, zorder=2)
    axs2.plot(1, median_H, "xk", markersize=9, zorder=2)
    axs2.vlines(1, avg_H - std_H, avg_H + std_H, color='black', lw=2, zorder=3) #lw was 1 I just made it 2 CDA 20240928
    
    newlabels = (f"{labels[0]} \n (n={len(even_arrays)})", 
                  f"{labels[1]} \n (n={len(odd_arrays)})")  

    if even_arrays_pc is not None:
        # Plot avg, median, std for positive control data (as before)
        avg_C_pc, median_C_pc, std_C_pc = np.mean(even_arrays_pc), np.median(even_arrays_pc), np.std(even_arrays_pc)
        avg_H_pc, median_H_pc, std_H_pc = np.mean(odd_arrays_pc), np.median(odd_arrays_pc), np.std(odd_arrays_pc)
        
        axs2.plot(2, avg_C_pc, "ok", markersize=7, zorder=2)
        axs2.plot(2, median_C_pc, "xk", markersize=9, zorder=2)
        axs2.vlines(2, avg_C_pc - std_C_pc, avg_C_pc + std_C_pc, color='black', lw=2, zorder=3) #lw was 1 I just made it 2 CDA 20240928

        axs2.plot(3, avg_H_pc, "ok", markersize=7, zorder=2)
        axs2.plot(3, median_H_pc, "xk", markersize=9, zorder=2)
        axs2.vlines(3, avg_H_pc - std_H_pc, avg_H_pc + std_H_pc, color='black', lw=2, zorder=3) #lw was 1 I just made it 2 CDA 20240928
    
        # Generate positive control labels
        endash = "\u2013"  # En dash
        labels_pc = [f'+1{endash}{PositiveControls}C', f'+1{endash}{PositiveControls}H']
    
        newlabels_pc = (f"{labels_pc[0]} \n (n={len(even_arrays_pc)})", 
                      f"{labels_pc[1]} \n (n={len(odd_arrays_pc)})")

        axs2.set_xticklabels(newlabels + newlabels_pc)
    else:
        axs2.set_xticklabels(newlabels)
    
    # Handle horizontal lines (if any), x/y limits, and ticks as before
    if hor_lines is not None:
        for i, line_pos in enumerate(hor_lines):
            axs2.axhline(y=line_pos, color='red', linestyle='--', linewidth=1)
            if hor_labels is not None and i < len(hor_labels):
                axs2.text(1.05, line_pos, hor_labels[i], color='red', va='bottom', ha='left', fontsize=SMALL_SIZE, transform=axs2.get_yaxis_transform())

    axs2.spines[['right', 'top']].set_visible(False)  
    axs2.set_xlim(-0.5, len(labels + (labels_pc if even_arrays_pc is not None else [])) - 0.5)
    axs2.set_ylim(miny_avg, maxy_avg)
    axs2.set_yticks(my_yticks)

    for my_label in axs2.get_xticklabels() + axs2.get_yticklabels():
        my_label.set_fontproperties(font_text)

    axs2.tick_params(labelsize=MEDIUM_SIZE, bottom=False)
    
    # Draw vertical lines between each pair of Control and Hypo
    if even_arrays_pc is not None:
        axs2.axvline(1.5, color='black', linestyle='--', linewidth=1)
    
    # Add legend for mean, std, and median
    legend_elements = [
    Line2D([0], [0], marker='o', color='black', label='mean ± std', 
           markerfacecolor='black', markersize=7, linestyle='-', lw=1),  # Dot with line
    Line2D([0], [0], marker='x', color='black', label='median', markersize=7, linestyle='')
        ]

    # Add the legend to the plot
    legend_font = FontProperties(fname=font_text.get_file(), size=MEDIUM_SIZE)
    
    # Assuming legend_elements is already a list of handles
    legend = axs2.legend(
        handles=legend_elements,  # Pass the list directly without wrapping it in another list
        loc='upper right',
        bbox_to_anchor=(1, 1.05),  # Adjust these values as needed
        frameon=True,             # Turn off the frame around the legend
        prop=legend_font,          # Adjust font size as needed
        handletextpad=0.5,         # Spacing between the legend symbol and text
        labelspacing=0.5           # Spacing between rows of the legend
    )
    
    # # Customize the frame (facecolor sets the background color)
    legend.get_frame().set_facecolor('white')  # Set background color to white
    legend.get_frame().set_edgecolor('none')   # Remove the edge of the frame (border)
    legend.get_frame().set_alpha(1.0)     
    
    
    
    
    # legend = axs2.legend(handles=legend_elements, loc='lower center', prop=legend_font, frameon=True)
    
    # # Customize the frame (facecolor sets the background color)
    # legend.get_frame().set_facecolor('white')  # Set background color to white
    # legend.get_frame().set_edgecolor('none')   # Remove the edge of the frame (border)
    # legend.get_frame().set_alpha(1.0)     
    
    ### ### ### ### ### ###
    plt.subplots_adjust(bottom=0, wspace=1.75, top=0.8)
    fig2.tight_layout()
    
def process_images_with_black_percentage(binarized_images, color_images):
    num_images = len(binarized_images)
    black_pixel_percentages = []
    contour_areas = []
    my_roundness_ratio = 0.0

    def find_round_contours(binary_image, image_idx):
        # Find contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        minpixeldiameter = 30
        maxpixeldiameter = 100
        for contour in contours:
            area = cv2.contourArea(contour)
            # Debugging: Print the contour areas
            #print(f"Image {image_idx}: Contour Area = {area}")

            if area < np.pi * (minpixeldiameter / 2) ** 2 or area > np.pi * (maxpixeldiameter / 2) ** 2:
                continue  # Skip contours not within the size range
            
            # Fit an enclosing circle and check roundness
            (x, y), radius = cv2.minEnclosingCircle(contour)
            circle_area = np.pi * (radius ** 2)
            roundness_ratio = area / circle_area

            # Debugging: Print roundness values
            #print(f"Image {image_idx}: Roundness Ratio = {roundness_ratio}, Area = {area}")

            if roundness_ratio > my_roundness_ratio:  # Adjust roundness threshold if needed
                return area, contour  # Return area of the round contour and the contour itself
        
        return np.nan, None  # Return NaN and None if no round contour is found

    for idx in range(num_images):
        print(f"Processing Image {idx}")
        bin_image = binarized_images[idx]
        color_image = color_images[idx]

        # Ensure color image is uint8 for processing
        image_uint8 = (color_image * 255).astype(np.uint8) if color_image.dtype != np.uint8 else color_image
        pil_image = Image.fromarray(image_uint8).convert("RGBA")
        data = np.array(pil_image)

        # Separate RGB and alpha channels
        alpha_channel = data[..., 3]
        non_transparent_mask = alpha_channel > 0

        # Ensure the binary image is 2D
        if bin_image.ndim == 3:
            bin_image = bin_image[..., 0]  # Take only the first channel if it's 3D

        # Binary image mask (black pixels = 0)
        black_pixel_mask = (bin_image == 0)

        # Ensure that both masks are 2D and apply them
        if non_transparent_mask.ndim == 3:
            non_transparent_mask = non_transparent_mask[..., 0]  # Make non-transparent mask 2D if needed

        # Calculate non-transparent and black pixel count
        non_transparent_pixels = np.count_nonzero(non_transparent_mask)
        black_pixels = np.count_nonzero(black_pixel_mask & non_transparent_mask)
        black_pixel_percentage = 100.0 * black_pixels / non_transparent_pixels if non_transparent_pixels > 0 else 0

        black_pixel_percentages.append(black_pixel_percentage)

        # Initialize an overlay to color the areas of the contours
        result_overlay = np.zeros_like(data[..., :3])  # RGB image for result (all black initially)
        white_pixels = np.zeros_like(bin_image)
        white_pixels[bin_image > 0] = 255  # White pixels
        result_overlay[..., 0] = white_pixels  # Red channel for white pixels (same as before)

        # Look for round contours
        area, round_contour = find_round_contours(bin_image, idx)
        contour_areas.append(area)

        if round_contour is not None:
            # Create a mask for the round contour and paint it green in the result overlay
            round_mask = np.zeros_like(bin_image)
            cv2.drawContours(round_mask, [round_contour], -1, 255, thickness=cv2.FILLED)
            result_overlay[round_mask > 0] = [0, 255, 0]  # Paint the round area green

        # Display the result
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(bin_image, cmap='gray')
        plt.title(f"Binarized Image {idx}")

        plt.subplot(1, 3, 2)
        plt.imshow(black_pixel_mask & non_transparent_mask, cmap='gray')
        plt.title(f"Black Pixels (Image {idx})")

        plt.subplot(1, 3, 3)
        plt.imshow(result_overlay)
        plt.title(f"Overlay with Colored Contours (Image {idx})")
        plt.show()

    return black_pixel_percentages, contour_areas


def combine_data(control_data_2, hypo_data_2, control_data_3, hypo_data_3, combine_type, concatenated):
    if combine_type == 'Difference':  # Difference
        concatenated.append(control_data_3 - control_data_2)
        concatenated.append(hypo_data_3 - hypo_data_2)
    
    elif combine_type == 'Ratio':  # Ratio
        concatenated.append(control_data_3 / control_data_2)
        concatenated.append(hypo_data_3 / hypo_data_2)
    
    elif combine_type == 'Percentage_change':  # Percentage Change
        concatenated.append((control_data_3 - control_data_2) / control_data_2 * 100)
        concatenated.append((hypo_data_3 - hypo_data_2) / hypo_data_2 * 100)
    
    elif combine_type == 'Harmonic_mean':  # Harmonic Mean Comparison
        control_harmonic_mean = (2 * control_data_2 * control_data_3) / (control_data_2 + control_data_3)
        hypo_harmonic_mean = (2 * hypo_data_2 * hypo_data_3) / (hypo_data_2 + hypo_data_3)
        concatenated.append(control_harmonic_mean)
        concatenated.append(hypo_harmonic_mean)
    
    elif combine_type == 'Midpoint_comparison':  # Midpoint Comparison
        control_midpoint = (control_data_3 - control_data_2) / ((control_data_3 + control_data_2) / 2)
        hypo_midpoint = (hypo_data_3 - hypo_data_2) / ((hypo_data_3 + hypo_data_2) / 2)
        concatenated.append(control_midpoint)
        concatenated.append(hypo_midpoint)
        
    elif combine_type == 'Visibility':  # Midpoint Comparison
        control_midpoint = (control_data_3 - control_data_2) / ((control_data_3 + control_data_2)) * 100
        hypo_midpoint = (hypo_data_3 - hypo_data_2) / ((hypo_data_3 + hypo_data_2)) * 100
        concatenated.append(control_midpoint)
        concatenated.append(hypo_midpoint)
    
    elif combine_type == 'RMS_difference':  # Root Mean Square (RMS) Difference
        control_rms = np.sqrt((control_data_2**2 + control_data_3**2) / 2)
        hypo_rms = np.sqrt((hypo_data_2**2 + hypo_data_3**2) / 2)
        concatenated.append(control_rms)
        concatenated.append(hypo_rms)

    elif combine_type == 'Geometric_mean':  # Geometric Mean
        control_geom_mean = np.sqrt(control_data_2 * control_data_3)
        hypo_geom_mean = np.sqrt(hypo_data_2 * hypo_data_3)
        concatenated.append(control_geom_mean)
        concatenated.append(hypo_geom_mean)
    
    elif combine_type == 'Symmetric_percentage_change':  # Symmetric Percentage Change
        control_symmetric = 2 * (control_data_3 - control_data_2) / (control_data_3 + control_data_2) * 100
        hypo_symmetric = 2 * (hypo_data_3 - hypo_data_2) / (hypo_data_3 + hypo_data_2) * 100
        concatenated.append(control_symmetric)
        concatenated.append(hypo_symmetric)
    
    elif combine_type == 'Relative_difference':  # Relative Difference
        control_relative_diff = np.abs(control_data_3 - control_data_2) / np.maximum(control_data_3, control_data_2)
        hypo_relative_diff = np.abs(hypo_data_3 - hypo_data_2) / np.maximum(hypo_data_3, hypo_data_2)
        concatenated.append(control_relative_diff)
        concatenated.append(hypo_relative_diff)
        
    elif combine_type == 'Relative_difference_nonabs':  # Relative Difference
        control_relative_diff = (control_data_3 - control_data_2) / np.maximum(control_data_3, control_data_2) * 100
        hypo_relative_diff = (hypo_data_3 - hypo_data_2) / np.maximum(hypo_data_3, hypo_data_2) * 100
        concatenated.append(control_relative_diff)
        concatenated.append(hypo_relative_diff)
    
    elif combine_type == 'Cosine_similarity':  # Cosine Similarity
        control_cosine = (control_data_2 * control_data_3) / (np.linalg.norm(control_data_2) * np.linalg.norm(control_data_3))
        hypo_cosine = (hypo_data_2 * hypo_data_3) / (np.linalg.norm(hypo_data_2) * np.linalg.norm(hypo_data_3))
        concatenated.append(control_cosine)
        concatenated.append(hypo_cosine)
    
    elif combine_type == 'Z_score_comparison':  # Z-score Comparison (assuming mean and std are known)
        mu_control = np.mean([control_data_2, control_data_3])
        sigma_control = np.std([control_data_2, control_data_3])
        control_z_score_diff = ((control_data_3 - mu_control) / sigma_control) - ((control_data_2 - mu_control) / sigma_control)

        mu_hypo = np.mean([hypo_data_2, hypo_data_3])
        sigma_hypo = np.std([hypo_data_2, hypo_data_3])
        hypo_z_score_diff = ((hypo_data_3 - mu_hypo) / sigma_hypo) - ((hypo_data_2 - mu_hypo) / sigma_hypo)

        concatenated.append(control_z_score_diff)
        concatenated.append(hypo_z_score_diff)
    
    elif combine_type == 'Max_min_normalized_difference':  # Max-Min Normalized Difference
        epsilon = 1e-8  # A small constant to avoid division by zero

        control_normalized_diff = (control_data_3 - control_data_2) / (np.maximum(control_data_3, control_data_2) - np.minimum(control_data_3, control_data_2) + epsilon)
        hypo_normalized_diff = (hypo_data_3 - hypo_data_2) / (np.maximum(hypo_data_3, hypo_data_2) - np.minimum(hypo_data_3, hypo_data_2) + epsilon)

        concatenated.append(control_normalized_diff)
        concatenated.append(hypo_normalized_diff)
    
    elif combine_type == 'Mahalanobis_distance':  # Mahalanobis Distance
        # Simplified for 1D case, assuming variance is calculated from the two data points
        control_variance = np.var([control_data_2, control_data_3])
        hypo_variance = np.var([hypo_data_2, hypo_data_3])

        if control_variance > 0:
            control_mahalanobis = np.abs(control_data_3 - control_data_2) / np.sqrt(control_variance)
        else:
            control_mahalanobis = 0  # If variance is zero, distance is 0

        if hypo_variance > 0:
            hypo_mahalanobis = np.abs(hypo_data_3 - hypo_data_2) / np.sqrt(hypo_variance)
        else:
            hypo_mahalanobis = 0  # If variance is zero, distance is 0

        concatenated.append(control_mahalanobis)
        concatenated.append(hypo_mahalanobis)
    
    return concatenated

def extract_mean_quantity_skipping(directory_path, string_quantity, stages):
    """
    Extract the values associated with the specified string from text files
    in the given directory. Skip files that do not exist and return the stages
    that were successfully processed.
    
    Args:
    - directory_path: Path to the folder containing the text files.
    - string_quantity: The string label (e.g., "Elongations:") you're searching for in the file.
    - stages: A list of specific stages (integers) to check (e.g., [32, 33, 35, 37]).
    
    Returns:
    - mean_quantity: A list of extracted values for the specified string.
    - successful_stages: A list of stages that were successfully processed.
    """
    mean_quantity = []
    successful_stages = []  # To track the stages with existing files

    for i in stages:  # Loop through the provided list of stages
        filename = f"Stage{i}_analysis.txt"
        file_path = os.path.join(directory_path, filename)
        
        if not os.path.exists(file_path):
            print(f"File {filename} does not exist. Skipping...")
            continue  # Skip this stage if the file doesn't exist
        
        try:
            with open(file_path, 'r') as file:
                file_lines = file.readlines()
                for j, file_line in enumerate(file_lines):
                    if string_quantity in file_line:
                        # Check if the next line contains the value
                        if j + 1 < len(file_lines):
                            next_line = file_lines[j + 1].strip()
                            if next_line:
                                try:
                                    # Remove the brackets and convert to float
                                    value = float(next_line.strip('[]'))
                                    mean_quantity.append(value)
                                    successful_stages.append(i)  # Add this stage to successful ones
                                    # Stop searching after finding the value
                                    break
                                except ValueError:
                                    print(f"Warning: Could not convert value to float in file {filename} for line: {next_line}")
                            else:
                                print(f"Warning: Next line is empty in file {filename} for line: {file_line.strip()}")
                        else:
                            print(f"Warning: No following line for value in file {filename} for line: {file_line.strip()}")
                        break
                else:
                    print(f"Warning: Did not find expected quantity '{string_quantity}' in file {filename}")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    return mean_quantity, successful_stages  # Return both the extracted values and successful stages

def plot_tukey_ARTanova_BM(fig3, ax3, tukey_interaction_ARTANOVA, title, font_title, font_text, 
                        SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, my_str, vmin, vmax, numbatch, ishealthy=False):
    """
    Plots the Tukey ART-ANOVA posthoc test results for three cases:
    1. Control to Control
    2. Hypo to Hypo
    3. Control + Hypo to Hypo + Control (larger matrix)
    
    All comparisons are plotted based on the log10(p-adj) values between groups.
    """
    # Extract the relevant data from the Tukey object
    results = tukey_interaction_ARTANOVA.summary()

    # Filter the results for Control-Control, Hypo-Hypo, and Control-Hypo comparisons
    control_control_pairs = []
    hypo_hypo_pairs = []
    control_hypo_pairs = []

    # Mapping function to convert Tukey labels to plot labels
    def map_labels(group):
        if "Control" in group:
            group = group.replace("Control_Batch", "B") + "C"
        elif "Hypo" in group:
            group = group.replace("Hypo_Batch", "B") + "H"
        
        # Ensure zero padding
        if "B" in group:
            parts = group.split("B")
            if len(parts) == 2 and len(parts[1]) > 0:
                batch_num = parts[1][:-1]
                suffix = parts[1][-1]
                group = f"B{int(batch_num):02d}{suffix}"
        return group

    # Loop over Tukey results to separate into the three categories
    for row in results.data[1:]:
        group1, group2, meandiff, p_adj, lower, upper, reject = row
        group1_mapped = map_labels(group1)
        group2_mapped = map_labels(group2)
        
        if "C" in group1_mapped and "C" in group2_mapped:
            control_control_pairs.append((group1_mapped, group2_mapped, p_adj))
        elif "H" in group1_mapped and "H" in group2_mapped:
            hypo_hypo_pairs.append((group1_mapped, group2_mapped, p_adj))
        elif ("C" in group1_mapped and "H" in group2_mapped) or ("H" in group1_mapped and "C" in group2_mapped):
            control_hypo_pairs.append((group1_mapped, group2_mapped, p_adj))

    # Create matrices with upper diagonal elements
    def create_upper_diagonal_matrix(pairs, labels):
        size = len(labels)
        matrix = np.full((size, size), np.nan)
        label_idx = {label: i for i, label in enumerate(labels)}

        for group1, group2, p_adj in pairs:
            if group1 in label_idx and group2 in label_idx:
                i, j = label_idx[group1], label_idx[group2]
                if i < j:
                    matrix[i, j] = np.log10(p_adj)

        return matrix

    # Create a combined matrix for control + hypo comparisons
    def create_combined_matrix(control_pairs, hypo_pairs, control_hypo_pairs, control_labels, hypo_labels):
        combined_labels = control_labels + hypo_labels
        size = len(combined_labels)
        matrix = np.full((size, size), np.nan)
        label_idx = {label: i for i, label in enumerate(combined_labels)}

        # Fill the upper diagonal for control-control and hypo-hypo comparisons
        for pairs, labels in [(control_pairs, control_labels), (hypo_pairs, hypo_labels)]:
            for group1, group2, p_adj in pairs:
                if group1 in label_idx and group2 in label_idx:
                    i, j = label_idx[group1], label_idx[group2]
                    if i < j:
                        matrix[i, j] = np.log10(p_adj)

        # Fill the off-diagonal for control-hypo comparisons
        for group1, group2, p_adj in control_hypo_pairs:
            if group1 in label_idx and group2 in label_idx:
                i, j = label_idx[group1], label_idx[group2]
                if i != j:
                    matrix[i, j] = np.log10(p_adj)

        return matrix

    # Customize plot labels and title
    hypo_labels = [f"B{i:02}H" for i in range(1, numbatch + 1)]
    control_labels = [f"B{i:02}C" for i in range(1, numbatch + 1)]
    
    # Create the matrices
    control_control_matrix = create_upper_diagonal_matrix(control_control_pairs, control_labels)
    hypo_hypo_matrix = create_upper_diagonal_matrix(hypo_hypo_pairs, hypo_labels)
    combined_matrix = create_combined_matrix(control_control_pairs, hypo_hypo_pairs, control_hypo_pairs, control_labels, hypo_labels)

    # Function to plot a matrix
    def plot_matrix(matrix, title, labels, cmap, vmin, vmax, my_str, my_fig, my_ax, ishypo=False):
        sns.heatmap(
            matrix, annot=True, fmt='.1f', cmap=cmap, cbar_kws={'label': 'log(p-adj)', 'shrink': .5}, 
            square=True, mask=np.isnan(matrix), ax=my_ax, annot_kws={"fontsize": SMALL_SIZE, "fontproperties": font_text},
            vmin=vmin, vmax=vmax
        )
        
        cbar = my_ax.collections[0].colorbar
        cbar.ax.axhline(y=-2, color='black', linewidth=1)
        cbar.set_label('log(p adj.)', rotation=270, labelpad=20, fontsize=MEDIUM_SIZE, fontproperties=font_text)
        cbar.ax.yaxis.set_tick_params(labelsize=MEDIUM_SIZE)
        for mylabel in cbar.ax.get_yticklabels():
            mylabel.set_fontproperties(font_text)
    
        for i in range(matrix.shape[0]):
            for j in range(i + 1, matrix.shape[1]):
                if matrix[i, j] < -2 and not np.isnan(matrix[i, j]):
                    my_ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2))
    
        # Check if the matrix is combined (double the size of labels)
        if len(labels) == numbatch * 2:  # Combined case
            #xticks_labels = labels.copy()  # Combined labels for x-axis
            #yticks_labels = labels.copy()  # Combined labels for y-axis
            
            xticks_labels = [f"B{i}C" for i in range(1, numbatch + 1)] + [f"B{i}H" for i in range(1, numbatch + 1)]
            xticks_labels[7] = '+1C'
            xticks_labels[8] = '+2C'
            xticks_labels[9] = '+3C'
            xticks_labels[-3] = '+1H'
            xticks_labels[-2] = '+2H'
            xticks_labels[-1] = '+3H'
            yticks_labels = xticks_labels.copy()
            yticks_labels[-1] = ''
            xticks_labels[0] = ''
        else:
            if ishypo:
                xticks_labels = [f"B{i}H" for i in range(1, numbatch + 1)]
                yticks_labels = [f"B{i}H" for i in range(1, numbatch + 1)]
            else:
                xticks_labels = [f"B{i}C" for i in range(1, numbatch + 1)]
                yticks_labels = [f"B{i}C" for i in range(1, numbatch + 1)]
    
        # Ensure the labels match the number of ticks
        my_ax.set_xticks(np.arange(len(labels)) + 0.5)
        my_ax.set_yticks(np.arange(len(labels)) + 0.5)
    
        my_ax.set_xticklabels(xticks_labels, fontsize=MEDIUM_SIZE, fontproperties=font_text)
        my_ax.set_yticklabels(yticks_labels, fontsize=MEDIUM_SIZE, fontproperties=font_text, rotation=0)
        my_ax.tick_params(axis='both', which='both', length=0)
        
        legend_font = FontProperties(fname=font_text.get_file(), size=MEDIUM_SIZE)
    
        my_fig.text(0.5, 0.875, my_str, fontsize=SMALL_SIZE, fontproperties=font_text, 
                 bbox=dict(facecolor='white', edgecolor='none'), ha='center')
    
        my_fig.suptitle(title, fontsize=BIGGER_SIZE, fontproperties=font_title, x=0.5, y=0.98, ha='center')
        my_fig.subplots_adjust(left=0.1, right=0.975, top=0.85)


    # Plot Control-to-Control
    my_cmap = 'Greens_r' if ishealthy else 'Blues_r'

    # Plot Control + Hypo to Hypo + Control (combined matrix)
    combined_labels = control_labels + hypo_labels
    plot_matrix(combined_matrix, title, combined_labels, my_cmap, vmin, vmax, my_str, fig3, ax3)
    
def replace_D_with_4(array):
    """
    Replaces all occurrences of 'D' in the given list or numpy array with the number 4.
    
    Parameters:
    array (list or numpy.ndarray): The input list or array containing 'D' and other values.
    
    Returns:
    numpy.ndarray: The modified array with 'D' replaced by 4.
    """
    # Convert list to numpy array if needed
    if isinstance(array, list):
        array = np.array(array)
    
    # Convert array to string type if it's not already
    array = array.astype(str)
    
    # Replace 'D' with '4'
    modified_array = np.where(array == 'D', '4', array)
    
    # Convert the array back to numeric type
    return modified_array.astype(int)

def create_contingency_table(array1, array2, states):
    """
    Creates a contingency table for two arrays.

    Parameters:
    array1 (numpy.ndarray): First array.
    array2 (numpy.ndarray): Second array.
    states (list): List of unique states to include in the table.

    Returns:
    pandas.DataFrame: Contingency table.
    """
    # Initialize contingency table with zeros
    table = pd.DataFrame(np.zeros((len(states), len(states))), index=states, columns=states, dtype=int)

    # Populate contingency table
    for state1, state2 in zip(array1, array2):
        table.loc[state1, state2] += 1

    return table

def perform_chi_squared_test(contingency_table):
    """
    Perform chi-squared test on the contingency table.

    Parameters:
    contingency_table (pandas.DataFrame): The contingency table.

    Returns:
    tuple: Chi-squared statistic, p-value, degrees of freedom, and expected frequencies.
    """
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
    return chi2_stat, p_value, dof, expected

def fisher_exact_test(table):
    """
    Perform Fisher's Exact Test.

    Parameters:
    table (pandas.DataFrame): 2x2 Contingency Table.

    Returns:
    tuple: Odds Ratio, P-Value
    """
    oddsratio, p_val = fisher_exact(table)
    return oddsratio, p_val

def cramers_v(table):
    """
    Calculate Cramér's V statistic.

    Parameters:
    table (pandas.DataFrame): Contingency table.

    Returns:
    float: Cramér's V statistic
    """
    chi2_stat, _, _, _ = chi2_contingency(table, correction=False)
    n = table.sum().sum()
    k = min(table.shape) - 1
    return np.sqrt(chi2_stat / (n * k))

def odds_ratio(table):
    """
    Calculate Odds Ratio for a 2x2 table.

    Parameters:
    table (pandas.DataFrame): 2x2 Contingency Table.

    Returns:
    float: Odds Ratio
    """
    table = Table2x2(table)
    return table.oddsratio

def relative_risk(table):
    """
    Calculate Relative Risk for a 2x2 table.

    Parameters:
    table (pandas.DataFrame): 2x2 Contingency Table.

    Returns:
    float: Relative Risk
    """
    table = Table2x2(table)
    return table.relative_risk

def likelihood_ratio_test(table):
    """
    Perform Likelihood Ratio Test.

    Parameters:
    table (pandas.DataFrame): Contingency table.

    Returns:
    tuple: Likelihood Ratio Statistic, P-Value
    """
    chi2_stat, p_val, _, _ = chi2_contingency(table, lambda_="log-likelihood")
    return chi2_stat, p_val

def kappa_statistic(array1, array2):
    """
    Calculate Kappa Statistic.

    Parameters:
    array1 (numpy.ndarray): First array.
    array2 (numpy.ndarray): Second array.

    Returns:
    float: Kappa Statistic
    """
    return cohen_kappa_score(array1, array2)

def mcnemar_test(table):
    """
    Perform McNemar's Test.

    Parameters:
    table (pandas.DataFrame): 2x2 Contingency Table.

    Returns:
    tuple: McNemar Statistic, P-Value
    """
    table = Table2x2(table)
    return table.mcnemar

def mantel_haenszel_test(table):
    """
    Perform Mantel-Haenszel Chi-Squared Test.

    Parameters:
    table (pandas.DataFrame): Contingency table.

    Returns:
    tuple: Mantel-Haenszel Statistic, P-Value
    """
    chi2_stat, p_val, _, _ = chi2_contingency(table, lambda_="mantel-haenszel")
    return chi2_stat, p_val

def t_test_proportions(array1, array2):
    """
    Perform T-test for Proportions.

    Parameters:
    array1 (numpy.ndarray): First array.
    array2 (numpy.ndarray): Second array.

    Returns:
    tuple: T-statistic, P-Value
    """
    count1 = np.bincount(array1)
    count2 = np.bincount(array2)
    success1 = count1[1:]
    success2 = count2[1:]
    n1 = len(array1)
    n2 = len(array2)
    return proportions_ztest([success1, success2], [n1, n2])

def count_differences_for_pairs(concatenated):
    # Length of concatenated should be even
    if len(concatenated) % 2 != 0:
        raise ValueError("The length of concatenated array should be even.")
    
    # Function to calculate count differences between two arrays
    def count_differences(arr1, arr2):
        diff = [np.sum(arr1 == i) - np.sum(arr2 == i) for i in range(1, 5)]
        return diff

    # Process pairs
    results = []
    for i in range(0, len(concatenated), 2):
        arr1 = concatenated[i]
        arr2 = concatenated[i + 1]
        diff = count_differences(arr1, arr2)
        results.append(diff)

    return results

def plot_array_with_spacing(data):
    # Check input data
    if not isinstance(data, list) or not all(isinstance(sublist, list) for sublist in data):
        raise ValueError("Input should be a 2D list.")
    
    # Flatten the 2D list and create a list of x and y coordinates for plotting
    x_coords = []
    y_coords = []
    
    for i, sublist in enumerate(data):
        x_spacing = 5  # Space between different sub-arrays
        for j, value in enumerate(sublist):
            x_coords.append(j + i * x_spacing)
            y_coords.append(value)
    
    # Plotting
    plt.figure(figsize=(15, 6))
    plt.scatter(x_coords, y_coords, c='blue', marker='o')
    
    # Adding labels
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Scatter Plot of 2D Array with Spacing Between Sub-Arrays')
    
    # Optionally, add grid and adjust limits
    plt.grid(True)
    plt.tight_layout()
    
    plt.show()
    
def plot_array_with_bars(data):
    # Check input data
    if not isinstance(data, list) or not all(isinstance(sublist, list) for sublist in data):
        raise ValueError("Input should be a 2D list.")
    
    # Calculate bar positions and heights
    x_positions = []
    heights = []
    colors = []
    
    for i, sublist in enumerate(data):
        x_spacing = 5  # Space between different sub-arrays
        for j, value in enumerate(sublist):
            x_positions.append(j + i * x_spacing)
            heights.append(value)
            colors.append('blue' if value > 0 else 'red')  # Blue for positive, red for negative
    
    # Plotting
    plt.figure(figsize=(15, 6))
    plt.bar(x_positions, heights, color=colors, width=1.0)  # Width adjusted for clarity
    
    # Adding labels
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Bar Plot of 2D Array with Spacing Between Sub-Arrays')
    
    # Optionally, add grid and adjust limits
    plt.grid(True)
    plt.tight_layout()
    
    plt.show()
    
def plot_violins_bio(fig, axs, concatenated, conditionlabels, stick_colors, xpos, miny, maxy, my_yticks, font_title, font_text, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, hor_lines=None, hor_labels=None, p_array=None):
    # Create a violin plot
    #parts = sns.violinplot(data=concatenated, color='white', inner='stick', linewidth=0.5, fill=True, ax=axs)
    #plt.setp(parts.collections, alpha=0.2)

    # Draw vertical lines between each pair of Control and Hypo
    for i in range(1, len(conditionlabels) - 1, 2): 
        axs.axvline(x=i + 0.5, color='black', linestyle='--', linewidth=1)

    #Extract the stick lines (lines corresponding to the individual data points)
    #stick_lines = [child for child in axs.get_children() if isinstance(child, LineCollection)]

    # Adjust the color of each stick based on the organized stick_colors
    # start_idx = 0
    # for batch_idx, (control_colors, hypo_colors) in enumerate(zip(stick_colors[0], stick_colors[1])):
    #     # Each batch has its own control and hypo colors
    #     batch_colors = [control_colors, hypo_colors]
    #     for condition_idx, condition_colors in enumerate(batch_colors):
    #         segments = stick_lines[start_idx].get_segments()

    #         # Check if the number of segments matches the number of colors
    #         if len(segments) != len(condition_colors):
    #             raise ValueError(f"Mismatch between segments ({len(segments)}) and colors ({len(condition_colors)}) in batch {batch_idx + 1}, condition {condition_idx + 1}")

    #         # Assign colors to each segment
    #         for segment, my_color in zip(segments, condition_colors):
    #             axs.add_collection(LineCollection([segment], colors=[my_color], linewidths=0.5))  # Set the color for each segment

    #         start_idx += 1

    # Plot avg, median, std for each violin
    for my_idx in range(len(xpos)):
        axs.plot(xpos[my_idx], np.mean(concatenated[my_idx]), marker="o", markersize=7, markeredgecolor="k", markerfacecolor="k",zorder=2)
        axs.plot(xpos[my_idx], np.median(concatenated[my_idx]), marker="x", markersize=7, markeredgecolor="k", markerfacecolor="k",zorder=3)
        axs.vlines(xpos[my_idx], np.mean(concatenated[my_idx])-np.std(concatenated[my_idx]), np.mean(concatenated[my_idx])+np.std(concatenated[my_idx]), color='black', lw=1,zorder=1)
       
    # Add legend for mean, std, and median
    legend_elements = [
    Line2D([0], [0], marker='o', color='black', label='mean ± std', 
           markerfacecolor='black', markersize=7, linestyle='-', lw=1),  # Dot with line
    Line2D([0], [0], marker='x', color='black', label='median', markersize=7, linestyle='')
        ]

    # Add the legend to the plot
    legend_font = FontProperties(fname=font_text.get_file(), size=MEDIUM_SIZE)
    # legend = axs.legend(handles=legend_elements, loc='lower center', prop=legend_font, frameon=True)
    
    # Assuming legend_elements is already a list of handles
    legend = axs.legend(
        handles=legend_elements,  # Pass the list directly without wrapping it in another list
        loc='upper right',
        bbox_to_anchor=(1, 1.05),  # Adjust these values as needed
        frameon=True,             # Turn off the frame around the legend
        prop=legend_font,          # Adjust font size as needed
        handletextpad=0.5,         # Spacing between the legend symbol and text
        labelspacing=0.5           # Spacing between rows of the legend
    )
    
    # # Customize the frame (facecolor sets the background color)
    legend.get_frame().set_facecolor('white')  # Set background color to white
    legend.get_frame().set_edgecolor('none')   # Remove the edge of the frame (border)
    legend.get_frame().set_alpha(1.0)     

       
    if hor_lines is not None:
         for i, line_pos in enumerate(hor_lines):
             axs.axhline(y=line_pos, color='red', linestyle='--', linewidth=1)
             # Add text next to the horizontal line
             if hor_labels is not None and i < len(hor_labels):
                 axs.text(xpos[-1] + 0.5, line_pos, hor_labels[i], color='red', va='bottom', ha='left', fontsize=MEDIUM_SIZE,fontproperties=font_text)

    axs.spines[['right', 'top']].set_visible(False)
    axs.set_xlim(-0.5, len(conditionlabels))  # set x axis limits to appropriate values; fixed
    axs.set_ylim(miny, maxy)  # variable limits for each test
    axs.set_yticks(my_yticks)
    axs.set_yticklabels(['healthy', '1 cond.', '2+ conds.', 'dead'])
    axs.axhline(y=1.0, color='green', linestyle='--', linewidth=1)
    axs.axhline(y=2.0, color='orange', linestyle='--', linewidth=1)
    axs.axhline(y=3.0, color='red', linestyle='--', linewidth=1)
    axs.axhline(y=4.0, color='black', linestyle='--', linewidth=1)
    
    # Add custom horizontal lines and text based on p_array
    if p_array is not None:
        # Line between the first and third violins
        y_line1 = maxy # maxy - 0.1 * (maxy - miny)
        axs.hlines(y=y_line1, xmin=0, xmax=2, color='black', linestyle='-', linewidth=1)
        axs.text(1, y_line1 + 0.02 * (maxy - miny), p_array[0], color='black', va='bottom', ha='center', fontsize=MEDIUM_SIZE, fontproperties=font_text)

        # Line between the first and second violins
        y_line2 = maxy - 0.15 * (maxy - miny)
        axs.hlines(y=y_line2, xmin=0, xmax=1, color='black', linestyle='-', linewidth=1)
        axs.text(0.5, y_line2 + 0.02 * (maxy - miny), p_array[1], color='black', va='bottom', ha='center', fontsize=MEDIUM_SIZE, fontproperties=font_text)

        # Line between the third and fourth violins
        y_line3 = maxy - 0.15 * (maxy - miny)
        axs.hlines(y=y_line3, xmin=2, xmax=3, color='black', linestyle='-', linewidth=1)
        axs.text(2.5, y_line3 + 0.02 * (maxy - miny), p_array[2], color='black', va='bottom', ha='center', fontsize=MEDIUM_SIZE, fontproperties=font_text)

    # Apply the font properties to the tick labels
    for my_label in axs.get_xticklabels():
        my_label.set_fontproperties(font_text)
    for my_label in axs.get_yticklabels():
        my_label.set_fontproperties(font_text)
    axs.tick_params(labelsize=MEDIUM_SIZE)
    axs.tick_params(bottom=False)
    # setting ticks for x-axis; fixed
    axs.set_xticks([i + 0.5 for i in range(0, len(conditionlabels), 2)])  # Tick positions between the pairs
    reduced_labels = [label[:-1] for label in conditionlabels[::2]]
    axs.set_xticklabels(reduced_labels) 
    
    # Define arrow and text parameters
    arrowprops = dict(facecolor='black', arrowstyle="->", lw=1)
    # Adjust figure-relative coordinates for arrows
    xposc = 0.025  # Adjust the position for C
    xposh = 0.0725   # Adjust the position for H
    yposup_text = 0.89
    yposup = 0.95*maxy 
    yposdown = 0.85*maxy
    # Add the arrow at x = xposc, pointing down (relative to the figure)
    axs.annotate('', xy=(0, yposdown), xytext=(0, yposup), arrowprops=arrowprops, transform=axs.transAxes) 
    # Add the arrow at x = xposh, pointing down (relative to the figure)
    axs.annotate('', xy=(1, yposdown), xytext=(1, yposup), arrowprops=arrowprops, transform=axs.transAxes)
    # Define the text box properties with a white background and no visible frame
    bbox_props = dict(boxstyle="round,pad=0.25", edgecolor="none", facecolor="white")
    # Add text with a white background (relative to the figure)
    axs.text(xposc, yposup_text + 0.05, 'C', ha='center', fontsize=MEDIUM_SIZE, font=font_text, transform=axs.transAxes, bbox=bbox_props)
    axs.text(xposh, yposup_text + 0.05, 'H', ha='center', fontsize=MEDIUM_SIZE, font=font_text, transform=axs.transAxes, bbox=bbox_props)
    
    
    plt.subplots_adjust(bottom=0, wspace=1.75, top=0.8)
    fig.tight_layout()
    
def plot_avg_violin_bio(fig2, axs2, even_arrays, odd_arrays, stick_colors, labels, miny_avg, maxy_avg, my_yticks, font_title, font_text, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, hor_lines=None, hor_labels=None, even_arrays_pc=None, odd_arrays_pc=None, stick_colors_pc=None, PositiveControls=0):
    
    ######### Experiment violins
    parts2 = sns.violinplot(data=[even_arrays, odd_arrays], color='grey', inner=None, linewidth=0.5, fill=True, ax=axs2)
    plt.setp(parts2.collections, alpha=0.2)
    
    #Get LineCollections for experiment violins only (first two violins)
    # experiment_lines = [child for child in axs2.get_children() if isinstance(child, LineCollection)][:2]
    
    # #Adjust stick colors for experiment violins
    # for idx, color_list in enumerate(stick_colors):
    #     segments = experiment_lines[idx].get_segments()
    #     for segment, my_color in zip(segments, color_list):
    #         segment_line = LineCollection([segment], colors=[my_color], linewidths=0.5)
    #         axs2.add_collection(segment_line)
            
    # # Define the hard cut y-values
    # y_min = 1
    # y_max = 4
    # # Apply alpha adjustment based on y-values
    # axs2 = adjust_violin_alpha(axs2, y_min, y_max)
    
    ######### Positive Control violins (stickcolors_pc handling)
    if even_arrays_pc is not None:
        # Create an x-variable for placing the positive control violins at x positions 2 and 3
        x_pc = [2] * len(even_arrays_pc) + [3] * len(odd_arrays_pc)
        data_pc = np.concatenate([even_arrays_pc, odd_arrays_pc])
        hue_pc = ['even'] * len(even_arrays_pc) + ['odd'] * len(odd_arrays_pc)
        
        #Plot the positive control violins at positions 2 and 3
        parts3 = sns.violinplot(x=x_pc, y=data_pc, hue=hue_pc, color='grey', inner=None, linewidth=0.5, fill=True, ax=axs2)
        plt.setp(parts3.collections, alpha=0.2)
        
        # #Get the LineCollections only for positive control violins (last two violins)
        # pc_lines = [child for child in axs2.get_children() if isinstance(child, LineCollection)][-2:]
        
        # #Adjust stick colors for positive control violins
        # for idx, color_list in enumerate(stick_colors_pc):
        #     segments = pc_lines[idx].get_segments()
        #     for segment, my_color in zip(segments, color_list):
        #         segment_line = LineCollection([segment], colors=[my_color], linewidths=0.5)
        #         axs2.add_collection(segment_line)
    
    # Plot avg, median, std for experiment data (as before)
    avg_C, median_C, std_C = np.mean(even_arrays), np.median(even_arrays), np.std(even_arrays)
    avg_H, median_H, std_H = np.mean(odd_arrays), np.median(odd_arrays), np.std(odd_arrays)
    
    axs2.plot(0, avg_C, "ok", markersize=7, zorder=2)
    axs2.plot(0, median_C, "xk", markersize=7, zorder=3)
    axs2.vlines(0, avg_C - std_C, avg_C + std_C, color='black', lw=1, zorder=1)

    axs2.plot(1, avg_H, "ok", markersize=7, zorder=2)
    axs2.plot(1, median_H, "xk", markersize=7, zorder=3)
    axs2.vlines(1, avg_H - std_H, avg_H + std_H, color='black', lw=1, zorder=1)
    
    newlabels = (f"{labels[0]} \n (n={len(even_arrays)})", 
                  f"{labels[1]} \n (n={len(odd_arrays)})")  

    if even_arrays_pc is not None:
        # Plot avg, median, std for positive control data (as before)
        avg_C_pc, median_C_pc, std_C_pc = np.mean(even_arrays_pc), np.median(even_arrays_pc), np.std(even_arrays_pc)
        avg_H_pc, median_H_pc, std_H_pc = np.mean(odd_arrays_pc), np.median(odd_arrays_pc), np.std(odd_arrays_pc)
        
        axs2.plot(2, avg_C_pc, "ok", markersize=7, zorder=2)
        axs2.plot(2, median_C_pc, "xk", markersize=7, zorder=3)
        axs2.vlines(2, avg_C_pc - std_C_pc, avg_C_pc + std_C_pc, color='black', lw=1, zorder=1)

        axs2.plot(3, avg_H_pc, "ok", markersize=7, zorder=2)
        axs2.plot(3, median_H_pc, "xk", markersize=7, zorder=3)
        axs2.vlines(3, avg_H_pc - std_H_pc, avg_H_pc + std_H_pc, color='black', lw=1, zorder=1)
    
        # Generate positive control labels
        endash = "\u2013"  # En dash
        labels_pc = [f'+1{endash}{PositiveControls}C', f'+1{endash}{PositiveControls}H']
    
        newlabels_pc = (f"{labels_pc[0]} \n (n={len(even_arrays_pc)})", 
                      f"{labels_pc[1]} \n (n={len(odd_arrays_pc)})")

        axs2.set_xticklabels(newlabels + newlabels_pc)
    else:
        axs2.set_xticklabels(newlabels)
        
    axs2.axhline(y=1.0, color='green', linestyle='--', linewidth=1)
    axs2.axhline(y=2.0, color='orange', linestyle='--', linewidth=1)
    axs2.axhline(y=3.0, color='red', linestyle='--', linewidth=1)
    axs2.axhline(y=4.0, color='black', linestyle='--', linewidth=1)
    
    # Handle horizontal lines (if any), x/y limits, and ticks as before
    if hor_lines is not None:
        for i, line_pos in enumerate(hor_lines):
            axs2.axhline(y=line_pos, color='red', linestyle='--', linewidth=1)
            if hor_labels is not None and i < len(hor_labels):
                axs2.text(1.05, line_pos, hor_labels[i], color='red', va='bottom', ha='left', fontsize=SMALL_SIZE, transform=axs2.get_yaxis_transform())

    axs2.spines[['right', 'top']].set_visible(False)  
    axs2.set_xlim(-0.5, len(labels + (labels_pc if even_arrays_pc is not None else [])) - 0.5)
    axs2.set_ylim(miny_avg, maxy_avg)
    axs2.set_yticks(my_yticks)
    axs2.set_yticklabels(['healthy', '1 cond.', '2+ conds.', 'dead'])

    for my_label in axs2.get_xticklabels() + axs2.get_yticklabels():
        my_label.set_fontproperties(font_text)

    axs2.tick_params(labelsize=MEDIUM_SIZE, bottom=False)
    
    # Draw vertical lines between each pair of Control and Hypo
    if even_arrays_pc is not None:
        axs2.axvline(1.5, color='black', linestyle='--', linewidth=1)
    
    # Add legend for mean, std, and median
    legend_elements = [
    Line2D([0], [0], marker='o', color='black', label='mean ± std', 
           markerfacecolor='black', markersize=7, linestyle='-', lw=1),  # Dot with line
    Line2D([0], [0], marker='x', color='black', label='median', markersize=7, linestyle='')
        ]

    # Add the legend to the plot
    legend_font = FontProperties(fname=font_text.get_file(), size=MEDIUM_SIZE)
    
    # Assuming legend_elements is already a list of handles
    legend = axs2.legend(
        handles=legend_elements,  # Pass the list directly without wrapping it in another list
        loc='upper right',
        bbox_to_anchor=(1, 1.05),  # Adjust these values as needed
        frameon=True,             # Turn off the frame around the legend
        prop=legend_font,          # Adjust font size as needed
        handletextpad=0.5,         # Spacing between the legend symbol and text
        labelspacing=0.5           # Spacing between rows of the legend
    )
    
    # # Customize the frame (facecolor sets the background color)
    legend.get_frame().set_facecolor('white')  # Set background color to white
    legend.get_frame().set_edgecolor('none')   # Remove the edge of the frame (border)
    legend.get_frame().set_alpha(1.0)     
    
    
    
    
    # legend = axs2.legend(handles=legend_elements, loc='lower center', prop=legend_font, frameon=True)
    
    # # Customize the frame (facecolor sets the background color)
    # legend.get_frame().set_facecolor('white')  # Set background color to white
    # legend.get_frame().set_edgecolor('none')   # Remove the edge of the frame (border)
    # legend.get_frame().set_alpha(1.0)     
    
    ### ### ### ### ### ###
    plt.subplots_adjust(bottom=0, wspace=1.75, top=0.8)
    fig2.tight_layout()
    
def adjust_violin_alpha(ax, y_min, y_max):
    for collection in ax.collections:
        if isinstance(collection, PathPatch):
            path = collection.get_path()
            vertices = path.vertices
            
            # Create a mask for vertices within the y-range
            mask = (vertices[:, 1] >= y_min) & (vertices[:, 1] <= y_max)
            
            # If any vertex is within the range, we set the alpha to 1
            if np.any(mask):
                collection.set_alpha(1)
            else:
                collection.set_alpha(0)

    # Redraw the plot to update transparency
    ax.figure.canvas.draw()
    return ax

def plot_dunn_test_matrix_delta(fig, ax, dunn_results, title, font_title, font_text, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, my_str, vmin, vmax, ishypo=False, ishealthy=False):
    """
    Plots both the upper half of the diagonal for the Dunn test result matrix with log10 scale and shared colorbar range.
    """
    # Extract the upper triangle of the matrix, excluding the diagonal
    dunn_matrix = np.triu(dunn_results, k=1)

    # Apply log10 transformation to the Dunn matrix
    with np.errstate(divide='ignore'):  # Ignore warnings about log10(0)
        dunn_log = np.log10(dunn_matrix)
        dunn_log[dunn_log == -np.inf] = np.nan  # Replace -inf with NaN for plotting purposes

    if ishealthy:
        my_cmap = 'Greens_r'
    else:
        my_cmap = 'Blues_r'

    # Plot log-transformed p-values with shared vmin and vmax
    #fig, ax = plt.subplots()
    sns.heatmap(
        dunn_log, annot=True, fmt='.1f', cmap=my_cmap, cbar_kws={'label': 'log(Dunn p value)', 'shrink': .5}, 
        square=True, mask=np.isnan(dunn_log), ax=ax, annot_kws={"fontsize": SMALL_SIZE, "fontproperties": font_text},
        vmin=vmin, vmax=vmax  # Set the shared colorbar range
    )
    
    # Set color bar and add a line at log(p value) == -2
    cbar = ax.collections[0].colorbar
    cbar.ax.axhline(y=-2, color='black', linewidth=1)
    cbar.set_label('log(Dunn p value)', rotation=270, labelpad=20, fontsize=MEDIUM_SIZE, fontproperties=font_text)
    cbar.ax.yaxis.set_tick_params(labelsize=MEDIUM_SIZE)
    for mylabel in cbar.ax.get_yticklabels():
        mylabel.set_fontproperties(font_text)

    # Add black frames around squares where the p-value is < 0.01
    for i in range(dunn_matrix.shape[0]):
        for j in range(i + 1, dunn_matrix.shape[1]):
            if dunn_matrix[i, j] < 0.01 and not np.isnan(dunn_matrix[i, j]):
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2))  # Increased linewidth for visibility

    # Ensure the limits include the full heatmap (to avoid clipping of rightmost squares)
    ax.set_xlim(0, dunn_matrix.shape[1])
    ax.set_ylim(dunn_matrix.shape[0], 0)  # Reverse order for proper display
    
    # Customize plot labels and title
    if ishypo:
        xticks_labels = [f"B{i+1}H" for i in range(1, dunn_matrix.shape[0] - 3)] + ['+1H', '+2H', '+3H'] 
        yticks_labels = [f"B{i}H" for i in range(1, dunn_matrix.shape[0] - 2)] + ['+1H', '+2H']
    else:
        xticks_labels = [f"B{i+1}C" for i in range(1, dunn_matrix.shape[0] -3)] + ['+1C', '+2C', '+3C']
        yticks_labels = [f"B{i}C" for i in range(1, dunn_matrix.shape[0] -2)] + ['+1C', '+2C']

    ax.set_xticks(np.arange(len(xticks_labels)) + 1.5)
    ax.set_yticks(np.arange(dunn_matrix.shape[0] - 1) + 0.5)
    
    ax.set_xticklabels(xticks_labels)#, fontsize=MEDIUM_SIZE, fontproperties=font_text)
    ax.set_yticklabels(yticks_labels, rotation=0) #, fontsize=MEDIUM_SIZE, fontproperties=font_text, rotation=0)
    for my_label in ax.get_xticklabels():
        my_label.set_fontproperties(font_text)
    for my_label in ax.get_yticklabels():
        my_label.set_fontproperties(font_text)
    ax.tick_params(labelsize=MEDIUM_SIZE)
    
    ax.tick_params(axis='both', which='both', length=0)  # Remove ticks
    
    # Adding the text box below the colorbar (adjusting position as needed)
    fig.text(0.5, 0.83, my_str, fontsize=MEDIUM_SIZE, fontproperties=font_text, #0.87
             bbox=dict(facecolor='white', edgecolor='none'), ha='center')

    plt.suptitle(title, fontsize=BIGGER_SIZE, fontproperties=font_title, x=0.5, y=0.98, ha='center')
    plt.subplots_adjust(left=0.1, right=0.83, top=0.8) #right was 0.825 originally 
    
def plot_tukey_ARTanova_delta(fig1, ax1, fig2, ax2, tukey_interaction_ARTANOVA, title, font_title, font_text, 
                        SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE, my_str, vmin, vmax, numbatch, ishealthy=False):
    """
    Plots the Tukey ART-ANOVA posthoc test results for three cases:
    1. Control to Control
    2. Hypo to Hypo
    
    All comparisons are plotted based on the log10(p-adj) values between groups.
    """
    # Extract the relevant data from the Tukey object
    results = tukey_interaction_ARTANOVA.summary()

    # Filter the results for Control-Control, Hypo-Hypo, and Control-Hypo comparisons
    control_control_pairs = []
    hypo_hypo_pairs = []
    control_hypo_pairs = []

    # Mapping function to convert Tukey labels to plot labels
    def map_labels(group):
        if "Control" in group:
            group = group.replace("Control_Batch", "B") + "C"
        elif "Hypo" in group:
            group = group.replace("Hypo_Batch", "B") + "H"
        
        # Ensure zero padding
        if "B" in group:
            # Add zero padding to the batch numbers (e.g., B1 -> B01)
            parts = group.split("B")
            if len(parts) == 2 and len(parts[1]) > 0:
                batch_num = parts[1][:-1]  # Extract the batch number (ignores last char which is 'C' or 'H')
                suffix = parts[1][-1]  # 'C' or 'H'
                group = f"B{int(batch_num):02d}{suffix}"  # Pad with zero
        return group

    # Loop over Tukey results to separate into the three categories
    for row in results.data[1:]:  # Skip header
        group1, group2, meandiff, p_adj, lower, upper, reject = row
        group1_mapped = map_labels(group1)
        group2_mapped = map_labels(group2)
        
        # Debugging prints to check label mappings
        #print(f"Processing pair: {group1_mapped} vs {group2_mapped}, p_adj: {p_adj}")
        
        if "C" in group1_mapped and "C" in group2_mapped:
            control_control_pairs.append((group1_mapped, group2_mapped, p_adj))
        elif "H" in group1_mapped and "H" in group2_mapped:
            hypo_hypo_pairs.append((group1_mapped, group2_mapped, p_adj))
        elif ("C" in group1_mapped and "H" in group2_mapped) or ("H" in group1_mapped and "C" in group2_mapped):
            control_hypo_pairs.append((group1_mapped, group2_mapped, p_adj))

    # Convert the extracted pairs into matrices for plotting (log10(p-adj))
    def create_upper_diagonal_matrix(pairs, labels):
        size = len(labels)
        matrix = np.full((size, size), np.nan)  # Initialize with NaNs
        label_idx = {label: i for i, label in enumerate(labels)}  # Map labels to indices

        for group1, group2, p_adj in pairs:
            #print(f"Inserting {group1} vs {group2} at indices: {label_idx.get(group1, 'NA')} vs {label_idx.get(group2, 'NA')}")
            # Check if both group1 and group2 are in label_idx
            if group1 in label_idx and group2 in label_idx:
                i, j = label_idx[group1], label_idx[group2]
                if i < j:  # Only fill upper diagonal
                    matrix[i, j] = np.log10(p_adj)
            else:
                print(f"Warning: One of the groups {group1} or {group2} not found in labels.")

        return matrix

    # Customize plot labels and title -leave padding here
    hypo_labels = [f"B{i:02}H" for i in range(1,numbatch+1)]  # Zero-padded hypo labels
    control_labels = [f"B{i:02}C" for i in range(1, numbatch+1)]  # Zero-padded control labels
    
    # Create the matrices with only upper diagonal elements
    control_control_matrix = create_upper_diagonal_matrix(control_control_pairs, control_labels)
    hypo_hypo_matrix = create_upper_diagonal_matrix(hypo_hypo_pairs, hypo_labels)
    
    # Plot each matrix with formatting similar to the provided function
    def plot_matrix(matrix, title, labels, cmap, vmin, vmax, my_str, my_fig, my_ax, ishypo=False):
        sns.heatmap(
            matrix, annot=True, fmt='.1f', cmap=cmap, cbar_kws={'label': 'log(p-adj)', 'shrink': .5}, 
            square=True, mask=np.isnan(matrix), ax=my_ax, annot_kws={"fontsize": SMALL_SIZE, "fontproperties": font_text},
            vmin=vmin, vmax=vmax
        )
        
        # Set color bar and formatting
        cbar = my_ax.collections[0].colorbar
        cbar.ax.axhline(y=-2, color='black', linewidth=1)
        cbar.set_label('log(p adj.)', rotation=270, labelpad=20, fontsize=MEDIUM_SIZE, fontproperties=font_text)
        cbar.ax.yaxis.set_tick_params(labelsize=MEDIUM_SIZE)
        for mylabel in cbar.ax.get_yticklabels():
            mylabel.set_fontproperties(font_text)

        # Add black frames around squares where p-adj < 0.01
        for i in range(matrix.shape[0]):
            for j in range(i + 1, matrix.shape[1]):  # Only upper diagonal squares
                if matrix[i, j] < -2 and not np.isnan(matrix[i, j]):  # log10(p-adj) < -2
                    my_ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=2))

        # Set labels
        my_ax.set_xticks(np.arange(len(labels)) + 0.5)
        my_ax.set_yticks(np.arange(len(labels)) + 0.5)
        
        
        # Set labels
        # Remove the first element from x-tick labels (B1) and replace with an empty string
        if ishypo:
            xticks_labels = [f"B{i}H" for i in range(1, numbatch+1)] #labels.copy()  # Copy the original labels
            yticks_labels = [f"B{i}H" for i in range(1, numbatch+1)] 
        else:
            xticks_labels = [f"B{i}C" for i in range(1, numbatch+1)] #labels.copy()  # Copy the original labels
            yticks_labels = [f"B{i}C" for i in range(1, numbatch+1)] 
        
        xticks_labels[0] = ''  # Replace the first x-tick with an empty string
        if ishypo:
            xticks_labels[-3] = '+1H' 
            xticks_labels[-2] = '+2H'
            xticks_labels[-1] = '+3H' 
        else:
            xticks_labels[-3] = '+1C'
            xticks_labels[-2] = '+2C'
            xticks_labels[-1] = '+3C'
        # Remove the last element from y-tick labels (PC1C) and replace with an empty string
        yticks_labels[-1] = ''  # Replace the last y-tick with an empty string
        if ishypo:
            yticks_labels[-3] = '+1H'
            yticks_labels[-2] = '+2H'
        else:
            yticks_labels[-3] = '+1C'
            yticks_labels[-2] = '+2C'
        
        my_ax.set_xticklabels(xticks_labels, fontsize=MEDIUM_SIZE, fontproperties=font_text)
        my_ax.set_yticklabels(yticks_labels, fontsize=MEDIUM_SIZE, fontproperties=font_text, rotation=0)
        
        my_ax.tick_params(axis='both', which='both', length=0)  # Remove ticks
        
        # Set color bar and add a line at log(p value) == -2
        cbar = my_ax.collections[0].colorbar
        cbar.ax.axhline(y=-2, color='black', linewidth=1)
        cbar.set_label('log(p value adj.)', rotation=270, labelpad=20, fontsize=MEDIUM_SIZE, fontproperties=font_text)
        cbar.ax.yaxis.set_tick_params(labelsize=MEDIUM_SIZE)
        for mylabel in cbar.ax.get_yticklabels():
            mylabel.set_fontproperties(font_text)
        
        # Add text box and title
        my_fig.text(0.5, 0.755, my_str, fontsize=SMALL_SIZE, fontproperties=font_text, #was 0.79
                 bbox=dict(facecolor='white', edgecolor='none'), ha='center')

        my_fig.suptitle(title, fontsize=BIGGER_SIZE, fontproperties=font_title, x=0.5, y=0.98, ha='center')
        my_fig.subplots_adjust(left=0.1, right=0.83, top=0.73) #was 0.75
        
        #plt.show()

    # Plot Control-to-Control
    if ishealthy:
        my_cmap = 'Greens_r'
    else:
        my_cmap = 'Blues_r'
    
    plot_matrix(control_control_matrix, title, control_labels, my_cmap, vmin, vmax, my_str, fig1, ax1)
    # Plot Hypo-to-Hypo
    plot_matrix(hypo_hypo_matrix, title, hypo_labels, my_cmap, vmin, vmax, my_str, fig2, ax2, ishypo=True)
    
                                                                                    
def plot_D2_vertical_new(fig, num_images1, num_images2, images_with_overlay, bin_images, plot_title, footer_txt, BIGGER_SIZE, MEDIUM_SIZE, font_title, font_text, hull_images=None, alpha_threshold=1e-5):
    # Set the title
    plt.suptitle(plot_title, fontsize=BIGGER_SIZE, fontproperties=font_title)

    # Calculate the number of rows based on the larger of num_images1 and num_images2
    num_rows = max(num_images1, num_images2)

    # Set up GridSpec with 2 columns (Control on the left, Hypo on the right) and dynamically calculated rows
    gs = gridspec.GridSpec(num_rows, 2, width_ratios=[1, 1], wspace=0.2, hspace=0.02)
    
    def shift_image_left_2(image, image2, alpha_threshold=1e-5):
        # Find the first non-transparent pixel
        first_non_transparent = find_first_non_transparent_pixel2(image, alpha_threshold)
        if first_non_transparent == 0:
            return image, image2  # No need to shift if it's already aligned

        # Shift the image data to the left
        shifted_image = np.zeros_like(image)
        shifted_image2 = np.zeros_like(image2)
        width = image.shape[1]
        shifted_image[:, :width - first_non_transparent]   = image[:, first_non_transparent:]
        shifted_image2[:, :width - first_non_transparent] = image2[:, first_non_transparent:]

        return shifted_image, shifted_image2
    
    def process_image_new(binary_image, color_image):
        # Ensure the color image data is in uint8 format
        image_uint8 = (color_image * 255).astype(np.uint8) if color_image.dtype != np.uint8 else color_image
        
        # Convert the numpy array to a PIL Image and handle transparency
        pil_image = Image.fromarray(image_uint8).convert("RGBA")
        data = np.array(pil_image)

        # Separate the RGB and alpha channels
        rgb_image = data[..., :3]  # RGB part
        alpha_channel = data[..., 3]  # Alpha (transparency) part

        # Ensure the binary image is 2D (height, width)
        if binary_image.ndim == 3:
            binary_image = binary_image[..., 0]  # If it's a 3D image, reduce it to 2D

        # Use the binarized image as a mask, where white (True) pixels are the regions of interest
        mask = binary_image > 0
        non_transparent_mask = alpha_channel > 0
        masked_region = mask & non_transparent_mask  # Mask with both binarized image and non-transparent region

       # if not np.any(masked_region):
        #    return None, None, None, None

        # Apply the mask to the RGB image
        # masked_rgb_image = np.zeros_like(rgb_image)
        # masked_rgb_image[masked_region] = rgb_image[masked_region]
        
        # Create a new alpha channel where pixels outside the mask are set to 0 (transparent)
        new_alpha_channel = np.zeros_like(alpha_channel)
        new_alpha_channel[masked_region] = 255  # Set the masked region to fully opaque (alpha = 255)

        # Create the new RGBA image by combining the RGB and new alpha channels
        new_image = np.dstack((rgb_image, new_alpha_channel))  # Stack RGB and new alpha channel

        
        return new_image #masked_rgb_image

    # Plot images from the first folder (Control, left column)
    #thick = 20 #was 10
    for idx in range(num_images1):
        ax = plt.subplot(gs[idx, 0])  # Always plot in the first column (Control)
        shifted_image, bin_mask_left = shift_image_left_2(images_with_overlay[0][idx],bin_images[0][idx])
        images_with_overlay[0][idx] =  process_image_new(bin_mask_left, shifted_image) #shifted_image #add_frame(shifted_image, framecolors[0][idx], thick)  # Add frame

        if hull_images:
            raise ValueError('Hull not yet implemented with shift')
           # shifted_hull_image = shift_image_left(hull_images[0][idx])
           # ax.imshow(shifted_hull_image, cmap='Blues')
           # ax.imshow(images_with_overlay[0][idx], alpha=0.3)
        else:
            ax.imshow(images_with_overlay[0][idx])

        # Apply a colored frame around the image using `framecolors`
        # my_color = framecolors[0][idx]
        # for spine in ax.spines.values():
        #     spine.set_visible(True)
        #     spine.set_edgecolor(my_color)
        #     spine.set_linewidth(8)  # Increased width for visibility
        
        ax.axis('off')

    # Plot images from the second folder (Hypo, right column)
    for idx in range(num_images2):
        ax = plt.subplot(gs[idx, 1])  # Always plot in the second column (Hypo)
        shifted_image2, bin_mask_left2 = shift_image_left_2(images_with_overlay[1][idx],bin_images[1][idx])
        images_with_overlay[1][idx] = process_image_new(bin_mask_left2, shifted_image2) #shifted_image  #add_frame(shifted_image, framecolors[1][idx], thick)  # Add frame

        if hull_images:
            raise ValueError('Hull not yet implemented with shift')
           # shifted_hull_image = shift_image_left(hull_images[1][idx])
          #  ax.imshow(shifted_hull_image, cmap='Blues')
          #  ax.imshow(images_with_overlay[1][idx], alpha=0.3)
        else:
            ax.imshow(images_with_overlay[1][idx])

        # Apply a colored frame around the image using `framecolors`
        # color = framecolors[1][idx]
        # for spine in ax.spines.values():
        #     spine.set_visible(True)
        #     spine.set_edgecolor(color)
        #     spine.set_linewidth(8)  # Increased width for visibility
        
        ax.axis('off')

    # Add a vertical line exactly in the middle between the two columns
    middle_x = 0.5  # Middle of the figure in normalized coordinates
    line_height = 0.85  # Height of the line as a fraction of the figure height
    line_y_start = 0.05
    ax_line = fig.add_axes([middle_x - 0.002, line_y_start, 0.004, line_height], frameon=False)
    ax_line.axvline(x=0.5, color='black', linestyle='-', linewidth=5)
    ax_line.axis('off')

    # Adjust layout
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.01, hspace=0.01)

    # Add a line of text at the bottom of the figure
    fig.text(0.5, 0, footer_txt, ha='center', fontsize=MEDIUM_SIZE, fontproperties=font_text)

    # Force drawing update
    plt.draw()
    
def plot_D2_vertical_nocolor(fig, num_images1, num_images2, images_with_overlay, plot_title, footer_txt, BIGGER_SIZE, MEDIUM_SIZE, font_title, font_text, hull_images=None, alpha_threshold=1e-5):
    # Set the title
    plt.suptitle(plot_title, fontsize=BIGGER_SIZE, fontproperties=font_title)

    # Calculate the number of rows based on the larger of num_images1 and num_images2
    num_rows = max(num_images1, num_images2)

    # Set up GridSpec with 2 columns (Control on the left, Hypo on the right) and dynamically calculated rows
    gs = gridspec.GridSpec(num_rows, 2, width_ratios=[1, 1], wspace=0.2, hspace=0.02)

    # Function to shift the image content to the left
    def shift_image_left(image, alpha_threshold=1e-5):
        # Find the first non-transparent pixel
        first_non_transparent = find_first_non_transparent_pixel2(image, alpha_threshold)
        if first_non_transparent == 0:
            return image  # No need to shift if it's already aligned

        # Shift the image data to the left
        shifted_image = np.zeros_like(image)
        width = image.shape[1]
        shifted_image[:, :width - first_non_transparent] = image[:, first_non_transparent:]

        return shifted_image

    # Plot images from the first folder (Control, left column)
   # thick = 20 #was 10
    for idx in range(num_images1):
        ax = plt.subplot(gs[idx, 0])  # Always plot in the first column (Control)
        shifted_image = shift_image_left(images_with_overlay[0][idx])
        images_with_overlay[0][idx] = shifted_image

        if hull_images:
            shifted_hull_image = shift_image_left(hull_images[0][idx])
            ax.imshow(shifted_hull_image, cmap='Blues')
            ax.imshow(images_with_overlay[0][idx], alpha=0.3)
        else:
            ax.imshow(images_with_overlay[0][idx])

        # # Apply a colored frame around the image using `framecolors`
        # my_color = framecolors[0][idx]
        # for spine in ax.spines.values():
        #     spine.set_visible(True)
        #     spine.set_edgecolor(my_color)
        #     spine.set_linewidth(8)  # Increased width for visibility
        
        ax.axis('off')

    # Plot images from the second folder (Hypo, right column)
    for idx in range(num_images2):
        ax = plt.subplot(gs[idx, 1])  # Always plot in the second column (Hypo)
        shifted_image = shift_image_left(images_with_overlay[1][idx])
        images_with_overlay[1][idx] = shifted_image #add_frame(shifted_image, framecolors[1][idx], thick)  # Add frame

        if hull_images:
            shifted_hull_image = shift_image_left(hull_images[1][idx])
            ax.imshow(shifted_hull_image, cmap='Blues')
            ax.imshow(images_with_overlay[1][idx], alpha=0.3)
        else:
            ax.imshow(images_with_overlay[1][idx])

        # Apply a colored frame around the image using `framecolors`
        # color = framecolors[1][idx]
        # for spine in ax.spines.values():
        #     spine.set_visible(True)
        #     spine.set_edgecolor(color)
        #     spine.set_linewidth(8)  # Increased width for visibility
        
        ax.axis('off')

    # Add a vertical line exactly in the middle between the two columns
    middle_x = 0.5  # Middle of the figure in normalized coordinates
    line_height = 0.85  # Height of the line as a fraction of the figure height
    line_y_start = 0.05
    ax_line = fig.add_axes([middle_x - 0.002, line_y_start, 0.004, line_height], frameon=False)
    ax_line.axvline(x=0.5, color='black', linestyle='-', linewidth=5)
    ax_line.axis('off')

    # Adjust layout
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.01, hspace=0.01)

    # Add a line of text at the bottom of the figure
    fig.text(0.5, 0, footer_txt, ha='center', fontsize=MEDIUM_SIZE, fontproperties=font_text)

    # Force drawing update
    plt.draw()
    
def plot_D3_vertical_old(fig, num_images1, num_images2, images_with_overlay, plot_title, footer_txt, framecolors, BIGGER_SIZE, MEDIUM_SIZE, font_title, font_text, hull_images=None, alpha_threshold=1e-5):
    # Set the title
    plt.suptitle(plot_title, fontsize=BIGGER_SIZE, fontproperties=font_title)

    # Calculate the number of rows based on the larger of num_images1 and num_images2
    num_rows = max(num_images1, num_images2)

    # Set up GridSpec with 2 columns (Control on the left, Hypo on the right) and dynamically calculated rows
    gs = gridspec.GridSpec(num_rows, 2, width_ratios=[1, 1], wspace=0.2, hspace=0.02)

    # Function to shift the image content to the left
    # def shift_image_left(image, alpha_threshold=1e-5):
    #     # Find the first non-transparent pixel
    #     first_non_transparent = find_first_non_transparent_pixel2(image, alpha_threshold)
    #     if first_non_transparent == 0:
    #         return image  # No need to shift if it's already aligned

    #     # Shift the image data to the left
    #     shifted_image = np.zeros_like(image)
    #     width = image.shape[1]
    #     shifted_image[:, :width - first_non_transparent] = image[:, first_non_transparent:]

    #     return shifted_image

    # Plot images from the first folder (Control, left column)
    thick = 20 #was 10
    for idx in range(num_images1):
        ax = plt.subplot(gs[idx, 0])  # Always plot in the first column (Control)
        #shifted_image = images_with_overlay[0][idx] #shift_image_left(images_with_overlay[0][idx])
        #images_with_overlay[0][idx] = shifted_image #add_frame(shifted_image, framecolors[0][idx], thick)  # Add frame

        if hull_images:
          #  shifted_hull_image = shift_image_left(hull_images[0][idx])
          #  ax.imshow(shifted_hull_image, cmap='Blues')
            ax.imshow(images_with_overlay[0][idx], alpha=0.3)
        else:
            ax.imshow(images_with_overlay[0][idx])

        # Apply a colored frame around the image using `framecolors`
        my_color = framecolors[0][idx]
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(my_color)
            spine.set_linewidth(8)  # Increased width for visibility
        
        ax.axis('off')

    # Plot images from the second folder (Hypo, right column)
    for idx in range(num_images2):
        ax = plt.subplot(gs[idx, 1])  # Always plot in the second column (Hypo)
        #shifted_image = images_with_overlay[1][idx] #shift_image_left(images_with_overlay[1][idx])
        #images_with_overlay[1][idx] = shifted_image #add_frame(shifted_image, framecolors[1][idx], thick)  # Add frame

        if hull_images:
         #   shifted_hull_image = shift_image_left(hull_images[1][idx])
         #   ax.imshow(shifted_hull_image, cmap='Blues')
            ax.imshow(images_with_overlay[1][idx], alpha=0.3)
        else:
            ax.imshow(images_with_overlay[1][idx])

        # Apply a colored frame around the image using `framecolors`
        color = framecolors[1][idx]
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(8)  # Increased width for visibility
        
        ax.axis('off')

    # Add a vertical line exactly in the middle between the two columns
    middle_x = 0.5  # Middle of the figure in normalized coordinates
    line_height = 0.85  # Height of the line as a fraction of the figure height
    line_y_start = 0.05
    ax_line = fig.add_axes([middle_x - 0.002, line_y_start, 0.004, line_height], frameon=False)
    ax_line.axvline(x=0.5, color='black', linestyle='-', linewidth=5)
    ax_line.axis('off')

    # Adjust layout
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.01, hspace=0.01)

    # Add a line of text at the bottom of the figure
    fig.text(0.5, 0, footer_txt, ha='center', fontsize=MEDIUM_SIZE, fontproperties=font_text)

    # Force drawing update
    plt.draw()

def plot_D3_vertical(fig, num_images1, num_images2, images_with_overlay, plot_title, footer_txt, framecolors, BIGGER_SIZE, MEDIUM_SIZE, font_title, font_text, hull_images=None, alpha_threshold=1e-5):
    # Set the title
    plt.suptitle(plot_title, fontsize=BIGGER_SIZE, fontproperties=font_title)

    # Calculate the number of rows based on the larger of num_images1 and num_images2
    num_rows = max(num_images1, num_images2)

    # Set up GridSpec with 2 columns (Control on the left, Hypo on the right) and dynamically calculated rows
    gs = gridspec.GridSpec(num_rows, 2, width_ratios=[1, 1], wspace=0.2, hspace=0.02)

    # Function to shift the image content to the left
    def shift_image_left(image, alpha_threshold=1e-5):
        # Find the first non-transparent pixel
        first_non_transparent = find_first_non_transparent_pixel2(image, alpha_threshold)
        if first_non_transparent == 0:
            return image, 0  # No need to shift if it's already aligned, return shift value 0

        # Shift the image data to the left
        shifted_image = np.zeros_like(image)
        width = image.shape[1]
        shift_amount = first_non_transparent  # How many pixels to shift
        shifted_image[:, :width - shift_amount] = image[:, shift_amount:]

        return shifted_image, shift_amount

    # Plot images from the first folder (Control, left column)
    thick = 20  # was 10
    for idx in range(num_images1):
        ax = plt.subplot(gs[idx, 0])  # Always plot in the first column (Control)
        shifted_image, shift_amount = shift_image_left(images_with_overlay[0][idx])
        images_with_overlay[0][idx] = shifted_image  # Update with shifted image

        if hull_images:
            shifted_hull_image = shift_image_left(hull_images[0][idx])[0]
            ax.imshow(shifted_hull_image, cmap='Blues')
            ax.imshow(images_with_overlay[0][idx], alpha=0.3)
        else:
            ax.imshow(images_with_overlay[0][idx])

        # Shift the axis position horizontally based on how much the image was shifted
        pos = ax.get_position()
        new_pos = [pos.x0 - shift_amount / images_with_overlay[0][idx].shape[1], pos.y0, pos.width, pos.height]
        ax.set_position(new_pos)  # Shift the axis by the same amount

        # Apply a colored frame around the image using `framecolors`
        my_color = framecolors[0][idx]
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(my_color)
            spine.set_linewidth(8)  # Increased width for visibility
        
        ax.axis('off')

    # Plot images from the second folder (Hypo, right column)
    for idx in range(num_images2):
        ax = plt.subplot(gs[idx, 1])  # Always plot in the second column (Hypo)
        shifted_image, shift_amount = shift_image_left(images_with_overlay[1][idx])
        images_with_overlay[1][idx] = shifted_image  # Update with shifted image

        if hull_images:
            shifted_hull_image = shift_image_left(hull_images[1][idx])[0]
            ax.imshow(shifted_hull_image, cmap='Blues')
            ax.imshow(images_with_overlay[1][idx], alpha=0.3)
        else:
            ax.imshow(images_with_overlay[1][idx])

        # Shift the axis position horizontally based on how much the image was shifted
        pos = ax.get_position()
        new_pos = [pos.x0 - shift_amount / images_with_overlay[1][idx].shape[1], pos.y0, pos.width, pos.height]
        ax.set_position(new_pos)  # Shift the axis by the same amount

        # Apply a colored frame around the image using `framecolors`
        color = framecolors[1][idx]
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(8)  # Increased width for visibility
        
        ax.axis('off')

    # Add a vertical line exactly in the middle between the two columns
    middle_x = 0.5  # Middle of the figure in normalized coordinates
    line_height = 0.85  # Height of the line as a fraction of the figure height
    line_y_start = 0.05
    ax_line = fig.add_axes([middle_x - 0.002, line_y_start, 0.004, line_height], frameon=False)
    ax_line.axvline(x=0.5, color='black', linestyle='-', linewidth=5)
    ax_line.axis('off')

    # Adjust layout
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.01, hspace=0.01)

    # Add a line of text at the bottom of the figure
    fig.text(0.5, 0, footer_txt, ha='center', fontsize=MEDIUM_SIZE, fontproperties=font_text)

    # Force drawing update
    plt.draw()
    
def process_images_to_plot_old(image_array_list, target_size=None):
    # Process each image in the list
    processed_images = []
    
    for image_array in image_array_list:
        # Convert the image array to RGBA format if not already
        image = Image.fromarray(image_array).convert("RGBA")
        data = np.array(image)

        non_transparent = data[..., 3] > 0
        if not np.any(non_transparent):
            processed_images.append((0, 0, 0, 0, image))  # No non-transparent pixels found
            continue  # Skip to next image

        labeled_image = label(non_transparent)
        regions = regionprops(labeled_image)
        if not regions:
            processed_images.append((np.sum(non_transparent), 0, 0, 0, data))  # No regions found
            continue

        region = max(regions, key=lambda r: r.area)
        centroid = region.centroid  # Get the original centroid
        orientation = region.orientation
        major_length = region.major_axis_length
        minor_length = region.minor_axis_length

        # Rotate image to make the major axis horizontal
        rotation_angle = -np.degrees(orientation) + 90  # Rotate additional 90 degrees to make major axis horizontal
        rotated_image = rotate(data, rotation_angle, resize=True, mode='edge')

        # After rotation, the centroid will be shifted; translate it back
        new_centroid = np.array(rotated_image.shape[:2]) / 2  # Recenter the centroid to the image center

        # Redraw overlays on the rotated image
        rotated_non_transparent = rotated_image[..., 3] > 0
        rotated_labeled_image = label(rotated_non_transparent)
        rotated_regions = regionprops(rotated_labeled_image)      
        if rotated_regions:
            rotated_region = max(rotated_regions, key=lambda r: r.area)

            # Draw the major and minor axes centered on the new centroid
            major_axis_endpoints = calculate_endpoints(new_centroid, 0, rotated_region.major_axis_length)  # Major axis horizontal
            minor_axis_endpoints = calculate_endpoints(new_centroid, np.pi / 2, rotated_region.minor_axis_length)  # Minor axis vertical

            rr, cc = line(*major_axis_endpoints[0], *major_axis_endpoints[1])
            draw_thick_line(rotated_image, rr, cc, [0, 0, 0, 255], 10)  # Black for major axis

            rr, cc = line(*minor_axis_endpoints[0], *minor_axis_endpoints[1])
            draw_thick_line(rotated_image, rr, cc, [0, 0, 0, 255], 10)  # Black for minor axis

        # Append the processed image data
        processed_images.append((np.sum(rotated_non_transparent), 
                                 rotated_region.perimeter, 
                                 rotated_region.minor_axis_length, 
                                 rotated_region.major_axis_length, 
                                 rotated_image))

    return processed_images

def process_images_to_plot(image_array_list, target_size=None):
    # Process each image in the list
    processed_images = []
    
    for image_array in image_array_list:
        # Ensure the image is in uint8 format before processing
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)  # Scale if necessary and convert to uint8

        # Convert the image array to RGBA format if not already
        image = Image.fromarray(image_array).convert("RGBA")
        data = np.array(image)

        non_transparent = data[..., 3] > 0
        if not np.any(non_transparent):
            processed_images.append((0, 0, 0, 0, image))  # No non-transparent pixels found
            continue  # Skip to next image

        labeled_image = label(non_transparent)
        regions = regionprops(labeled_image)
        if not regions:
            processed_images.append((np.sum(non_transparent), 0, 0, 0, data))  # No regions found
            continue

        region = max(regions, key=lambda r: r.area)
        centroid = region.centroid  # Get the original centroid
        orientation = region.orientation
        major_length = region.major_axis_length
        minor_length = region.minor_axis_length

        # Rotate image to make the major axis horizontal
        rotation_angle = -np.degrees(orientation) + 90  # Rotate additional 90 degrees to make major axis horizontal
        rotated_image = rotate(data, rotation_angle, resize=True, mode='edge')

        # After rotation, the centroid will be shifted; translate it back
        new_centroid = np.array(rotated_image.shape[:2]) / 2  # Recenter the centroid to the image center

        # Redraw overlays on the rotated image
        rotated_non_transparent = rotated_image[..., 3] > 0
        rotated_labeled_image = label(rotated_non_transparent)
        rotated_regions = regionprops(rotated_labeled_image)      
        if rotated_regions:
            rotated_region = max(rotated_regions, key=lambda r: r.area)

            # Draw the major and minor axes centered on the new centroid
            major_axis_endpoints = calculate_endpoints(new_centroid, 0, rotated_region.major_axis_length)  # Major axis horizontal
            minor_axis_endpoints = calculate_endpoints(new_centroid, np.pi / 2, rotated_region.minor_axis_length)  # Minor axis vertical

            rr, cc = line(*major_axis_endpoints[0], *major_axis_endpoints[1])
            draw_thick_line(rotated_image, rr, cc, [0, 0, 0, 255], 10)  # Black for major axis

            rr, cc = line(*minor_axis_endpoints[0], *minor_axis_endpoints[1])
            draw_thick_line(rotated_image, rr, cc, [0, 0, 0, 255], 10)  # Black for minor axis

        # Append the processed image data
        processed_images.append((np.sum(rotated_non_transparent), 
                                 rotated_region.perimeter, 
                                 rotated_region.minor_axis_length, 
                                 rotated_region.major_axis_length, 
                                 rotated_image))

    return processed_images

