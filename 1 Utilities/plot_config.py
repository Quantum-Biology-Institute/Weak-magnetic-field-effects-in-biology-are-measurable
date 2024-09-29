# plot_config.py

import os
from matplotlib.font_manager import FontProperties
import matplotlib

# Function to clear the font cache
def clear_font_cache():
    cache_dir = matplotlib.get_cachedir()
    cache_file = os.path.join(cache_dir, 'fontList.json')
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print("Font cache cleared.")

# Function to load fonts
def load_fonts():
    # Load title font
    font_title_path = '/Users/clarice/Library/Fonts/ChapMedium.ttf'
    if not os.path.exists(font_title_path):
        raise FileNotFoundError(f"The title font file was not found at {font_title_path}")
    font_title = FontProperties(fname=font_title_path)
    
    # Load text font
    font_text_path = '/Users/clarice/Library/Fonts/apercu-regular.ttf'
    if not os.path.exists(font_text_path):
        raise FileNotFoundError(f"The text font file was not found at {font_text_path}")
    font_text = FontProperties(fname=font_text_path)
    
    return font_title, font_text

# Function to set default sizes
def set_default_sizes():
    SMALL_SIZE = 10
    MEDIUM_SIZE = 11.5  # size of main text
    BIGGER_SIZE = 14 # size of section text
    
    # Apply sizes to matplotlib defaults
    matplotlib.rc('font', size=SMALL_SIZE)          # Controls default text sizes
    matplotlib.rc('axes', titlesize=MEDIUM_SIZE)    # Fontsize of the axes title
    matplotlib.rc('axes', labelsize=MEDIUM_SIZE)    # Fontsize of the x and y labels
    matplotlib.rc('xtick', labelsize=SMALL_SIZE)    # Fontsize of the tick labels
    matplotlib.rc('ytick', labelsize=SMALL_SIZE)    # Fontsize of the tick labels
    matplotlib.rc('legend', fontsize=SMALL_SIZE)    # Legend fontsize
    matplotlib.rc('figure', titlesize=BIGGER_SIZE)  # Fontsize of the figure title
    
    return SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE

# Function to reset configurations to defaults
def reset_configurations():
    matplotlib.rcdefaults()

# Main function to setup all configurations
def setup():
    clear_font_cache()
    font_title, font_text = load_fonts()
    SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = set_default_sizes()
    
    return font_title, font_text, SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE
