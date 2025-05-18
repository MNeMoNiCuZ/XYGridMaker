import os
import re
from PIL import Image, ImageDraw, ImageFont
import math
from tqdm import tqdm

# Maximum dimension (adjustable)
MAX_DIMENSION = 65500  # Maximum supported image dimension in pixels

# Input and Output folders
input_folder = "./input/"
output_folder = "./output/"
output_image_name = "xygrid"

# Column label options
enable_column_labels = True  # Set to True to enable column labels
column_label_position = "both"  # Position of column labels: "top", "bottom", or "both"
column_label_height = 100  # Height of the column label area in pixels
column_font_size = 50  # Font size for the column labels
column_font_type = "arial.ttf"  # Font type for the column labels (path to the .ttf file)
column_label_y_offset = 20  # Y offset for column labels (can be positive or negative)
column_label_bg_color = "black"  # Background color of the column labels
column_label_font_color = "white"  # Font color of the column labels
crop_column_label_text = True  # If True, crop the column label text to fit within the row height
repeat_column_labels_every_n_rows = 10  # Repeat the column labels every N rows (0 means no repetition)

# Row label options
enable_row_labels = True  # Set to True to enable row labels
row_label_position = "both"  # Position of row labels: "left" or "right" or "both"
row_label_width = 500 #Width of the row label column
row_font_size = 50  # Font size for the row labels
row_font_type = "arial.ttf"  # Font type for the row labels
row_label_x_alignment = "center"  # Alignment of the row label text: "left", "center", "right"
row_label_y_alignment = "center"  # Alignment of the row label text: "top", "center", "bottom"
row_label_bg_color = "black"  # Background color of the row labels
row_label_font_color = "white"  # Font color of the row labels
row_label_vertical_padding = 50  # Vertical padding inside the row label text area
remove_tags = True  # Removes tags contained inside < and > from the row label

# Row label data source options
row_label_data_source = "positive_prompt"  # Options: "positive_prompt", "single_file", "multiple_files"
row_label_data_file = "./row_label_data.txt"  # File used when row_label_data_source is "single_file". One line in the text-file represents one row of images
row_label_data_folder = "./input"  # Folder used when row_label_data_source is "multiple_files". One .txt-file per input image, matching the input image name to pair up

# Other settings
input_column_images_grouped = False  # If true, it will look for images in sequence instead of interweaved. If false, images are expected to be interweaved
num_columns = 3  # Number of columns
column_titles = ["Column 1", "Column 2", "Column 3"] # Titles for each column
column_order_override = []  # Override column order. Example: [0,3,1,2] would put the last column second. Partial lists are supported.
output_scale_percent = 100  # Percentage to scale images (100% is default)
output_format = "JPG"  # Output format: "PNG", "JPG" or "JPEG"
output_image_path = os.path.join(output_folder, f"{output_image_name}.{output_format.lower()}")

# Map 'JPG' to 'JPEG' for Pillow
if output_format.upper() == "JPG":
    output_format = "JPEG"

# Ensure the column titles match the number of columns
assert len(column_titles) == num_columns, "Number of column titles must match the number of columns."

# Validate repeat_column_labels_every_n_rows
try:
    repeat_column_labels_every_n_rows = int(repeat_column_labels_every_n_rows)
    if repeat_column_labels_every_n_rows <= 0:
        repeat_column_labels_every_n_rows = 0  # Disable repetition if not a positive integer
except (ValueError, TypeError):
    repeat_column_labels_every_n_rows = 0  # Disable repetition if not an integer

# Create the output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Create the input directory if it doesn't exist
if not os.path.exists(input_folder):
    os.makedirs(input_folder)
    print(f"Created input folder at {input_folder}")
    print("Please place your images in this folder and run the script again.")
    exit()

# Function to get EXIF data, using robust parsing from metadatasearch.py
def get_exif_data(image_path):
    try:
        with Image.open(image_path) as img:
            exif_data = img.info
            if not exif_data or 'parameters' not in exif_data:
                return None
            params = exif_data['parameters']
            positive_end = params.find('Negative prompt:')
            if positive_end != -1:
                return params[:positive_end].strip()
            else:
                return params.strip()
    except Exception:
        return None

# Function to remove tags from text
def regex_strip_tags(text):
    return re.sub(r'<[^>]*>', '', text) if text else text

# Function to preprocess metadata
def preprocess_metadata(image_paths):
    print("Preprocessing metadata...")
    metadata_cache = {}
    for path in image_paths:
        metadata_cache[path] = get_exif_data(path)
    print("Metadata preprocessing complete.")
    return metadata_cache

# Function to get row label data based on the selected source
def get_row_label_data(row_idx, metadata_cache, image_paths):
    if row_label_data_source == "positive_prompt":
        image_path = image_paths[0][row_idx] if row_idx < len(image_paths[0]) else None
        if not image_path or image_path not in metadata_cache:
            return None
        return metadata_cache[image_path]
    elif row_label_data_source == "single_file":
        try:
            with open(row_label_data_file, 'r') as f:
                lines = f.readlines()
                if row_idx < len(lines):
                    return lines[row_idx].strip()
        except Exception as e:
            print(f"Error reading row label data from file: {e}")
    elif row_label_data_source == "multiple_files":
        try:
            # Get the first image path for this row
            if row_idx >= len(image_paths[0]):
                return None
            image_path = image_paths[0][row_idx]
            
            # Extract base filename and create matching .txt filename
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            txt_filename = base_name + ".txt"
            txt_path = os.path.join(row_label_data_folder, txt_filename)
            
            # Read the matching .txt file if it exists
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    return f.read().strip()
            return None
        except Exception as e:
            print(f"Error reading row label data from folder: {e}")
    return None

# Function to dynamically adjust font size
def adjust_font_size(draw, text, font_path, max_width, max_height, initial_font_size):
    if max_width <= 0:
        max_width = 200  # Default width if not set
    max_width -= 2 * row_label_vertical_padding  # Apply horizontal padding
    font_size = initial_font_size
    min_font_size = 5  # Minimum font size to prevent excessive reduction
    
    while font_size >= min_font_size:
        font = ImageFont.truetype(font_path, font_size)
        words = text.split() if text else []
        temp_lines = []
        current_line = ''
        
        # Precompute line height once per font size
        line_height = draw.textbbox((0, 0), "A", font=font)[3]
        
        for word in words:
            test_line = current_line + word + ' '
            if draw.textbbox((0, 0), test_line, font=font)[2] <= max_width:
                current_line = test_line
            else:
                if current_line:
                    temp_lines.append(current_line.strip())
                current_line = word + ' '
        if current_line:
            temp_lines.append(current_line.strip())
        
        if len(temp_lines) * line_height <= max_height:
            return font, font_size, temp_lines
        
        font_size -= 1
    
    # Fallback to minimum font size if no fit
    font = ImageFont.truetype(font_path, min_font_size)
    line_height = draw.textbbox((0, 0), "A", font=font)[3]
    lines = []
    current_line = ''
    for word in words:
        test_line = current_line + word + ' '
        if draw.textbbox((0, 0), test_line, font=font)[2] <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line.strip())
            current_line = word + ' '
    if current_line:
        lines.append(current_line.strip())
    
    return font, min_font_size, lines

# Load images from input folder
image_files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# Group images by model
# Apply column order override if specified
if column_order_override != [0]:
    # Create a complete order list by filling in missing indices
    complete_order = []
    used_indices = set()
    
    # Add specified indices first
    for idx in column_order_override:
        if idx < num_columns and idx not in used_indices:
            complete_order.append(idx)
            used_indices.add(idx)
    
    # Add remaining indices in order
    for idx in range(num_columns):
        if idx not in used_indices:
            complete_order.append(idx)
    
    # Reorder column titles
    column_titles = [column_titles[i] for i in complete_order]

images_by_model = [[] for _ in range(num_columns)]
if input_column_images_grouped:
    images_per_column = len(image_files) // num_columns
    for col in range(num_columns):
        start_idx = col * images_per_column
        end_idx = (col + 1) * images_per_column
        images_by_model[col] = image_files[start_idx:end_idx]
    
    # Apply column order to images after grouping
    if column_order_override != [0]:
        images_by_model = [images_by_model[i] for i in complete_order]
else:
    for idx, image_file in enumerate(image_files):
        model_index = idx % num_columns
        images_by_model[model_index].append(image_file)
    
    # Apply column order to images after grouping
    if column_order_override != [0]:
        images_by_model = [images_by_model[i] for i in complete_order]

# Calculate height constraints before processing
print("\n=== Image Grid Height Analysis ===")
print("Note: This analysis assumes all images are the same size as the first image.")

# Get dimensions of the first image
first_image_path = images_by_model[0][0] if images_by_model[0] else None
if not first_image_path:
    raise ValueError("No images found in the input folder.")
with Image.open(first_image_path) as first_image:
    original_image_height = first_image.height
    original_image_width = first_image.width

# Calculate number of rows
num_rows = len(images_by_model[0])
total_images = num_rows * num_columns

# Calculate label heights
label_instances = 0
if enable_column_labels:
    if column_label_position == "top":
        label_instances += 1
    elif column_label_position == "bottom":
        label_instances += 1
    elif column_label_position == "both":
        label_instances += 2
total_label_height = label_instances * column_label_height

# Calculate repeated label heights
repeated_labels = 0  # Initialize to 0
if enable_column_labels and repeat_column_labels_every_n_rows > 0:
    repeated_labels = (num_rows - 1) // repeat_column_labels_every_n_rows
    total_label_height += repeated_labels * column_label_height

# Calculate total height at 100% scale
total_height_100 = total_label_height + (num_rows * original_image_height)

# Calculate how many rows fit within MAX_DIMENSION
available_height = MAX_DIMENSION - total_label_height
rows_that_fit = available_height // original_image_height
images_that_fit = rows_that_fit * num_columns
total_height_fit = total_label_height + (rows_that_fit * original_image_height)

# Calculate scale to fit all images
if total_height_100 > MAX_DIMENSION:
    scale_factor = MAX_DIMENSION / total_height_100
else:
    scale_factor = 1.0

scaled_image_height = int(original_image_height * scale_factor)
scaled_label_height = int(column_label_height * scale_factor)
scaled_font_size = int(column_font_size * scale_factor)
total_scaled_height = int(total_label_height * scale_factor) + (num_rows * scaled_image_height)

# Display information
print("\n--- Input Summary ---")
print(f"Total Images: {total_images}")
print(f"Total Rows: {num_rows}")
print(f"Original Image Size: {original_image_width}x{original_image_height} pixels")
print(f"Column Label Height: {column_label_height} pixels")
print(f"Repeated Labels Every: {repeat_column_labels_every_n_rows} rows")
print(f"Total Label Instances: {label_instances + repeated_labels}")
print(f"Total Label Height: {total_label_height} pixels")

print("\n--- Fit Analysis at 100% Scale ---")
print(f"Maximum Dimension: {MAX_DIMENSION} pixels")
print(f"Total Height Required: {total_height_100} pixels")
print(f"Rows That Fit: {rows_that_fit}")
print(f"Images That Fit: {images_that_fit}")
print(f"Total Height Used: {total_height_fit} pixels")

print("\n--- Scaling to Fit All Images ---")
print(f"Scale Factor: {scale_factor:.2%}")
print(f"Scaled Image Height: {scaled_image_height} pixels")
print(f"Scaled Label Height: {scaled_label_height} pixels")
print(f"Scaled Font Size: {scaled_font_size} pixels")
print(f"Total Scaled Height: {total_scaled_height} pixels")

# Prompt user for scaling decision only if scaling is needed
apply_scale = True
if scale_factor < 1.0:
    print("\n--- Processing Decision ---")
    print(f"Do you want to apply the scale factor of {scale_factor:.2%} to fit all images? (Y/n, default is Yes)")
    user_input = input().strip().lower()
    apply_scale = user_input != 'n'

if apply_scale:
    output_scale_percent = scale_factor * 100
    output_image_path = os.path.join(output_folder, f"{output_image_name}.{output_format.lower()}")
else:
    print(f"Do you want to split the output into multiple images, each fitting {rows_that_fit} rows? (Y/n, default is Yes)")
    user_input = input().strip().lower()
    split_output = user_input != 'n'
    
    if not split_output:
        print("Aborting operation as per user request.")
        exit()

# Open the images and scale them
print("Loading and scaling images...")
scaled_images = []
image_paths = []  # Store original file paths
for model_idx, model_images in enumerate(images_by_model):
    scaled_model_images = []
    model_paths = []
    for img_path in model_images:
        img = Image.open(img_path)
        if output_scale_percent != 100:
            img = img.resize((int(img.width * output_scale_percent / 100), int(img.height * output_scale_percent / 100)), Image.LANCZOS)
        scaled_model_images.append(img)
        model_paths.append(img_path)
    scaled_images.append(scaled_model_images)
    image_paths.append(model_paths)

# Preprocess metadata
metadata_cache = preprocess_metadata(image_files)

# Determine the size of the output image(s)
max_image_width = max(img.width for model_images in scaled_images for img in model_images)
max_image_height = max(img.height for model_images in scaled_images for img in model_images)
column_label_height_scaled = int(column_label_height * output_scale_percent / 100)
column_font_size_scaled = int(column_font_size * output_scale_percent / 100)

def create_output_image(start_row, end_row, output_path, suffix=""):
    # Calculate total height for this chunk
    total_height = 0
    if enable_column_labels and column_label_position in ["top", "both"]:
        total_height += column_label_height_scaled
    
    for row in range(start_row, end_row):
        total_height += max_image_height
        
        # Check if we need a repeated label for this row
        is_last_row = (row == end_row - 1)
        needs_repeated_label = (repeat_column_labels_every_n_rows > 0 and (row + 1) % repeat_column_labels_every_n_rows == 0)
        
        # Add height for repeated labels, but skip if it's the last row and we'd show bottom labels
        if enable_column_labels and column_label_position == "both" and needs_repeated_label and not (is_last_row and column_label_position in ["bottom", "both"]):
            total_height += column_label_height_scaled
    
    if enable_column_labels and column_label_position in ["bottom", "both"]:
        total_height += column_label_height_scaled

    # Calculate total width without row labels first
    total_width = max_image_width * num_columns
    
    # Only add row label width if row labels are enabled
    if enable_row_labels:
        if row_label_position in ["left", "right"]:
            total_width += row_label_width
        elif row_label_position == "both":
            total_width += 2 * row_label_width  # Add width for both sides

    # Create the output image
    output_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(output_image)
    draw.rectangle([0, 0, total_width, total_height], fill=row_label_bg_color)

    # Load fonts
    try:
        column_font = ImageFont.truetype(column_font_type, column_font_size_scaled)
        row_font = ImageFont.truetype(row_font_type, row_font_size)
    except IOError:
        column_font = ImageFont.load_default()
        row_font = ImageFont.load_default()

    # Draw column labels and images
    current_y_offset = 0
    if enable_column_labels and column_label_position in ["top", "both"]:
        for col in range(num_columns):
            # Only add row label width if row labels are enabled
            row_label_offset = row_label_width if enable_row_labels and row_label_position in ["left", "both"] else 0
            col_x_offset = (col * max_image_width) + row_label_offset
            draw.rectangle([col_x_offset, current_y_offset, col_x_offset + max_image_width, current_y_offset + column_label_height_scaled], fill=column_label_bg_color)
            text_y = current_y_offset + (column_label_height_scaled - column_font_size_scaled) // 2 + int(column_label_y_offset * output_scale_percent / 100)
            draw.text((col_x_offset + max_image_width // 2, text_y), column_titles[col], fill=column_label_font_color, font=column_font, anchor="mm")
        current_y_offset += column_label_height_scaled

    print("Processing rows...")
    for row in tqdm(range(start_row, end_row), desc="Rendering rows"):
        if enable_row_labels:
            # Left label
            if row_label_position in ["left", "both"]:
                row_label_x_offset_left = 0
                x_offset = row_label_width
                draw.rectangle([row_label_x_offset_left, current_y_offset, row_label_x_offset_left + row_label_width, current_y_offset + max_image_height], fill=row_label_bg_color)
                row_label_text = get_row_label_data(row, metadata_cache, image_paths)
                if row_label_text and remove_tags:
                    row_label_text = regex_strip_tags(row_label_text)
                if row_label_text:
                    line_height = row_font_size + 5
                    draw_text_with_wrapping(
                        draw, row_label_text, (row_label_x_offset_left, current_y_offset),
                        row_font, row_label_width, line_height,
                        alignment=row_label_x_alignment,
                        vertical_alignment=row_label_y_alignment,
                        fill=row_label_font_color,
                        max_height=max_image_height - (2 * row_label_vertical_padding),
                        vertical_padding=row_label_vertical_padding
                    )
            else:
                x_offset = 0

            # Right label
            if row_label_position in ["right", "both"]:
                row_label_x_offset_right = total_width - row_label_width
                draw.rectangle([row_label_x_offset_right, current_y_offset, row_label_x_offset_right + row_label_width, current_y_offset + max_image_height], fill=row_label_bg_color)
                row_label_text = get_row_label_data(row, metadata_cache, image_paths)
                if row_label_text and remove_tags:
                    row_label_text = regex_strip_tags(row_label_text)
                if row_label_text:
                    line_height = row_font_size + 5
                    draw_text_with_wrapping(
                        draw, row_label_text, (row_label_x_offset_right, current_y_offset),
                        row_font, row_label_width, line_height,
                        alignment=row_label_x_alignment,
                        vertical_alignment=row_label_y_alignment,
                        fill=row_label_font_color,
                        max_height=max_image_height - (2 * row_label_vertical_padding),
                        vertical_padding=row_label_vertical_padding
                    )

        for col, img in enumerate(scaled_images):
            if row < len(img):
                # Only add row label width if row labels are enabled
                row_label_offset = row_label_width if enable_row_labels and row_label_position in ["left", "both"] else 0
                col_x_offset = (col * max_image_width) + row_label_offset
                output_image.paste(img[row], (col_x_offset, current_y_offset))
        
        current_y_offset += max_image_height
        # Draw repeated column labels after certain rows if enabled
        is_last_row = (row == end_row - 1)
        needs_repeated_label = (repeat_column_labels_every_n_rows > 0 and (row + 1) % repeat_column_labels_every_n_rows == 0)
        
        # Draw repeated labels, but skip if we're on the last row and would draw bottom labels
        if enable_column_labels and column_label_position == "both" and needs_repeated_label and not (is_last_row and column_label_position in ["bottom", "both"]):
            for col in range(num_columns):
                # Only add row label width if row labels are enabled
                row_label_offset = row_label_width if enable_row_labels and row_label_position in ["left", "both"] else 0
                col_x_offset = (col * max_image_width) + row_label_offset
                draw.rectangle([col_x_offset, current_y_offset, col_x_offset + max_image_width, current_y_offset + column_label_height_scaled], fill=column_label_bg_color)
                text_y = current_y_offset + (column_label_height_scaled - column_font_size_scaled) // 2 + int(column_label_y_offset * output_scale_percent / 100)
                draw.text((col_x_offset + max_image_width // 2, text_y), column_titles[col], fill=column_label_font_color, font=column_font, anchor="mm")
            current_y_offset += column_label_height_scaled

    # Draw bottom labels if needed
    if enable_column_labels and column_label_position in ["bottom", "both"]:
        for col in range(num_columns):
            # Only add row label width if row labels are enabled
            row_label_offset = row_label_width if enable_row_labels and row_label_position in ["left", "both"] else 0
            col_x_offset = (col * max_image_width) + row_label_offset
            draw.rectangle([col_x_offset, current_y_offset, col_x_offset + max_image_width, current_y_offset + column_label_height_scaled], fill=column_label_bg_color)
            text_y = current_y_offset + (column_label_height_scaled - column_font_size_scaled) // 2 + int(column_label_y_offset * output_scale_percent / 100)
            draw.text((col_x_offset + max_image_width // 2, text_y), column_titles[col], fill=column_label_font_color, font=column_font, anchor="mm")
        current_y_offset += column_label_height_scaled

    # Save the image
    final_output_path = output_path if not suffix else os.path.join(output_folder, f"{output_image_name}_{suffix}.{output_format.lower()}")
    output_image.save(final_output_path, format=output_format.upper())
    print(f"Image saved to {final_output_path}")

# Function to draw text with wrapping and dynamic font scaling
def draw_text_with_wrapping(draw, text, position, font, max_width, line_height, fill="black", alignment="left", vertical_alignment="top", max_height=None, vertical_padding=0):
    if not text:
        return
    # Adjust font size dynamically if needed
    if max_height:
        # Adjust for padding within the text area
        max_height -= 2 * vertical_padding
        font, line_height, lines = adjust_font_size(draw, text, font.path, max_width, max_height, line_height)
    else:
        words = text.split()
        lines = []
        while words:
            line = ''
            while words and draw.textbbox((0, 0), line + words[0], font=font)[2] <= max_width:
                line += (words.pop(0) + ' ')
            lines.append(line.strip())
    
    text_height = len(lines) * line_height
    y = position[1] + vertical_padding  # Add vertical padding to the starting y position
    
    if vertical_alignment == "center":
        y += (max_height - text_height) // 2 if max_height else (max_image_height - text_height) // 2
    elif vertical_alignment == "bottom":
        y += max_height - text_height if max_height else max_image_height - text_height

    for line in lines:
        if alignment == "center":
            x = position[0] + row_label_vertical_padding + (max_width - 2 * row_label_vertical_padding - draw.textbbox((0, 0), line, font=font)[2]) // 2
        elif alignment == "right":
            x = position[0] + max_width - draw.textbbox((0, 0), line, font=font)[2] - row_label_vertical_padding
        else:  # left alignment
            x = position[0] + row_label_vertical_padding
        
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height

# Generate output image(s)
if apply_scale:
    create_output_image(0, num_rows, output_image_path)
else:
    # Split into multiple images
    for chunk in range(0, num_rows, rows_that_fit):
        start_row = chunk
        end_row = min(chunk + rows_that_fit, num_rows)
        create_output_image(start_row, end_row, output_image_path, suffix=str(chunk // rows_that_fit + 1))
