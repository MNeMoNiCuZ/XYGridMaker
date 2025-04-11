# XYGridMaker

XYGridMaker is a powerful tool for creating grid-based image comparisons with customizable row and column labels. It's designed to help visualize and compare multiple image sets by organizing them in a structured grid format.

![xygrid](https://github.com/user-attachments/assets/b16c5333-2cdf-4af1-a62f-6852e3c3966c)

*Screenshot: Example of an XYGrid with labeled columns and rows. The labels are repeated every 2nd row*

## Future Improvements

Currently, only prompt metadata is supported, but additional types will be supported in the future, as well as the option of having the data in either row or column.

The tool will be expanded with input arguments and a simple GUI to choose options.

## Features

- Create image comparison grids with customizable layout
- Add column and row labels to clearly identify image sets
- Position labels at top/bottom for columns and left/right for rows
- Flexible data sources for row labels (image metadata, text files)
- Automatic scaling or splitting into multiple outputs to handle large image sets
- Repetition of column headers for better readability in long grids

## Requirements

- Python 3.6 or higher
- Pillow (PIL Fork)
- tqdm (for progress bars)

## Installation

1. Clone or download this repository
2. Run `venv_create.bat` to set up your environment:
   - Choose your Python version when prompted
   - Accept the default virtual environment name (venv) or choose your own
   - Allow pip upgrade when prompted
   - Allow installation of dependencies from requirements.txt

The script will create:
- A virtual environment
- `venv_activate.bat` for activating the environment
- `venv_update.bat` for updating pip

## Usage

1. Place your images in the `input` folder
2. Configure the settings in `xygrid.py` (see Configuration section)
3. Run the script:
   ```
   python xygrid.py
   ```
4. The output grid will be saved in the `output` folder

## Configuration

### Key Settings

#### Column Label Options

```python
enable_column_labels = True  # Enable/disable column labels
column_label_position = "both"  # Position: "top", "bottom", or "both"
column_label_height = 100  # Height of label area in pixels
column_font_size = 50  # Font size for column labels
column_label_y_offset = 20  # Y offset for positioning text
repeat_column_labels_every_n_rows = 3  # Repeat labels every N rows (0 = no repetition)
```

#### Row Label Options

```python
enable_row_labels = True  # Enable/disable row labels
row_label_position = "both"  # Position: "left", "right", or "both"
row_label_width = 400  # Width of row label column in pixels
row_font_size = 50  # Font size for row labels
row_label_x_alignment = "center"  # Text alignment: "left", "center", "right"
row_label_y_alignment = "center"  # Vertical alignment: "top", "center", "bottom"
remove_tags = True  # Remove text between < and > from row labels
```

### Row Label Data Sources

XYGridMaker offers three different sources for row label data:

1. **Positive Prompt** (`row_label_data_source = "positive_prompt"`):  
   Extracts the positive prompt from the image metadata/EXIF data.

2. **Single File** (`row_label_data_source = "single_file"`):  
   Uses a single text file where each line represents the label for a row.
   - Configure with: `row_label_data_file = "./row_label_data.txt"`

3. **Multiple Files** (`row_label_data_source = "multiple_files"`):  
   Uses individual text files that match the filenames of the first image in each row.
   - Configure with: `row_label_data_folder = "./input"`
   - Example: For a row starting with "image1.png", it will look for "image1.txt"

### Other Important Settings

```python
num_columns = 6  # Number of columns in the grid
column_titles = ["Model 1", "Model 2", "Model 3"...]  # Titles for each column
output_scale_percent = 100  # Scale the output image (100% = original size)
output_format = "JPG"  # Output format: "PNG", "JPG", or "JPEG"
```

## Input and Output Organization

- **Input Images**: Place all images in the `input` folder  
  - Images will be arranged in rows and columns based on their filenames
  - First N images become the first row, next N become the second row, etc.
  - N is the number of columns specified in the configuration

- **Output**: The final grid is saved in the `output` folder
  - Default filename: `xygrid.jpg` (or .png depending on configuration)
  - The script may split into multiple files if the grid exceeds size limits

## Advanced Features

### Automatic Scaling

For very large grids, XYGridMaker will automatically calculate if scaling is needed to avoid exceeding image dimension limits. You can:
- Accept the suggested scaling
- Split into multiple images
- Or abort the operation

### Font and Color Customization

```python
column_font_type = "arial.ttf"  # Font for column labels
column_label_bg_color = "black"  # Background color
column_label_font_color = "white"  # Text color

row_font_type = "arial.ttf"  # Font for row labels
row_label_bg_color = "black"  # Background color
row_label_font_color = "white"  # Text color
```

## Tips

- For best results, use images with the same dimensions
- When using "multiple_files" for row labels, ensure text files have the same base name as the images
- Experiment with row and column label positions for the most readable layout
- Use repeat_column_labels_every_n_rows for grids with many rows to improve readability
- For large comparison grids, enable scaling to avoid dimension limits

## Troubleshooting

- If you encounter font issues, make sure the specified font file exists
- If the grid is too large, try reducing the scale percentage or splitting into multiple images
- For memory errors, try reducing the number of images or their size

## License

This tool is free to use and modify for personal and commercial purposes. 
