# Image Noise and Filtering Application

This application allows users to apply different types of noise to an image and then apply various filters to reduce the noise. The application also calculates and displays the Peak Signal-to-Noise Ratio (PSNR) for each filter applied.

## Features

- Load an image
- Apply Gaussian Noise or Salt & Pepper Noise to the image
- Apply various filters to the noisy image:
  - Mean Filter
  - Gaussian Filter
  - Median Filter
  - Min-Max Filter
- Display the PSNR results for each filter
- Identify and display the best filter based on PSNR

## Requirements

- Python 3.x

Install the required libraries using:
```sh
pip install -r requirements.txt
```

## Usage

1. **Load an Image**: Click the "Load Image" button to load an image.
2. **Select Noise Type**: Choose between "Gaussian Noise" and "Salt & Pepper Noise" from the combobox.
3. **Configure Noise Parameters**:
   - For Gaussian Noise, set the sigma value.
   - For Salt & Pepper Noise, set the probability value.
4. **Apply Noise**: Click the "Apply Noise" button to apply the selected noise to the image.
5. **Select Filters**: Choose the filters you want to apply from the available options.
6. **Apply Filters**: Click the "Apply Filters" button to apply the selected filters to the noisy image.
7. **View Results**: The PSNR results for each filter will be displayed, and the best filter will be highlighted.

## Functions

- apply_noise() : Applies the selected noise to the loaded image.

- apply_filters() : Applies the selected filters to the noisy image and 
calculates the PSNR for each filter.

- display_psnr_results(): Displays the PSNR results and identifies the best filter.

- on_noise_type_change(event): Handles the noise type selection event.

- toggle_entry(checkbox_var, entry): Enables or disables an entry based on the checkbox state.



