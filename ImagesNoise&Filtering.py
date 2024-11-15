import customtkinter as ctk
import os
from tkinter import filedialog,messagebox
import cv2
import numpy as np
import random
import math
from PIL import Image

# Fonction pour générer du bruit gaussien
def bruit_gaussien(mu, sigma, largeur, hauteur):
    bruit = np.zeros((hauteur, largeur))
    for i in range(hauteur):
        for j in range(largeur):
            u1 = np.random.random()
            u2 = np.random.random()
            z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
            bruit[i][j] = mu + sigma * z0
    
    return bruit

# Fonction pour générer du bruit de poivre et sel
def bruit_poivre_et_sel(image, prob):
    noisy_image = np.copy(image)
    for i in range(noisy_image.shape[0]):
        for j in range(noisy_image.shape[1]):
            rand = random.random()
            if rand < prob:
                noisy_image[i][j] = 0  # Poivre
            elif rand > 1 - prob:
                noisy_image[i][j] = 255  # Sel
    return noisy_image

# Filtre médian
def filtre_median(img, window_size):
    m, n = img.shape
    offset = window_size // 2
    img_new = np.zeros([m, n])

    for i in range(m):
        for j in range(n):
            window = []
            for k in range(-offset, offset + 1):
                for l in range(-offset, offset + 1):
                    if 0 <= i + k < m and 0 <= j + l < n:
                        window.append(img[i + k, j + l])
            window.sort()
            median = window[len(window) // 2]
            img_new[i, j] = median
    return np.clip(img_new, 0, 255).astype(np.uint8)

# Filtre moyen
def filtre_moyen(img, t):
    m, n = img.shape
    mask = np.ones([t, t], dtype=int) / t ** 2
    img_new = np.zeros([m, n])
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            temp = np.sum(img[i - 1:i + 2, j - 1:j + 2] * mask)
            img_new[i, j] = temp
    return img_new

# Filtre gaussien
def filtre_gaussian(image, kernel_size=5, sigma=1.0):

    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    
    # Create the Gaussian kernel
    k = kernel_size // 2
    x, y = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1))
    gaussian_kernel = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    gaussian_kernel /= gaussian_kernel.sum()  # Normalize the kernel
    
    # Check if image is grayscale or RGB
    if len(image.shape) == 2:  # Grayscale image
        filtered_image = np.zeros_like(image, dtype=np.float64)
        padded_image = np.pad(image, k, mode='constant')
    elif len(image.shape) == 3:  # RGB image
        filtered_image = np.zeros_like(image, dtype=np.float64)
        padded_image = np.pad(image, ((k, k), (k, k), (0, 0)), mode='constant')
    else:
        raise ValueError("Unsupported image shape.")
    
    # Convolution
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if len(image.shape) == 2:  # Grayscale
                region = padded_image[i:i + kernel_size, j:j + kernel_size]
                filtered_image[i, j] = np.sum(region * gaussian_kernel)
            else:  # RGB
                for c in range(image.shape[2]):
                    region = padded_image[i:i + kernel_size, j:j + kernel_size, c]
                    filtered_image[i, j, c] = np.sum(region * gaussian_kernel)
    
    # Clip values to valid range
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
    
    return filtered_image

# Fonction de filtre Min-Max
def filtre_min_max(img, window_size):
    m, n = img.shape
    offset = window_size // 2
    img_new = np.zeros([m, n])

    for i in range(m):
        for j in range(n):
            window = []
            for k in range(-offset, offset + 1):
                for l in range(-offset, offset + 1):
                    if 0 <= i + k < m and 0 <= j + l < n:
                        window.append(img[i + k, j + l])

            i_min = min(window)
            i_max = max(window)

            # Application de la règle min-max
            if img[i, j] < i_min:
                img_new[i, j] = i_min
            elif img[i, j] > i_max:
                img_new[i, j] = i_max
            else:
                img_new[i, j] = img[i, j]

    return np.clip(img_new, 0, 255).astype(np.uint8)


# Fonction PSNR
def peack_signal_noise_ration(img_origin, img_bruit):
    img_origin, img_bruit = img_origin.astype(np.float64), img_bruit.astype(np.float64)
    mse = np.mean((img_origin - img_bruit) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * math.log10(255 ** 2 / mse)

# Interface Tkinter
ctk.set_appearance_mode("Dark")

# Function to load the image
def load_image():
    global image_data, image_name
    file_path = filedialog.askopenfilename(filetypes=[("BMP files", "*.bmp")])  # Only allow BMP files
    if file_path:
        image_name = os.path.basename(file_path)  # Extract the image name from the file path

        # Check if the file has a .bmp extension
        if not file_path.lower().endswith('.bmp'):
            print("Selected file is not a BMP image.")
            return

        # Open and process the BMP image
        image = Image.open(file_path).convert("L")
        image_data = np.array(image)
        display_image_in_new_window(image_data,image_name)
        print(f"Original Image Name: {image_name}")
    else:
        messagebox.showerror("No Image","No image selected.")
        
# Display Image in a new CTkToplevel window
def display_image_in_new_window(image_array,name):
    if not isinstance(image_array, np.ndarray):
        print("Error: The image data is not in the correct format (not a NumPy array).")
        return

    try:
        # Convert the NumPy array to a PIL image
        image = Image.fromarray(image_array)

        # Create a new Toplevel window
        new_window = ctk.CTkToplevel(app)
        new_window.title(name)
        new_window.geometry("600x600")  # Adjust size as needed
        

        # Convert the PIL image to CTkImage and store it in a global variable
        global image_tk  # Ensure image_tk is global to retain the reference
        image_tk = ctk.CTkImage(light_image=image, dark_image=image, size=(600, 600))  
        print("Image converted to CTkImage successfully")
        # Create a label in the new window to display the image
        image_label = ctk.CTkLabel(new_window, image=image_tk ,text="")
        image_label.pack(pady=20)

        # Update the window to ensure image loads
        new_window.update()
        
        print("Image displayed successfully in new window")

    except Exception as e:
        print("An error occurred while displaying the image:", e)

# Function to handle the noise application
def apply_noise():
    global noisy_image ,image_name
    if image_data is None:
        messagebox.showerror("Image Error", "No image loaded.")
        return
    else:
        base_name, ext = os.path.splitext(image_name)
        noise_type = noise_type_combobox.get()
        if noise_type == "Gaussian Noise":
            sigma = float(sigma_entry.get()) if sigma_entry.get() else 0
            noise = bruit_gaussien(0, sigma, image_data.shape[1], image_data.shape[0])
            noisy_image = np.clip(image_data + noise, 0, 255).astype(np.uint8)
            new_image_name=f"{base_name}_{noise_type} s={sigma}{ext}"
            cv2.imwrite(new_image_name, noisy_image)
            display_image_in_new_window(noisy_image, new_image_name)
        elif noise_type == "Salt & Pepper Noise":
            prob = float(probability_entry.get()) if probability_entry.get() else 0.05
            noisy_image = bruit_poivre_et_sel(image_data, prob)
            new_image_name=f"{base_name}_{noise_type} p={prob}{ext}"
            cv2.imwrite(new_image_name, noisy_image)
            display_image_in_new_window(noisy_image, new_image_name)
        else:
            print("Invalid noise type selected.")

# Function to handle filter application
def apply_filters():
    global psnr_values ,image_name
    if noisy_image is None:
        messagebox.showerror("Image Error", "No image loaded or noise applied.")
        return
    base_name, ext = os.path.splitext(image_name)
    filters_selected = [filter for filter, var in filter_vars.items() if var.get()]
    if not filters_selected:
        messagebox.showerror("Filter Error", "No filters selected.")
        return
    psnr_values = {}
    for filter_type in filters_selected:
        if filter_type == "Mean Filter":
            filtered_image = filtre_moyen(noisy_image, int(window_size_entry.get()))
            new_image_name=f"{base_name}_{filter_type} ws={window_size_entry.get()}{ext}"
            cv2.imwrite(new_image_name, filtered_image)
            display_image_in_new_window(filtered_image, new_image_name)
            psnr_values["Mean Filter"] = peack_signal_noise_ration(image_data, filtered_image)
        elif filter_type == "Gaussian Filter":
            sigma = float(sigma_entry.get())
            filtered_image = filtre_gaussian(noisy_image,kernel_size=5, sigma=sigma) 
            new_image_name=f"{base_name}_{filter_type} s={sigma}{ext}"
            cv2.imwrite(new_image_name, filtered_image)
            display_image_in_new_window(filtered_image, new_image_name)
            psnr_values["Gaussian Filter"] = peack_signal_noise_ration(image_data, filtered_image)
        elif filter_type == "Median Filter":
            filtered_image = filtre_median(noisy_image, int(window_size_entry.get()))
            new_image_name=f"{base_name}_{filter_type} ws={window_size_entry.get()}{ext}"
            cv2.imwrite(new_image_name, filtered_image)
            display_image_in_new_window(filtered_image, new_image_name)
            psnr_values["Median Filter"] = peack_signal_noise_ration(image_data, filtered_image)
        elif filter_type == "Min-Max Filter":
            filtered_image = filtre_min_max(noisy_image, int(window_size_entry.get()))
            new_image_name=f"{base_name}_{filter_type} ws={window_size_entry.get()}{ext}"
            display_image_in_new_window(filtered_image, new_image_name)
            psnr_values["Min-Max Filter"] = peack_signal_noise_ration(image_data, filtered_image)
        else:
            print(f"Invalid filter: {filter_type}")
    display_psnr_results()

# Function to display PSNR results
def display_psnr_results():
    if not psnr_values:
        return
    best_filter = max(psnr_values, key=psnr_values.get)
    best_psnr = psnr_values[best_filter]
    print(f"Best Filter: {best_filter} with PSNR = {best_psnr:.2f} dB")
    messagebox.showinfo("PSNR Results", f"Best Filter: {best_filter} with PSNR = {best_psnr:.2f} dB")
    psnr_label.configure(text=f"Best Filter: {best_filter} with PSNR = {best_psnr:.2f} dB")

# Noise type combobox event
def on_noise_type_change(event):
    noise_type = noise_type_combobox.get()
    if noise_type == "Gaussian Noise":
        sigma_entry.configure(state="normal")
        probability_entry.configure(state="disabled")
    elif noise_type == "Salt & Pepper Noise":
        sigma_entry.configure(state="disabled")
        probability_entry.configure(state="normal")

# Initialize application
app = ctk.CTk()
app.geometry("800x600")
app.title("Image Noise and Filter Application")

image_data = None
noisy_image = None
psnr_values = {}

# UI Elements
load_image_button = ctk.CTkButton(app, text="Load Image", command=load_image)
load_image_button.pack(pady=10)

# Noise selection
noise_frame = ctk.CTkFrame(app)
noise_frame.pack(pady=10, padx=20, fill="x")
ctk.CTkLabel(noise_frame, text="Select Noise Type:").grid(row=0, column=0, padx=10, pady=5)
noise_type_combobox = ctk.CTkComboBox(noise_frame, values=["Gaussian Noise", "Salt & Pepper Noise"], state="readonly")
noise_type_combobox.grid(row=0, column=1, padx=10, pady=5)
noise_type_combobox.bind("<<ComboboxSelected>>", on_noise_type_change)

# Noise parameters
sigma_label = ctk.CTkLabel(noise_frame, text="Sigma:")
sigma_label.grid(row=1, column=0, padx=10, pady=5)
sigma_entry = ctk.CTkEntry(noise_frame)
sigma_entry.grid(row=1, column=1, padx=10, pady=5)

probability_label = ctk.CTkLabel(noise_frame, text="Probability:")
probability_label.grid(row=2, column=0, padx=10, pady=5)
probability_entry = ctk.CTkEntry(noise_frame)
probability_entry.grid(row=2, column=1, padx=10, pady=5)

apply_noise_button = ctk.CTkButton(app, text="Apply Noise", command=apply_noise)
apply_noise_button.pack(pady=10)

# Filter selection
filter_frame = ctk.CTkFrame(app)
filter_frame.pack(pady=10, padx=20, fill="x")
filter_vars = {
    "Mean Filter": ctk.IntVar(),
    "Gaussian Filter": ctk.IntVar(),
    "Median Filter": ctk.IntVar(),
    "Min-Max Filter": ctk.IntVar()
}
ctk.CTkLabel(filter_frame, text="Select Filters:").grid(row=0, column=0, padx=10, pady=5)
for idx, (filter_name, var) in enumerate(filter_vars.items()):
    ctk.CTkCheckBox(filter_frame, text=filter_name, variable=var).grid(row=0, column=idx + 1, padx=10, pady=5)

# Window size entry
window_size_label = ctk.CTkLabel(filter_frame, text="Window Size:")
window_size_label.grid(row=1, column=0, padx=10, pady=5)
window_size_entry = ctk.CTkEntry(filter_frame)
window_size_entry.grid(row=1, column=1, padx=10, pady=5)

apply_filters_button = ctk.CTkButton(app, text="Apply Filters", command=apply_filters)
apply_filters_button.pack(pady=10)

# PSNR result
psnr_label = ctk.CTkLabel(app, text="PSNR Results: None", font=("Arial", 14))
psnr_label.pack(pady=20)

app.mainloop()