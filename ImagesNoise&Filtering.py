import customtkinter as ctk
import os
from tkinter import filedialog
import cv2
import numpy as np
import random
import math
from PIL import Image, ImageTk

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
    cv2.imwrite("bruit_poivre_et_sel.png", noisy_image)
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
def filtre_gaussian(image, sigma, kernel_size=5):
    noisy_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image
    return cv2.GaussianBlur(noisy_image, (kernel_size, kernel_size), sigma)

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
app = ctk.CTk()
app.title("Image Noise and Filter Application")
app.geometry("600x600")

# Load Image
def load_image():
    global image_data, image_name
    file_path = filedialog.askopenfilename()
    if file_path:
        image_name = os.path.basename(file_path)  # Extract the image name from the file path
        image = Image.open(file_path).convert("L")
        image_data = np.array(image)
        display_image(image_data)
        print(f"Original Image Name: {image_name}")  # Debugging print
    else:
        print("No file selected")

# Display Image
def display_image(image_array):
    if not isinstance(image_array, np.ndarray):
        print("Error: The image data is not in the correct format (not a NumPy array).")
        return

    # Ensure the array is of type uint8
    image_array = image_array.astype(np.uint8)
    
    try:
        # Convert the NumPy array to a PIL image
        image = Image.fromarray(image_array)

        # Convert the PIL image to CTkImage
        image_tk = ctk.CTkImage(light_image=image, dark_image=image)  # Using same image for both light and dark modes

        # Update the display_label with the new CTkImage
        display_label.configure(image=image_tk)
        display_label.image = image_tk  # Keep a reference to avoid garbage collection
    except Exception as e:
        print("An error occurred while displaying the image:", e)

# Apply Filter or Noise
def apply():
    global image_data, image_name
    sigma = float(sigma_entry.get()) if sigma_entry.get() else 0
    probality = float(probabilite_entry.get()) if probabilite_entry.get() else 0.05
    window_size = int(window_size_entry.get()) if window_size_entry.get() else 3
    noise_type = noise_option.get()
    base_name, ext = os.path.splitext(image_name)
    
    if noise_type == "Gaussian Noise":
        bruit = bruit_gaussien(0, sigma, image_data.shape[1], image_data.shape[0])
        processed_image = np.clip(image_data + bruit, 0, 255).astype(np.uint8)
        new_image_name = f"{base_name}_{noise_type}_{sigma}{ext}"
        cv2.imwrite(new_image_name, processed_image)
    elif noise_type == "Salt and Pepper Noise":
        processed_image = bruit_poivre_et_sel(image_data, 0.05)
        new_image_name = f"{base_name}_{noise_type}_{probality}{ext}"
        cv2.imwrite(new_image_name, processed_image)
    elif noise_type == "Mean Filter":
        processed_image = filtre_moyen(image_data, window_size).astype(np.uint8)
        new_image_name = f"{base_name}_{noise_type}_{window_size}{ext}"
        cv2.imwrite(new_image_name, processed_image)
    elif noise_type == "Gaussian Filter":
        processed_image = filtre_gaussian(image_data, sigma)
        new_image_name = f"{base_name}_{noise_type}_{sigma}{ext}"
        cv2.imwrite(new_image_name, processed_image)
    elif noise_type == "Median Filter":
        processed_image = filtre_median(image_data, window_size)
    else:
        processed_image = filtre_min_max(image_data, window_size)
        
    psnr_value = peack_signal_noise_ration(image_data, processed_image)
    psnr_label.configure(text=f"PSNR: {psnr_value:.2f}")
    display_image(processed_image)


# Callback function to update secondary options
def update_secondary_options(*args):
    choice = primary_option_var.get()
    print(f"Primary option selected: {choice}")
    
    # Update the secondary options based on the primary choice
    if choice == "Noise":
        secondary_option.configure(values=["Gaussian Noise", "Salt and Pepper Noise"])
    elif choice == "Filter":
        secondary_option.configure(values=["Mean Filter", "Gaussian Filter", "Median Filter", "Min-Max Filter"])
    
    # Clear the secondary combobox selection
    secondary_option.set("")  # Debugging line

# Interface Widgets
app = ctk.CTk()
app.title("Image Noise and Filtering")
app.geometry("600x600")
# Global variables
image_data = None
image_name = ""

primary_label = ctk.CTkLabel(app, text="Select Noise or Filter:")
primary_label.pack(pady=10)

# Create a StringVar to monitor combobox value
primary_option_var = ctk.StringVar(app)

# Create the primary combobox and associate it with the StringVar
primary_option = ctk.CTkComboBox(app, values=["Noise", "Filter"], variable=primary_option_var)
primary_option.pack()

# Trace changes to the primary_option variable
primary_option_var.trace_add("write", update_secondary_options)

secondary_label = ctk.CTkLabel(app, text="Select Type:")
secondary_label.pack(pady=10)

# Create the secondary combobox
secondary_option = ctk.CTkComboBox(app, values=[])
secondary_option.pack()

# Initialize noise_option
noise_option = secondary_option

sigma_label = ctk.CTkLabel(app, text="Sigma:")
sigma_label.pack()

sigma_entry = ctk.CTkEntry(app)
sigma_entry.pack(pady=5)

probabilite_label = ctk.CTkLabel(app, text="Probability:")
probabilite_label.pack()

probabilite_entry = ctk.CTkEntry(app)
probabilite_entry.pack(pady=5)

window_size_label = ctk.CTkLabel(app, text="Window Size:")
window_size_label.pack()

window_size_entry = ctk.CTkEntry(app)
window_size_entry.pack(pady=5)

apply_button = ctk.CTkButton(app, text="Apply")
apply_button.pack(pady=20)

psnr_label = ctk.CTkLabel(app, text="PSNR: ")
psnr_label.pack()

display_label = ctk.CTkLabel(app)
display_label.pack(pady=20)

# Load image button
load_button = ctk.CTkButton(app, text="Load Image", command=load_image)
load_button.pack(pady=20)

app.mainloop()