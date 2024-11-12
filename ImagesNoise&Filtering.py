import random
import math
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


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


# filtre moyen
def filtre_moyen(img, t):
    m, n = img.shape
    mask = np.ones([t, t], dtype = int)
    mask = mask / t**2

    img_new = np.zeros([m, n])
    for i in range(1, m-1):
        for j in range(1, n-1):
            temp = img[i-1, j-1]*mask[0, 0]+img[i-1, j]*mask[0, 1]+img[i-1, j + 1]*mask[0, 2]+img[i, j-1]*mask[1, 0]+ img[i, j]*mask[1, 1]+img[i, j + 1]*mask[1, 2]+img[i + 1, j-1]*mask[2, 0]+img[i + 1, j]*mask[2, 1]+img[i + 1, j + 1]*mask[2, 2]
            img_new[i, j]= temp

    return img_new


def filtre_gaussian(image, sigma, kernel_size=5):
    image_data = np.array(image)
    
    # If the image is grayscale, convert it to a 3-channel image (for consistency)
    if len(image_data.shape) == 2:
        noisy_image = cv2.cvtColor(image_data, cv2.COLOR_GRAY2BGR)
    else:
        noisy_image = image_data
    # Apply Gaussian filter
    filtered_image = cv2.GaussianBlur(noisy_image, (kernel_size, kernel_size), sigma)
    
    return filtered_image

def peack_signal_noise_ration(img_origin, img_bruit):
    m, n = img_origin.shape
    some = 0
    r = 255
    for i in range(1, m-1):
        for j in range(1, n-1):
            some = some + (img_origin[i, j] - img_bruit[i, j]) ** 2

    pnsr = 10 * math.log10(r ** 2 / (some / (m * n)))
    print('peack signal noise ration (PSNR) :', pnsr)


# Load and convert the image
image_path = 'ford-mustang.jpg'
image = Image.open(image_path).convert("L")
image_data = np.array(image)

# Noise parameters
mu = 0

hauteur, largeur = image_data.shape

wanted = print("Enter the type of noise or filtre you want ? g = Gaussian noise, sp = Salt and Pepper noise :, fm = Mean filter, fg = Gaussian filter")
wanted = str(input())
if wanted == "g":
    sigma = print("Enter the value of sigma: ")
    sigma = float(input()) 
    # Generate Gaussian noise and add to the image
    bruit = bruit_gaussien(mu, sigma, largeur, hauteur)
    image_gaussienne = np.clip(image_data + bruit, 0, 255).astype(np.uint8)
    cv2.imshow('Image Originale', image_data)
    cv2.imshow(f'Image avec Bruit Gaussien avec sigma {sigma} ', image_gaussienne)
    peack_signal_noise_ration(image_data, image_gaussienne)
    
elif wanted == "sp":
    prob = 0.05
    # Add salt-and-pepper noise
    image_poivre_sel = bruit_poivre_et_sel(image_data, prob)
    cv2.imshow('Image Originale', image_data)
    cv2.imshow(f'Image avec Bruit Poivre et Sel avec prob de {prob}', image_poivre_sel)
    peack_signal_noise_ration(image_data, image_poivre_sel)
    
elif wanted == "fm":
    image_filtre_moyen = filtre_moyen(image_data , 3)
    cv2.imshow('Image Originale', image_data)
    cv2.imshow('Image avec Filtre Moyen', image_filtre_moyen.astype(np.uint8))
    peack_signal_noise_ration(image_data, image_filtre_moyen)
    
elif wanted == "fg":
    sigma = print("Enter the value of sigma: ")
    sigma = float(input())
    filtered_image_gaussian = filtre_gaussian(image, sigma, kernel_size=5)
    cv2.imshow('Image Originale', image_data)
    cv2.imshow('Image avec Filtre gaussian', filtered_image_gaussian)
    peack_signal_noise_ration(image_data, filtered_image_gaussian)
    
else:
    print("Invalid input")

# # Generate Gaussian noise and add to the image
# bruit = bruit_gaussien(mu, sigma, largeur, hauteur)
# image_gaussienne = np.clip(image_data + bruit, 0, 255).astype(np.uint8)

# # Add salt-and-pepper noise
# prob = 0.05  # Probability of noise
# image_poivre_sel = bruit_poivre_et_sel(image_data, prob)

# Apply mean filter
# image_filtre_moyen = filtre_moyen(image_data , 3)
# # Apply Gaussian filter to the image
# filtered_image_gaussian = filtre_gaussian(image, kernel_size=5, sigma=10)

# Display images
# cv2.imshow('Image Originale', image_data)
# cv2.imshow('Image avec Bruit Gaussien', image_gaussienne)
# cv2.imshow('Image avec Bruit Poivre et Sel', image_poivre_sel)
# cv2.imshow('Image avec Filtre Moyen', image_filtre_moyen.astype(np.uint8))
# cv2.imshow('Image avec Filtre gaussian', filtered_image_gaussian)



# Wait for a key to close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()