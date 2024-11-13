import random
import math
import cv2
import numpy as np
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
def filtre_gaussian(image, sigma, kernel_size=5):
    noisy_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image
    return cv2.GaussianBlur(noisy_image, (kernel_size, kernel_size), sigma)

# Fonction PSNR
def peack_signal_noise_ration(img_origin, img_bruit):
    img_origin, img_bruit = img_origin.astype(np.float64), img_bruit.astype(np.float64)
    mse = np.mean((img_origin - img_bruit) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * math.log10(255 ** 2 / mse)
    print('PSNR:', psnr)

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

# Charger l'image
image_path = 'image_poivre_sel.png'
image = Image.open(image_path).convert("L")
image_data = np.array(image)

# Choix du type de bruit ou de filtre
print("Enter the type of noise or filter you want ? g = Gaussian noise, sp = Salt and Pepper noise, fm = Mean filter, fg = Gaussian filter, fme = Median filter, fmm = Min-Max filter")
wanted = input().strip()

if wanted == "g":
    sigma = float(input("Enter the value of sigma: "))
    bruit = bruit_gaussien(0, sigma, image_data.shape[1], image_data.shape[0])
    image_gaussienne = np.clip(image_data + bruit, 0, 255).astype(np.uint8)
    cv2.imwrite('image_gaussienne.png', image_gaussienne)
    peack_signal_noise_ration(image_data, image_gaussienne)

elif wanted == "sp":
    prob = 0.05
    image_poivre_sel = bruit_poivre_et_sel(image_data, prob)
    cv2.imwrite('image_poivre_sel.png', image_poivre_sel)
    peack_signal_noise_ration(image_data, image_poivre_sel)

elif wanted == "fm":
    image_filtre_moyen = filtre_moyen(image_data, 3)
    cv2.imwrite('image_filtre_moyen.png', image_filtre_moyen.astype(np.uint8))
    peack_signal_noise_ration(image_data, image_filtre_moyen)

elif wanted == "fg":
    sigma = float(input("Enter the value of sigma: "))
    filtered_image_gaussian = filtre_gaussian(image_data, sigma, kernel_size=5)
    cv2.imwrite('image_filtre_gaussian.png', filtered_image_gaussian)
    peack_signal_noise_ration(image_data, filtered_image_gaussian)

elif wanted == "fme":
    window_size = int(input("Enter the window size for Median filter: "))
    image_filtre_median = filtre_median(image_data, window_size)
    cv2.imwrite('image_filtre_median.png', image_filtre_median)
    peack_signal_noise_ration(image_data, image_filtre_median)

elif wanted == "fmm":
    window_size = int(input("Enter the window size for Min-Max filter: "))
    image_filtre_min_max = filtre_min_max(image_data, window_size)
    cv2.imwrite('image_filtre_min_max.png', image_filtre_min_max)
    peack_signal_noise_ration(image_data, image_filtre_min_max)

else:
    print("Invalid input")
