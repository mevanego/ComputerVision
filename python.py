#Matt Evanego - Homework 1
from PIL import Image
import numpy as np
import glob
import sys
import cv2;



def main():
    imagePath = input("Enter the path of your image: ")
    print(imagePath)

    sigma = input("Enter the sigma value: ")
    print(sigma)
    print(sys.version)

def filter_image(image, sigma):
    #image = imread(image)
    image = Image.open(image)
    image = np.asarray(image)
    #print(image)
    filter_size = 2 * int(4 * sigma + 0.5) + 1
    gaussian_filter = np.zeros((filter_size, filter_size), np.float32)
    m = filter_size//2
    n = filter_size//2
    
    for x in range(-m, m+1):
        for y in range(-n, n+1):
            x1 = 2*np.pi*(sigma**2)
            x2 = np.exp(-(x**2 + y**2)/(2* sigma**2))
            gaussian_filter[x+m, y+n] = (1/x1)*x2
    
    im_filtered = np.zeros_like(image, dtype=np.float32)
    for c in range(3):
        im_filtered[:, :, c] = np.convolve(image[:, :, c], gaussian_filter)
    return (im_filtered.astype(np.uint8))

#Takes in an image that already has a Gaussian Blur
def sobel_filter(image):
    filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    new_image_x = np.convolve(image, filter)
    new_image_y = np.convolve(image, np.flip(filter.T, axis=0))
    gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
    gradient_magnitude *= 255.0 / gradient_magnitude.max()

if __name__ == "__main__":
    main()