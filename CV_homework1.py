# Python code to read image
import cv2
import math
import numpy as np
 
# To read image from disk, we use
# cv2.imread function, in below method,
img = cv2.imread("kangaroo.pgm", cv2.IMREAD_COLOR)

i = 2

def gaus(p, o, pos):
    #print(p, o, pos)
    answer_sum = 0
    weight_sum = 0
    weight = 0
    for x in range(len(p)):
        for y in range(len(p[x])):
            body = 1 / (2 * math.pi * o ** 2)
            exp = (-((x-pos[0]) ** 2 + (y-pos[1]) ** 2) / (2 * o ** 2)) 
            weight = body * math.e ** exp
            answer_sum += p[x][y][0] * weight
            weight_sum += weight
    answer_sum *= 1 / weight_sum
    return answer_sum


def gaussian_filter(img, sigma):
    x_shape, y_shape, z_shape = img.shape
    for x in range(0, x_shape):
        for y in range(0, y_shape):
            new_pixel = gaus(img[max(x-sigma,0):min(x+sigma+1,x_shape), max(y-sigma,0):min(y+sigma+1,y_shape)], sigma, (min(sigma, x), min(sigma, y)))
            img[x][y] = new_pixel
    return img


def sobel_filter(image):
    s_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    new_image_x = convolution(image, s_filter)
    new_image_y = convolution(image, np.flip(s_filter.T, axis=0))
    gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
    gradient_magnitude *= 255.0 / gradient_magnitude.max()
    return gradient_magnitude


def convolution(image, kernel):
    print("convolving")
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])

    return output

img = gaussian_filter(img, 4)

img = sobel_filter(img)


            


# Creating GUI window to display an image on screen
# first Parameter is windows title (should be in string format)
# Second Parameter is image array
cv2.imshow("image", img)
 
# To hold the window on screen, we use cv2.waitKey method
# Once it detected the close input, it will release the control
# To the next line
# First Parameter is for holding screen for specified milliseconds
# It should be positive integer. If 0 pass an parameter, then it will
# hold the screen until user close it.
cv2.waitKey(0)
 
# It is for removing/deleting created GUI window from screen
# and memory
cv2.destroyAllWindows()
