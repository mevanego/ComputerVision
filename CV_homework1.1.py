# Python code to read image
import cv2
import math
import numpy as np

def bounds(x,y,shape,sigma):
    if x < sigma:
        x1 = 0
        x2 = sigma * 2 + 1
    elif x > shape[0] - sigma - 1:
        x1 = shape[0] - sigma*2 - 1
        x2 = shape[0]
    else:
        x1 = x - sigma
        x2 = x + sigma + 1
    if y < sigma:
        y1 = 0
        y2 = sigma * 2 + 1
    elif y > shape[1] - sigma - 1:
        y1 = shape[1] - sigma*2 - 1 
        y2 = shape[1]
    else:
        y1 = y - sigma
        y2 = y + sigma + 1
    return (x1, x2, y1, y2)

def create_gaus_filter(sigma):
    gaus_filter = np.zeros((2 * sigma + 1, 2 * sigma + 1))
    for x in range(-sigma, sigma+1):
        for y in range(-sigma, sigma+1):
            body = 1 / (2 * math.pi * sigma ** 2)
            exp = (-(x ** 2 + y ** 2) / (2 * sigma ** 2)) 
            weight = body * math.e ** exp
            gaus_filter[x+sigma][y+sigma] = weight
    return gaus_filter

def gaus(p, fil):
    answer_sum = 0
    weight_sum = 0
    weighted_mat = np.multiply(p, fil)
    answer_sum = np.sum(weighted_mat)
    weight_sum = np.sum(fil)
    answer_sum *= 1 / weight_sum
    return answer_sum
        
def gaussian_filter(img, sigma):
    print("starting gaussian filter")
    shape = img.shape
    gaus_filter = create_gaus_filter(sigma)

    for x in range(0, shape[0]):
        for y in range(0, shape[1]):
            x1,x2,y1,y2 = bounds(x,y,shape,sigma)
            img[x][y] = gaus(img[x1:x2, y1:y2], gaus_filter)
    return img

def sobel_filter(image):
    print("starting sobel_filter")
    Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    rows, columns = np.shape(image) 
    sobel_filtered_image = np.zeros(shape=(rows, columns), dtype="uint8")  
    
    for i in range(rows - 2):
        for j in range(columns - 2):
            gx = np.sum(np.multiply(Gx, image[i:i + 3, j:j + 3]))
            gy = np.sum(np.multiply(Gy, image[i:i + 3, j:j + 3]))
            sobel_filtered_image[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)


    return sobel_filtered_image


def non_max_supression(image, T):
    print("starting non_max_supression")
    rows, columns = np.shape(image) 
    supression_image = np.zeros(shape=(rows, columns), dtype="uint8")

    d_y = np.zeros(shape=(rows, columns))
    d_x = np.zeros(shape=(rows, columns))
    edge_strength = np.zeros(shape=(rows, columns))
            
    for i in range(rows - 2):
        for j in range(columns - 2):
            d_y[i+1][j+1] = np.sum(np.multiply([-1,0,1], image[i, j:j+3]))
            d_x[i+1][j+1] = np.sum(np.multiply([-1,0,1], image[i:i+3, j]))
            edge_strength[i+1][j+1] = np.sqrt(d_y[i+1][j+1] ** 2 + d_x[i+1][j+1] ** 2)
    
    for i in range(rows - 2):
        for j in range(columns - 2):
            if edge_strength[i+1][j+1] < T:
                continue
            if d_x[i+1][j+1] != 0:
                theta = math.atan(d_y[i+1][j+1] / d_x[i+1][j+1])
            else:
                theta = math.pi / 2
            #horizontal
            if theta >= (math.pi * 3 / 8) or theta <= -(math.pi * 3 / 8):
                if edge_strength[i+1][j+1] > max(edge_strength[i+1][j],edge_strength[i+1][j+2]):
                    supression_image[i+1][j+1] = 255  #make sure this isnt edge strength

            #vertical
            elif theta >= -(math.pi * 1/8) and theta <= (math.pi * 1/8):
                if edge_strength[i+1][j+1] > max(edge_strength[i][j+1],edge_strength[i+2][j+1]):
                    supression_image[i+1][j+1] = 255 #make sure this isnt edge strength

            elif theta > (math.pi *  1/8) and theta < (math.pi * 3 / 8):
                if edge_strength[i+1][j+1] > max(edge_strength[i][j+2],edge_strength[i+2][j]):
                    supression_image[i+1][j+1] = 255  #make sure this isnt edge strength

            elif theta < -(math.pi * 1/8) and theta > -(math.pi * 3 / 8):
                if edge_strength[i+1][j+1] > max(edge_strength[i][j],edge_strength[i+2][j+2]):
                    supression_image[i+1][j+1] = 255  #make sure this isnt edge strength


    return supression_image


def main(image, sigma=3, threshold=20):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    img = gaussian_filter(img, sigma)
    img = sobel_filter(img)
    #img = non_max_supression(img, threshold)
        
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

main("red.pgm")
