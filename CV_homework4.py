#Matt Evanego and Dylan Regan HW 4
from scipy import signal as sig
import numpy as np
import cv2
import math
import random
import copy

import time

#K-means segmentation
def question_1(img, k=10):
    #gets the random points
    cluster_centers = []
    while len(cluster_centers) != k:
        cluster_centers = [random.choice(x) for x in random.choices(img, k=k)]
        cluster_centers = np.unique(cluster_centers, axis=0)

    new_cluster_centers = cluster_centers
    finished = False
    count = 0
    while not finished and count < 10:
        collections = [[] for _ in range(k)]
        cluster_sum = np.zeros((k,3))
        count += 1
        cluster_centers = new_cluster_centers
        for x in range(ROWS):
            for y in range(COLS):
                min_distance = math.inf
                cluster = 0
                for c in range(k):
                    distance = np.linalg.norm(cluster_centers[c] - img[x][y])
                    if (min_distance > distance):
                        min_distance = distance
                        cluster = c
                collections[cluster].append(img[x][y])
                cluster_sum[cluster] += img[x][y]
        new_cluster_centers = np.empty((k,3))

        for c in range(k):
            new_cluster_centers[c] = np.array([(cluster_sum[c][0] / len(collections[c])),
                                        (cluster_sum[c][1] / len(collections[c])),
                                        (cluster_sum[c][2] / len(collections[c]))])
       
        if np.count_nonzero(np.subtract(new_cluster_centers, cluster_centers)) == 0:
            finished = True
        else:
            print(np.subtract(new_cluster_centers, cluster_centers))
            print(np.count_nonzero(np.subtract(new_cluster_centers, cluster_centers)))

    cluster_centers = cluster_centers.astype("int32")

    # after finished becomes true, we loop over the image and set it's color to the closest cluster_center
    for x in range(ROWS):
        for y in range(COLS):
            min_distance = math.inf
            cluster = 0
            for c in range(k):
                distance = np.linalg.norm(cluster_centers[c] - img[x][y])
                if (min_distance > distance):
                    min_distance = distance
                    cluster = c
            img[x][y] = cluster_centers[cluster]

    return [img, cluster_centers]

#SLIC algorithm
def question_2(img):
    cluster_centers = []
    clusters = dict()
    for x in range(0, ROWS, 50):
        for y in range(0, COLS, 50):
            if x+49 < ROWS and y+49 < COLS:
                cluster_centers.append([x+24,y+24,img[x+24][y+24][0],img[x+24][y+24][1],img[x+24][y+24][2]])
                clusters[(COLS // 50) * (x//50) + (y//50)] = []

    for x in range(ROWS):
        for y in range(COLS):
            if x+49 < ROWS and y+49 < COLS:
                clusters[(COLS // 50) * (x // 50) + (y // 50)] += [[x, y, img[x][y][0], img[x][y][1], img[x][y][2]]]
    for i in range(3):
        for cluster in range(len(clusters)):
            min_dif = math.inf
            for pixel in clusters[cluster]:
                if pixel[0] != 0 and pixel[0] != ROWS-1 and pixel[1] != 0 and pixel[1] != COLS-1:
                    dif = 0
                    for x in range(pixel[0]-1, pixel[0]+2):
                        for y in range(pixel[1]-1, pixel[1]+2):
                            dif += np.linalg.norm(cluster_centers[cluster][2:5] - img[x][y])
                    if (min_dif > dif):
                        min_dif = dif
                        center = pixel[0:2]
            cluster_centers[cluster][0:2] = center
        cluster_sum = np.zeros((len(cluster_centers),5))
        for x in range(len(clusters)):
            clusters[x] = []
        for x in range(ROWS):
            for y in range(COLS):
                min_distance = math.inf
                cluster = 0
                for c in range(len(clusters)):
                    if np.linalg.norm([cluster_centers[c][0]-x, cluster_centers[c][1]-y]) < 100:
                        distance = np.linalg.norm([(cluster_centers[c][0] - x) // 2, (cluster_centers[c][1] - y) // 2, cluster_centers[c][2]-img[x][y][0], cluster_centers[c][3]-img[x][y][1], cluster_centers[c][4]-img[x][y][2]])
                        if (min_distance > distance):
                            min_distance = distance
                            cluster = c
                clusters[cluster] += [[x,y,img[x][y][0],img[x][y][1],img[x][y][2]]]
                cluster_sum[cluster] += [x,y,img[x][y][0],img[x][y][1],img[x][y][2]]
        for c in range(len(cluster_centers)):
            cluster_centers[c] = [(cluster_sum[c][0] / len(clusters[c])),
                                  (cluster_sum[c][1] / len(clusters[c])),
                                  (cluster_sum[c][2] / len(clusters[c])),
                                  (cluster_sum[c][3] / len(clusters[c])),
                                  (cluster_sum[c][4] / len(clusters[c]))]

    img_copy = np.empty((ROWS, COLS, 3), dtype="uint8")   
    for c in range(len(clusters)):
        for p in clusters[c]:
            img[p[0]][p[1]] = cluster_centers[c][2:5]
            img_copy[p[0]][p[1]] = cluster_centers[c][2:5]

    #clean up code to make final image not so black     
    for x in range(ROWS):
        for y in range(COLS):
            if x < ROWS-1 and y < COLS-1:
                if all(img[x-1][y] == img[x+1][y]) and all(img[x-1][y] == img[x][y-1]) and all(img[x-1][y] == img[x][y+1]):
                    img[x][y] = img[x-1][y]
                    img_copy[x][y] = img[x-1][y]
                    
    #make other pixels black
    for x in range(ROWS):
        for y in range(COLS):
            if any(img[x][y] != img[x-1][y]) or any(img[x][y] != img[x][y-1]) or (x < ROWS-1 and any(img[x][y] != img[x+1][y])) or (y < COLS-1 and any(img[x][y] != img[x][y+1])):
                img_copy[x][y] = [0,0,0]
                
    return img_copy
        

#Pixel Classification
#Use mask_image to seperate sky and non-sky pixels
def question_3(mask_img, img):
    sky_image = np.zeros((ROWS, COLS, 3))
    non_sky_image = np.zeros((ROWS, COLS, 3))
    for x in range(ROWS):
        for y in range(COLS):
            if np.array_equal(mask_img[x][y], [255, 255, 255]):
                sky_image[x][y] = img[x][y]
                non_sky_image[x][y] = [255, 255, 255]
            else:
                non_sky_image[x][y] = img[x][y]
                sky_image[x][y] = [255, 255, 255]
    
    return [sky_image, non_sky_image]

#Paints anything that is classified as a sky orange
def classify_sky(visual_word_sky, visual_word_non_sky, img):
    print("Started Clasify Sky")
    rows, cols, _ = np.shape(img)
    result_image = np.zeros((rows, cols, 3))
    is_sky = False
    for x in range(rows):
        for y in range(cols):
            min_distance = math.inf
            for c in range(len(visual_word_non_sky)):
                sky_distance = np.linalg.norm(visual_word_sky[c] - img[x][y])
                if sky_distance < min_distance:
                    min_distance = sky_distance
                    is_sky = True
                non_sky_distance = np.linalg.norm(visual_word_non_sky[c] - img[x][y])
                if non_sky_distance < min_distance:
                    min_distance = non_sky_distance
                    is_sky = False
            if is_sky:
                result_image[x][y] = [3, 112, 252]
            else:
                result_image[x][y] = img[x][y]
    return result_image


def display(name, mat):
    mat = np.interp(mat,(mat.min(), mat.max()),(0, 255)).astype(np.uint8)
    cv2.imshow(name, mat)

def save(name, mat):
    mat = np.interp(mat,(mat.min(), mat.max()),(0, 255)).astype(np.uint8)
    cv2.imwrite(name, mat)


#Driver
img1 = cv2.imread("white-tower.png", cv2.IMREAD_COLOR)

img_mask = cv2.imread("sky_train_no_sky.jpg", cv2.IMREAD_COLOR)
img_sky = cv2.imread("sky_train.jpg", cv2.IMREAD_COLOR)
sky_test = cv2.imread("sky_test4.jpg", cv2.IMREAD_COLOR)

ROWS, COLS, _ = np.shape(img1)
res = question_1(img1)
display("img1", res[0])
cv2.waitKey(0)




