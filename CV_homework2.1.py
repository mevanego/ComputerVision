from scipy import signal as sig
import numpy as np
import cv2
import math
import random
import operator

import copy

def create_gaus_filter(sigma):
    gaus_filter = np.zeros((2 * sigma + 1, 2 * sigma + 1))
    for x in range(-sigma, sigma+1):
        for y in range(-sigma, sigma+1):
            body = 1 / (2 * math.pi * sigma ** 2)
            exp = (-(x ** 2 + y ** 2) / (2 * sigma ** 2)) 
            weight = body * math.e ** exp
            gaus_filter[x+sigma][y+sigma] = weight
    sum_filter = sum(sum(gaus_filter))
    return gaus_filter * 1/sum_filter

def gaussian(img,sigma):
    kernel = create_gaus_filter(sigma)
    return sig.convolve2d(img, kernel, mode='same')

def sobel_x(img):
    kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    return sig.convolve2d(img, kernel_x, mode='same')

def sobel_y(img):
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return sig.convolve2d(img, kernel_y, mode='same')

def hessian(dxx,dxy,dyy):
    rows, columns = np.shape(dxx)
    hessian = np.zeros(shape=(rows, columns))
    for x in range(rows):
        for y in range(columns):
            hessian[x,y] = dxx[x,y] * dyy[x,y] - dxy[x,y] ** 2
    return hessian

def non_max_suppression(img,T):
    rows, columns = np.shape(img)
    nms = np.zeros(shape=(rows, columns))
    for x in range(1,rows-1):
        for y in range(1,columns-1):
            if img[x,y] < T:
                continue
            if img[x,y] == img[x-1:x+2, y-1:y+2].max():
                nms[x,y] = 255
    return nms

def find_RANSAC_points(img,sigma):
    rows, cols = np.shape(img)
    arr = []
    for x in range(sigma, rows-sigma):
        for y in range(sigma, cols-sigma):
            if img[x,y] == 255:
                arr.append((y,x))
    return arr

def distance(line,point):
    top = abs(line[0] * point[0] + line[1] * point[1] + line[2])
    bottom = math.sqrt(line[0] ** 2 + line[1] ** 2)
    return top / bottom

def line(p1,p2):
    a = p1[1] - p2[1]
    b = p2[0] - p1[0]
    c = a * p1[0] + b * p1[1]
    return [a,b,-c]

    

def RANSAC(points,max_distance):
    best = [[],[],[],[]]
    for i in range(10000):
        choosen_one = random.choice(points)
        choosen_two = random.choice(points)
        while choosen_one == choosen_two:
            choosen_two = random.choice(points)
        the_line = line(choosen_one, choosen_two)
        point_collection = [choosen_one, choosen_two]
        for point in points:
            if point == choosen_one or point == choosen_two or point[0] - max_distance < min(choosen_one[0], choosen_two[0]) or point[0] + max_distance > max(choosen_one[0], choosen_two[0]) or point[1] - max_distance < min(choosen_one[1], choosen_two[1]) or point[1] + max_distance > max(choosen_one[1], choosen_two[1]):
                continue
            if distance(the_line, point) < max_distance:
                point_collection.append(point)

        def contains(n):
            return len(set(n).intersection(point_collection))
        overlap = list(map(contains, best))
        if any(overlap):
            max_index = overlap.index(max(overlap))
            if len(best[max_index]) < len(point_collection):
                best[max_index] = point_collection
        else:
            best.sort(key = len)
            if len(best[0]) < len(point_collection):
                best[0] = point_collection
    return best

def make_color(img):
    rows, cols = np.shape(img)
    color = np.zeros(shape=(rows, cols, 3), dtype="uint8")
    for x in range(rows):
        for y in range(cols):
            color[x][y] = [img[x][y],img[x][y],img[x][y]]
    return color

def draw_points(image, points):
    for point in points:
        cv2.drawMarker(image, point, color=(0,255,0),markerSize=8, thickness=2)

def draw_line(image, best):
    for points in best:
        cv2.line(image, points[0], points[1], (0, 0, 255), 2)

def display(mat):
    #mat = np.interp(mat,(mat.min(), mat.max()),(0, 255)).astype(np.uint8)
    cv2.imshow("image", mat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def hough_trasnform(accumulator):
    #best lines score angle, distance, total votes
    best_lines = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
    for theta in range(len(accumulator)):
        for rho in range(len(accumulator[0])):
            if(accumulator[theta, rho] > best_lines[0][2]):
                best_lines[0] = [theta, rho, accumulator[theta, rho]]
                best_lines.sort(key=operator.itemgetter(2))
    return best_lines
    

def make_hough_accumulator(points, rho, theta=180):
    accumulator = np.zeros(shape=(theta,rho))
    temp_rho = 0
    for point in points:
        for dtheta in range(theta):
            #ρ= x cos θ + y sin θ
            #print(point , dtheta)
            temp_rho = int(point[0] * math.cos(dtheta * math.pi / 180) + point[1] * math.sin(dtheta * math.pi / 180))
            accumulator[dtheta, temp_rho] += 1 
    return accumulator


good_img = cv2.imread("road.png", cv2.IMREAD_GRAYSCALE)

img = copy.deepcopy(good_img)

sigma = 2

img = gaussian(img,sigma)
dxx = sobel_x(sobel_x(img))
dxy = sobel_x(sobel_y(img))
dyy = sobel_y(sobel_y(img))

img = hessian(dxx,dxy,dyy)

img = non_max_suppression(img,100000)
display(img)

points = find_RANSAC_points(img,sigma)

best = RANSAC(points,1)

good_img = make_color(good_img)
for b in best:
    draw_points(good_img, b)
draw_line(good_img, best)


#Hough Transformation
#Use the diagnoal of image + 1
rows, cols = np.shape(img)

rho = int(math.sqrt(rows * rows + cols * cols) + 1)
accum = make_hough_accumulator(points, rho)

display(accum)
lines = hough_trasnform(accum)
print(lines)



display(good_img)
