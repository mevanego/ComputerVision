#Matt Evanego and Dylan Regan HW 3
from scipy import signal as sig
import numpy as np
import cv2
import math
import random
import copy
import imutils

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

def non_max_suppression(img,sigma):
    rows, columns = np.shape(img)
    nms = np.zeros(shape=(rows, columns))
    for x in range(sigma,rows-sigma):
        for y in range(sigma,columns-sigma):
            if img[x,y] == 0:
                continue
            if img[x,y] == img[x-1:x+2, y-1:y+2].max():
                nms[x,y] = 255
    return nms


def question_1(good_img,sigma):
    img = copy.deepcopy(good_img)
    img = gaussian(img,sigma)
    dxx = sobel_x(sobel_x(img))
    dyy = sobel_y(sobel_y(img))
    lam1 = good_img * dxx
    lam2 = good_img * dyy
    R = lam1 * lam2 - .04 * (lam1 + lam2) ** 2
    flatR = R.flatten()
    flatR.sort()
    for x in range(len(R)):
        for y in range(len(R[x])):
            if flatR[-1000] > R[x][y]:
                R[x][y] = 0
    R = np.interp(R,(R.min(), R.max()),(0, 255)).astype(np.uint8)
    return R

def convert_to_sets(img):
    sets = []
    rows, cols = np.shape(img)
    for x in range(rows):
        for y in range(cols):
            if img[x][y] == 255:
                sets += [(x,y)]
    return sets

def ncc(set1,set2,img1,img2):
    img1 = img1.astype('float64')
    img2 = img2.astype('float64')
    best = []
    size = 3
    for g in set1:
        h_max = 0
        g_pixel = img1[g[0]-size:g[0]+size+1,g[1]-size:g[1]+size+1]
        g_mean = g_pixel.mean()
        g_avg = g_pixel - g_mean
        g_sqr = (g_avg ** 2).sum()
        for f in set2:
            f_pixel = img2[f[0]-size:f[0]+size+1,f[1]-size:f[1]+size+1]
            f_mean = f_pixel.mean()
            f_avg = f_pixel - f_mean
            H = ((g_avg * f_avg).sum()) / np.sqrt(g_sqr * (f_avg ** 2).sum())
            if H > h_max:
                h_max = H
                h_cord = (g,f)
        best += [(h_max,h_cord)]
    best.sort()
    best.reverse()
    return best[0:20]

#Question 2
def select_correspondences(set1, set2, num=30):
    correspondences=[]
    for x in range(num):
        correspondences.append([random.choice(set1), random.choice(set2)])
    return correspondences

#Display Functions
def make_color(img):
    rows, cols = np.shape(img)
    color = np.zeros(shape=(rows, cols, 3), dtype="uint8")
    for x in range(rows):
        for y in range(cols):
            color[x][y] = [img[x][y],img[x][y],img[x][y]]
    return color

def draw_line(image, best):
    for points in best:
        cv2.line(image, (points[0][1],points[0][0]), (points[1][1],points[1][0]), (0, 0, 255), 1)

def draw_points(image, points):
    for point in points:
        cv2.drawMarker(image, (point[1],point[0]), color=(0,255,0),markerSize=8, thickness=2)

def display(name, mat):
    mat = np.interp(mat,(mat.min(), mat.max()),(0, 255)).astype(np.uint8)
    cv2.imshow(name, mat)


#"Main" Function
sigma = 3

img1 = cv2.imread("uttower_left.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("uttower_right.jpg", cv2.IMREAD_GRAYSCALE)

#rotated45 = rotateImage(img1, 45)
rotated45 = imutils.rotate(img2, angle=45)
#cv2.imshow("Rotated", rotated45)
#cv2.waitKey(0)

rows, cols = np.shape(img1)

top_features1 = question_1(img1,sigma)
top_features2 = question_1(img2,sigma)


#display("image 1", top_features1)
#display("image 2", top_features2)
#cv2.waitKey(0)

nms1 = non_max_suppression(top_features1,sigma)
nms2 = non_max_suppression(top_features2,sigma)

set1 = convert_to_sets(nms1)
#print(set1)
set2 = convert_to_sets(nms2)

best = ncc(set1,set2,img1,img2)

#(value, ((x1,y1),(x1,y1))
#((y1,x1),(y1+width,x1))


best = [(x[1][0],(x[1][1][0],x[1][1][1]+cols)) for x in best]


line_em_up = np.hstack((img1,img2))
line_em_up = make_color(line_em_up)
draw_points(line_em_up, [x[0] for x in best])
draw_points(line_em_up, [x[1] for x in best])
draw_line(line_em_up, best[0:20])

cv2.imwrite("wide_boy.jpg", line_em_up)
display("pray", line_em_up)
cv2.waitKey(0)

correspondences = select_correspondences(set1, set2)
print(correspondences)

#rotation extra credit
top_features2_rotated = question_1(rotated45, sigma)
nms_rotated = non_max_suppression(top_features2_rotated, sigma)
set_rotated=convert_to_sets(nms_rotated)
best_rotated = ncc(set1, set_rotated, img1, rotated45)
best_rotated = [(x[1][0],(x[1][1][0],x[1][1][1]+cols)) for x in best_rotated]

line_em_up = np.hstack((img1,rotated45))
line_em_up = make_color(line_em_up)
draw_points(line_em_up, [x[0] for x in best_rotated])
draw_points(line_em_up, [x[1] for x in best_rotated])
draw_line(line_em_up, best_rotated[0:20])

#cv2.imwrite("rotated.jpg", line_em_up)
#display("extra credit", line_em_up)

#display("image 1", nms1)
#display("image 2", nms2)
#cv2.waitKey(0)

#display("image 1", img1)
#display("image 2", img2)
#cv2.waitKey(0)


#Pick 3 coorespondents
#2 triangles
#Get 2x3 mapping matrix
#Use 30 random points
#Simplex Affine transformation 


