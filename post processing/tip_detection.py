import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import generic_filter
from skimage.morphology import medial_axis
from scipy.ndimage import binary_dilation
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line
from skimage import data
from matplotlib import cm
import pandas as pd
from skimage.morphology import skeletonize
import math

import matplotlib.pyplot as plt

def tip_detector_right(path):
    image = cv2.imread(path)
    filename = os.path.basename(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]
    output = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S )
    (numLabels, labels, stats, centroids) = output
    mask = np.zeros(gray.shape, dtype="uint8")
    
    detected_point = []
    for i in range(1, numLabels):
        area = stats[i, cv2.CC_STAT_AREA]
        output = image.copy()
        cX, cY = centroids[i]
        x1 = stats[i, cv2.CC_STAT_LEFT]
        y2 = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        keepArea = area > 2000
        if (keepArea):
            componentMask = (labels == i).astype("uint8") * 255
#             cv2.rectangle(output, (x1, y2), (x1 + w, y2 + h), (0, 255, 0), 3)
#             cv2.circle(output, (int(x1+w), int(y2)), 10, (255, 0, 0), -1)
            print(filename)
            print("The tip location is: ({} ,{})".format(x1+w, y2))
            mask = cv2.bitwise_or(mask, componentMask)
            detected_point.append(list((int(x1+w), int(y2))))
        
    # Create a new figure and plot all labels on it
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    for i in range(1, numLabels):
        area = stats[i, cv2.CC_STAT_AREA]
        x1 = stats[i, cv2.CC_STAT_LEFT]
        y2 = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        keepArea = area > 2000
        if (keepArea):
            ax.add_patch(plt.Rectangle((x1, y2), w, h, fill=False, edgecolor='green', linewidth=2))
#             ax.scatter(int(x1+w), int(y2), s=50, color='red')
    plt.show()

    return mask, detected_point



def tip_detector_left(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]
    output = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S )
    (numLabels, labels, stats, centroids) = output
    mask = np.zeros(gray.shape, dtype="uint8")
    
    detected_point = []
    for i in range(1, numLabels):
        area = stats[i, cv2.CC_STAT_AREA]
        cX, cY = centroids[i]
        x1 = stats[i, cv2.CC_STAT_LEFT]
        y2 = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        keepArea = area > 1000
        if (keepArea):
            output = image.copy()
            componentMask = (labels == i).astype("uint8") * 255
            cv2.rectangle(output, (x1, y2), (x1 + w, y2 + h), (0, 255, 0), 3)
            cv2.circle(output, (int(x1), int(y2 + h)), 10, (255, 0, 0), -1)
#             plt.imshow(output, cmap='gray')
#             plt.show()
#             print("[INFO] keeping connected component '{}'".format(i))
            print("The tip location is: ({} ,{})".format(x1, y2+h))
#             res = cv2.bitwise_or(componentMask, gray)
#             mean = np.mean(res)
#             if mean > 290:
            mask = cv2.bitwise_or(mask, componentMask)
            detected_point.append(list((int(x1), int(y2+h))))
        # Create a new figure and plot all labels on it
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    for i in range(1, numLabels):
        area = stats[i, cv2.CC_STAT_AREA]
        x1 = stats[i, cv2.CC_STAT_LEFT]
        y2 = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        keepArea = area > 2000
        if (keepArea):
            ax.add_patch(plt.Rectangle((x1, y2), w, h, fill=False, edgecolor='green', linewidth=2))
#             ax.scatter(int(x1), int(y2+h), s=50, color='red')
    plt.show()

    return mask, detected_point
#******************************************************************************************

# Line ends filter skeletonize
def lineEnds(P):
    """Central pixel and just one other must be set to be a line end"""
    return 255 * ((P[1]==255) and np.sum(P)==510)

def skeletonize(path):
    im = cv2.imread(path,0)
    # Skeletonize
    skel = (medial_axis(im)*255).astype(np.uint8)

    # Find line ends
    result = generic_filter(skel, lineEnds, (3, 3))
    
    indices = cv2.findNonZero(result)
    
    plt.imshow(result)
    plt.show()

    return result, indices

def hough_lines_image(img_path, ground_truth_path):
    # read image and ground truth data
    img = cv2.imread(img_path, 0)
    gray = skeletonize(img)
    
    lines = cv2.HoughLinesP(gray, rho=1, theta=np.pi/180, threshold=30, minLineLength=20, maxLineGap=60)
    detected_points = []
    for line in lines:
        detected_points.append([line[0][0], line[0][1]])
#         detected_points.append([line[0][2], line[0][3]])
    
    for points in lines:
        # Extracted points nested in the list
        x1,y1,x2,y2=points[0]
        # Draw the lines joing the points On the original image
        cv2.line(gray,(x1,y1),(x2,y2),(0,0,255),1)
#     cv2.imwrite('C:\\Users\\z004b1tz\\Desktop\\Master Thesis Project\\5 Layer Binary UNET\\Dataset\\output\\Hough lines\\'+str(os.path.basename(img_path)) , gray)
    
    cells = pd.read_csv(ground_truth_path, header=None , engine = 'python')
    gt_points = []
    for idx,row in cells.iterrows():
        x_start = row.iloc[0]
        y_start = row.iloc[1]
        x_end = row.iloc[2]
        y_end = row.iloc[3]
        gt_points.append([x_start, y_start])
    
    # get starting points of detected lines
    detected_points = []
    for line in lines:
        detected_points.append([line[0][0], line[0][1]])
        detected_points.append([line[0][2], line[0][3]])
    
    # calculate error between detected points and ground truth
    distance_errors = []
    for point in detected_points:
        for gt_point in gt_points:
            distance_error = np.sqrt((point[0] - gt_point[0]) ** 2 + (point[1] - gt_point[1]) ** 2)
            distance_errors.append(distance_error)
    
    # remove duplicates from errors list
    errors = list(set(distance_errors))
    # sort errors in ascending order
    errors.sort()
    # extract the filename from the img_path
    file_name = os.path.basename(img_path)
    
    err = []
    # get the number of errors with the minimum error
    num_of_detected_points_with_min_errors = len(gt_points)
    detected_points_with_min_errors = []
    angle_errors = []
    for gt_point in gt_points:
        min_error = float('inf')
        min_point = None
        for point in detected_points:
            error = np.sqrt((point[0] - gt_point[0]) ** 2 + (point[1] - gt_point[1]) ** 2)
            if error < min_error:
                min_error = round(error, 2)
                min_point = point
                gt_angle = math.atan2(gt_point[1], gt_point[0])
        detected_angle = math.atan2(min_point[1], min_point[0])
        angle_error = round(abs(detected_angle - gt_angle),4)
        angle_errors.append(angle_error)
        err.append(min_error)
        detected_points_with_min_errors.append(min_point)
        detected_points.remove(min_point)
    
    result = {
        'file_name': file_name,
        'ground_truth_points': gt_points,
        'detected_points_with_min_errors': detected_points_with_min_errors,
        'distance_error': err,
        'angle_error' : angle_errors
    }
    return result

def get_min_error_points_folder(folder_path):
    result = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".PNG"):
            img_path = os.path.join(folder_path, file_name)
            ground_truth_path = os.path.join(folder_path, file_name.replace('.PNG', '.raw.csv'))
            result.append(hough_lines_image(img_path, ground_truth_path))
    return result

#         return mse_lines