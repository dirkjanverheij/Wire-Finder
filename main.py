###############################################    Wire Finder program   ###############################################
# developed by Dirkjan Verheij
# contact for questions or comments at dirkjanverheij@ctn.tecnico.ulisboa.pt
########################################################################################################################

# import necessary packages
import numpy as np
import cv2
import math
import os

# import numerical functions from wire_finder_num_functions
from wire_finder_num_functions import sort
from wire_finder_num_functions import find_extremes
from wire_finder_num_functions import vertical_contact
from wire_finder_num_functions import horizontal_contact

# import image functions from imfunction file
from imfunctions import imadjust
from imfunctions import imsharpen
from imfunctions import areafilt
from imfunctions import imfill
from imfunctions import MAfilt
from imfunctions import mafilt
from imfunctions import imrotate
from imfunctions import eccentricityfilt
from imfunctions import imcrop
from imfunctions import clear_border
from imfunctions import perimeterfilt


# main function where the alignment markers and wire coordinates are found
def wire_finder(img):
    # converts image to grayscale image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # calls the detect edges function and rewrites the img variable
    img = detect_edges(img)
    # calls the isolate alignment markers function and saves the output to another variable
    img2 = isolate_alignment_markers(img)
    # calls the find alignment markers coordinates function
    alignment_markers_coordinates = find_alignment_markers_coordinates(img2)
    # checks if the alignment markers were successfully found, if not the image is ignored
    if type(alignment_markers_coordinates) is np.ndarray:
        # sort the alignment markers coordinates from top left to bottom right
        alignment_markers_coordinates = sort(alignment_markers_coordinates)

        # the image is not always straight and consequently we need to correct any offset. In order to do so we
        # we calculate the angle between the top alignment markers and rotate the image accordingly
        h = np.linalg.norm(alignment_markers_coordinates[0] - alignment_markers_coordinates[1])
        d = alignment_markers_coordinates[1, 1] - alignment_markers_coordinates[0, 1]
        rotation_angle = (math.asin(d / h) * 360) / (2 * math.pi)

        img2 = imrotate(img2, rotation_angle)
        # we also rotate the original image
        img = imrotate(img, rotation_angle)
        # as the alignment markers changed position we find them again
        alignment_markers_coordinates = sort(find_alignment_markers_coordinates(img2))

        # now we also want to crop the image as to contain only the area limited by the border around the alignment
        # markers. The crop area is defined by the alignment markers plus an extra 60 pixels in each direction
        crop_x = alignment_markers_coordinates[0, 0] - 60
        crop_y = alignment_markers_coordinates[0, 1] - 60
        crop_width = alignment_markers_coordinates[3, 0] - alignment_markers_coordinates[0, 0] + 120
        crop_height = alignment_markers_coordinates[3, 1] - alignment_markers_coordinates[0, 1] + 120

        img2 = imcrop(img2, crop_x, crop_y, crop_width, crop_height)
        # we also crop the original image
        img = imcrop(img, crop_x, crop_y, crop_width, crop_height)
        # as the alignment markers changed position we find them again
        alignment_markers_coordinates = sort(find_alignment_markers_coordinates(img2))

        # call the function that removes everything except one wire (if there is one)
        img3 = isolate_wire(img)
        # calls function that obtains the coordinates of the extremities
        first_extremity_coordinates, second_extremity_coordinates, angle_of_wire = find_wire_extremity_coordinates(img3)
        for x, y in alignment_markers_coordinates:
            cv2.circle(img, (x, y), 10, 255)
        if type(first_extremity_coordinates) is np.ndarray:
            cv2.circle(img, (int(first_extremity_coordinates[0]), int(first_extremity_coordinates[1])), 10, 255)
            cv2.circle(img, (int(second_extremity_coordinates[0]), int(second_extremity_coordinates[1])), 5, 255)
    else:
        first_extremity_coordinates = []
        second_extremity_coordinates = []
        angle_of_wire = []

    return alignment_markers_coordinates, first_extremity_coordinates, second_extremity_coordinates, angle_of_wire


# function that detect the edges of the input image
def detect_edges(img):
    # improves image quality
    img = imadjust(img)
    img = imsharpen(img)
    kernel = np.ones((4, 4), np.float32) / 25
    img = cv2.filter2D(img, -1, kernel)

    # edge detection using canny algorithm
    img = cv2.Canny(img, 5, 100)

    # clear pixels connected to border of the image
    img = clear_border(img)
    # dilates image to close paths, filters areas outside of interesting pixel ranges, fills inside of structures
    # and finally erodes the structures to revert dilation effects
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    img = cv2.dilate(img, kernel)
    img = areafilt(img, 100, 700)
    img = imfill(img)
    output = cv2.erode(img, kernel)

    return output


# function that isolates the alignment markers in the image
def isolate_alignment_markers(img):
    # removes any structure with area, major axis and minor axis outside the defined ranges
    img = areafilt(img, 300, 660)
    img = MAfilt(img, 25, 50)
    output = mafilt(img, 25, 50)
    # finds the contours of the remaining structures
    contours, _ = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # sometimes the filters are not enough to eliminate all structures except the alignment markers. When this happens
    # I apply an extra filter based on the perimeter of the structures
    if len(contours) > 4:
        output = perimeterfilt(img, 100, 140)
    contours, _ = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # if we still have more than four structures I apply an extra filter based on the eccentricity
    if len(contours) > 4:
        output = eccentricityfilt(img, 0, 0.5)
    return output


# function that finds the alignment marker coordinates
def find_alignment_markers_coordinates(img):
    # get contours from structures present in the image
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # if we have less then three structures (meaning that the isolate alignment markers functions failed) we simply
    # ignore this image else, we use the moments of the contours in the image to calculate the centroid.
    if len(contours) < 3:
        centroids = []
        return centroids
    else:
        centroids = np.zeros((len(contours), 2), np.int32)
        for idx, cnt in enumerate(contours):
            M = cv2.moments(cnt)
            centroids[idx, 0] = M["m10"] / M["m00"]
            centroids[idx, 1] = M["m01"] / M["m00"]

        # if we only have detected three alignment markers, we deduce the fourth from the coordinates of the existing
        # ones
        if len(centroids) == 3:
            # count how many alignment markers there on the right side and on the bottom side
            right = np.count_nonzero(centroids[:, 0] > 550)
            bottom = np.count_nonzero(centroids[:, 1] > 450)

            # if there are two markers on the right and two on the bottom. The missing one is in the top left corner.
            # we use the smallest values of (x,y)
            if right == 2 and bottom == 2:
                centroids = np.concatenate([centroids, np.array([[min(centroids[:, 0]), min(centroids[:, 1])]])])
            # if there is one marker on the right and two on the bottom. The missing one is in the top right corner.
            # we use the biggest value of x and smallest value of y
            elif right == 1 and bottom == 2:
                centroids = np.concatenate([centroids, np.array([[max(centroids[:, 0]), min(centroids[:, 1])]])])
            # if there is one marker on the right and one on the bottom. The missing one is in the bottom left corner.
            # we use the biggest values of (x,y)
            elif right == 1 and bottom == 1:
                centroids = np.concatenate([centroids, np.array([[max(centroids[:, 0]), max(centroids[:, 1])]])])
            # if there are two markers on the right and one on the bottom. The missing one is in the bottom right corner.
            # we use the smallest value of x and biggest value of y
            elif right == 2 and bottom == 1:
                centroids = np.concatenate([centroids, np.array([[min(centroids[:, 0]), max(centroids[:, 1])]])])

        return centroids


# function that isolates the best microwire
def isolate_wire(img):
    img = eccentricityfilt(img, 0.9)
    img = MAfilt(img, 35, 220)
    img = mafilt(img, 1, 20)
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 2:
        MA_to_sort = np.zeros(len(contours))
        for i, cnt in enumerate(contours):
            _, (ma, MA), _ = cv2.fitEllipse(cnt)
            MA_to_sort[i] = MA

        MA = np.sort(MA_to_sort, 0).flatten()
        img = MAfilt(img, MA[-2] - 1e-4, MA[-1] + 1e-4)

        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        ecc_to_sort = np.zeros(2)

        for i, cnt in enumerate(contours):
            _, (ma, MA), _ = cv2.fitEllipse(cnt)
            ecc_to_sort[i] = np.sqrt(1 - ma ** 2 / MA ** 2)

        ecc = np.sort(ecc_to_sort, 0).flatten()
        output = eccentricityfilt(img, ecc[-1] - 1e-4, ecc[-1] + 1e-4)
    elif len(contours) == 2:
        ecc_to_sort = np.zeros(len(contours))
        for i, cnt in enumerate(contours):
            _, (ma, MA), _ = cv2.fitEllipse(cnt)
            ecc_to_sort[i] = np.sqrt(1 - ma ** 2 / MA ** 2)

        ecc = np.sort(ecc_to_sort, 0).flatten()
        output = eccentricityfilt(img, ecc[-1] - 1e-4, ecc[-1] + 1e-4)
    else:
        output = img
    return output


# function that finds the micrwoire coordinates
def find_wire_extremity_coordinates(img):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        first_contact = []
        second_contact = []
        ang = []
    else:
        extrema = find_extremes(contours)
        _, _, ang = cv2.fitEllipse(contours[0])
        x, y, w, h = cv2.boundingRect(contours[0])
        if ang < 45 or ang > 135:
            top_half_wire = img[y:int(y + h / 2), x:x + w]
            bottom_half_wire = img[int(y + h / 2):y + h, x:x + w]
            top_half_contour, _ = cv2.findContours(top_half_wire, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            bottom_half_contour, _ = cv2.findContours(bottom_half_wire, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            top_half_area = cv2.contourArea(top_half_contour[0])
            bottom_half_area = cv2.contourArea(bottom_half_contour[0])
            if top_half_area >= bottom_half_area:
                first_contact = extrema[2, :]
                second_contact = extrema[3, :]
            elif top_half_area < bottom_half_area:
                first_contact = extrema[3, :]
                second_contact = extrema[2, :]
        elif 45 <= ang <= 135:
            left_half_wire = img[y:y + h, x:int(x + w / 2)]
            right_half_wire = img[y:y + h, int(x + w / 2):x + w]
            left_half_contour, _ = cv2.findContours(left_half_wire, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            right_half_contour, _ = cv2.findContours(right_half_wire, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            left_half_area = cv2.contourArea(left_half_contour[0])
            right_half_area = cv2.contourArea(right_half_contour[0])

            if left_half_area > right_half_area:
                first_contact = extrema[0, :]
                second_contact = extrema[1, :]
            elif left_half_area < right_half_area:
                first_contact = extrema[1, :]
                second_contact = extrema[0, :]

    return first_contact, second_contact, ang


# function that calculates the contact coordinates
def calculate_contact_coordinates(alignment_markers_coord, first_contact_coord, second_contact_coord, n, m, k, w,
                                  sample, angle):
    if (sum(first_contact_coord) == 0):
        first_path_coordinates = 0
        second_path_coordinates = 0
        return first_path_coordinates, second_path_coordinates

    # this is the length of the side of the square formed by the markers
    dist_between_alignment_markers_real = np.array([200, 200, np.sqrt(2 * 200 ** 2)])
    reference_alignment_cross = np.array([1393.75, 1393.75])
    dist_between_samples = np.array([392.5, 392.5])
    dist_between_devices = np.array([4950, 4950])
    conversion_factor = []

    for i in range(len(alignment_markers_coord) - 1):
        dist_between_alignment_markers_image = np.linalg.norm(
            alignment_markers_coord[0] - alignment_markers_coord[i + 1])
        conversion_factor.append(dist_between_alignment_markers_real[i] / dist_between_alignment_markers_image)

    conversion_factor = np.mean(conversion_factor)

    first_contact_coord = (first_contact_coord[0:] - alignment_markers_coord[0]) * conversion_factor
    second_contact_coord = (second_contact_coord[0:] - alignment_markers_coord[0]) * conversion_factor

    first_contact_coord = np.array(
        [first_contact_coord[0] + reference_alignment_cross[0] + (n - (m * 6 - 5)) * dist_between_samples[0] + (
                k - (w * 2 - 1)) * dist_between_devices[0],
         200 - first_contact_coord[1] + reference_alignment_cross[1] + (m - 1) * dist_between_samples[1] + (
                 w - 1) * dist_between_devices[1]])
    second_contact_coord = np.array(
        [second_contact_coord[0] + reference_alignment_cross[0] + (n - (m * 6 - 5)) * dist_between_samples[0] + (
                k - (w * 2 - 1)) * dist_between_devices[0],
         200 - second_contact_coord[1] + reference_alignment_cross[1] + (m - 1) * dist_between_samples[1] + (
                 w - 1) * dist_between_devices[1]])

    orientation_of_paths = [[1, 'left'], [0, 'down'], [0, 'down'], [0, 'down'], [0, 'down'],
                            [1, 'right'], [1, 'right'], [1, 'left'], [1, 'right'], [1, 'left'],
                            [1, 'right'], [1, 'left'], [1, 'left'], [0, 'down'], [0, 'down'],
                            [0, 'down'], [0, 'down'], [1, 'right'], [1, 'right'], [0, 'up'],
                            [0, 'up'], [0, 'up'], [0, 'up'], [1, 'left'], [1, 'left'], [1, 'right'],
                            [1, 'left'], [1, 'right'], [1, 'left'], [1, 'right'], [1, 'right'], [0, 'up'],
                            [0, 'up'], [0, 'up'], [0, 'up'], [1, 'left']]
    lead = np.array([[1, 1], [1, 1]])
    if orientation_of_paths[sample - 1][0] == 1:
        print('horizontal')
        lead = np.array([[-26.25 + reference_alignment_cross[0] + (n - (m * 6 - 5)) * dist_between_samples[0] + (
                k - (w * 2 - 1)) * dist_between_devices[0],
                          reference_alignment_cross[1] + (m - 1) * dist_between_samples[1] + (w - 1) *
                          dist_between_devices[1] + dist_between_alignment_markers_real[1] / 2],
                         [26.25 + reference_alignment_cross[0] + (n - (m * 6 - 5)) * dist_between_samples[0] +
                          (k - (w * 2 - 1)) * dist_between_devices[0] + dist_between_alignment_markers_real[1],
                          reference_alignment_cross[1] + (m - 1) * dist_between_samples[1] + (w - 1) *
                          dist_between_devices[1] + dist_between_alignment_markers_real[1] / 2]])
        first_lead, second_lead = horizontal_contact(first_contact_coord, second_contact_coord, lead, angle,
                                                     orientation_of_paths[sample - 1][1])
    elif orientation_of_paths[sample - 1][0] == 0:
        print('vertical')
        lead = np.array([[reference_alignment_cross[0] + (n - (m * 6 - 5)) * dist_between_samples[0] + (
                k - (w * 2 - 1)) * dist_between_devices[0] + dist_between_alignment_markers_real[1] / 2,
                          26.25 + reference_alignment_cross[1] + (m - 1) * dist_between_samples[1] + (w - 1) *
                          dist_between_devices[1] + dist_between_alignment_markers_real[1]],
                         [reference_alignment_cross[0] + (n - (m * 6 - 5)) * dist_between_samples[0] + (
                                 k - (w * 2 - 1)) * dist_between_devices[0] + dist_between_alignment_markers_real[
                              1] / 2,
                          -26.25 + reference_alignment_cross[1] + (m - 1) * dist_between_samples[1] + (w - 1) *
                          dist_between_devices[1]]])
        first_lead, second_lead = vertical_contact(first_contact_coord, second_contact_coord, lead, angle,
                                                   orientation_of_paths[sample - 1][1])

    return first_contact_coord, second_contact_coord, first_lead, second_lead


# function that writes the contact coordinates as well as the libreCAD commands to a file
def write_contacts_to_script(p_contact, n_contact):
    p_contact_file = open('P.txt', 'a')
    p_contact_file.write('pl\r\n')
    p_contact_file.write(str(p_contact[0] - 5) + ',' + str(p_contact[1]) + '\n')
    p_contact_file.write(str(p_contact[0] - 2.5) + ',' + str(p_contact[1] - np.sqrt(75 / 4)) + '\n')
    p_contact_file.write(str(p_contact[0] + 2.5) + ',' + str(p_contact[1] - np.sqrt(75 / 4)) + '\n')
    p_contact_file.write(str(p_contact[0] + 5) + ',' + str(p_contact[1]) + '\n')
    p_contact_file.write(str(p_contact[0] + 2.5) + ',' + str(p_contact[1] + np.sqrt(75 / 4)) + '\n')
    p_contact_file.write(str(p_contact[0] - 2.5) + ',' + str(p_contact[1] + np.sqrt(75 / 4)) + '\n')
    p_contact_file.write(str(p_contact[0] - 5) + ',' + str(p_contact[1]) + '\n')
    p_contact_file.write('kill\r\n')
    p_contact_file.close()
    n_contact_file = open('N.txt', 'a')
    n_contact_file.write('pl\r\n')
    n_contact_file.write(str(n_contact[0] - 5) + ',' + str(n_contact[1]) + '\n')
    n_contact_file.write(str(n_contact[0] - 2.5) + ',' + str(n_contact[1] - np.sqrt(75 / 4)) + '\n')
    n_contact_file.write(str(n_contact[0] + 2.5) + ',' + str(n_contact[1] - np.sqrt(75 / 4)) + '\n')
    n_contact_file.write(str(n_contact[0] + 5) + ',' + str(n_contact[1]) + '\n')
    n_contact_file.write(str(n_contact[0] + 2.5) + ',' + str(n_contact[1] + np.sqrt(75 / 4)) + '\n')
    n_contact_file.write(str(n_contact[0] - 2.5) + ',' + str(n_contact[1] + np.sqrt(75 / 4)) + '\n')
    n_contact_file.write(str(n_contact[0] - 5) + ',' + str(n_contact[1]) + '\n')
    n_contact_file.write('kill\r\n')
    n_contact_file.close()

    p_contact_path_file = open('P_L.txt', 'a')
    n_contact_path_file = open('N_L.txt', 'a')
    p_contact_path_file.write('pl\r\n')
    n_contact_path_file.write('pl\r\n')
    for i in range(9):
        p_contact_path_file.write(str(first_lead[i, 0]) + ',' + str(first_lead[i, 1]) + '\n')
        n_contact_path_file.write(str(second_lead[i, 0]) + ',' + str(second_lead[i, 1]) + '\n')
    p_contact_path_file.write('kill\r\n')
    n_contact_path_file.write('kill\r\n')
    p_contact_path_file.close()
    n_contact_path_file.close()


# checks if the script files already exist in the folder, if they do, they are removed to avoid overwriting
if os.path.exists("P.txt"):
    os.remove("P.txt")
if os.path.exists("N.txt"):
    os.remove("N.txt")
if os.path.exists("P_L.txt"):
    os.remove("P_L.txt")
if os.path.exists("N_L.txt"):
    os.remove("N_L.txt")

# variable containing the location of the folder where images are stored
image_folder = '/images/'

# the following variables are used to count which square/sample is being worked on
# counts the total number of wires contacted
number_of_devices = 0
# counts the device site number (a device site is the area defined by 4 alignment markers)
device_site_number = 1
# counts the sample number (a sample is a set of 36 possible samples)
sample_number = 0
# counts the column number of the matrix of the squares defined by the alignment markers
column_number = 0
# counts the line number of the matrix of the squares defined by the alignment markers
line_number_new = 1
line_number_old = 1

# extra variables used for the image folder
f = 1
folder = 'str(f)'
# picture names are written as P1010001.JPG, where 1010001 increments by one every picture
pic = 10001

# begin a cycle with 144 iterations (4 devices * 36 possible sample sites = 144)
for idx in range(144):
    device_site_number += 1
    print('Progress: sample ', str(sample_number), ' on device site', str(device_site_number))

    # loads image and calls the wire_finder function
    image_file = 'P10' + str(pic) + '.JPG'
    I = cv2.imread(image_folder + folder + image_file)
    alignment_markers, p_contact, n_contact, angle = wire_finder(I)

    # if the line number is even the sample count decreases and vice-versa (due to snake pattern)
    if line_number_new % 2 == 1:
        column_number += 1
    elif line_number_new % 2 == 0:
        column_number -= 1

    # saves the line number for later comparison
    line_number_old = line_number_new
    # calculates current line number
    line_number_new = math.ceil(column_number / 6)

    # if the line number has changed and is larger than the previous line number, we are at the last column.
    # However, the sample number iterates only by one and we need to add 5 to match it with the column number.
    # For example, we go 1,2,...,5,6 and the line number changes to 2. Now we need to go 12,11,...,8,7. So we add 5
    # (7+5 = 12) and start decreasing count (see above if).
    if line_number_new != line_number_old and line_number_new > line_number_old:
        column_number += 5
    # if the line number has changed and is smaller than the previous line number, we are at the first column.
    # However, the sample number was decreasing (12,11,...,8,7) and in the new line we need 13,14,...,17,18.
    # So we add 7 (6+7 = 13) and start increasing the count (see above if)
    elif line_number_new != line_number_old and line_number_new < line_number_old:
        column_number += 7
        line_number_new += 2

    # calculates the sample matrix (2 by 2) column number
    device_column_number = math.ceil(sample_number / 2)

    # checks if p_contact is not empty, meaning there is a wire to be contacted and calls the function to calculate
    # the contact coordinates and to write them to a file. Finally, it adds +1 to the total device count.
    if type(p_contact) == np.ndarray:
        p_contact, n_contact, first_lead, second_lead = calculate_contact_coordinates(alignment_markers,
                                                                                      p_contact,
                                                                                      n_contact,
                                                                                      column_number,
                                                                                      line_number_new,
                                                                                      device_site_number,
                                                                                      device_column_number,
                                                                                      sample_number,
                                                                                      angle)
        write_contacts_to_script(p_contact, n_contact)
        number_of_devices += 1
    # increases the picture count
    pic += 1
    # if we reach the last device site we reset the counting variables and increment the sample number
    if device_site_number == 36:
        sample_number = 0
        column_number = 0
        line_number_new = 1
        line_number_old = 1
        sample_number += 1
        f += 1
        folder = str(f)
        pic = 10001
