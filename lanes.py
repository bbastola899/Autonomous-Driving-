import cv2
import numpy as np
# import matplotlib.pyplot as plt
# Used for knowing the dimensions of the lane
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = int(image.shape[0])
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [[x1, y1, x2, y2]]


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None

    for line in lines:
        for x1, y1, x2, y2 in line:
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope  < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)

    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])
        # reshaping 1 dimensional array into 4 elements
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image
#def display_lines(image, lines):
    #line_image = np.zeros_like(image)
    #if lines is not None:
        #for line in lines:
            #for x1, y1, x2, y2 in line:
                #cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    #return line_image
def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[
    (200, height), (1100, height), (550, 250)] ])
    mask = np.zeros_like(image)

    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


image = cv2.imread('test_image.jpg')
# Importing image into multidimensional array ***


lane_image = np.copy(image)
#This one is just for copying image

#canny_image = canny(lane_image)
#cropped_image = region_of_interest(canny_image)
#lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
#averaged_lines = display_lines(lane_image, lines)
#line_image = display_lines(lane_image, averaged_lines)
#combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1 )
# gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
# changing the colored image to gray so it would be easy for detecting the
#intensity change in the image
# blur = cv2.GaussianBlur(gray, (5,5), 0)
#Using GaussianBlur to remove noise from gray image using kernel
# with deviation 0

# canny = cv2.Canny(blur, 50, 150)
## Outling the strongest gradient in our blur image
#cv2.imshow('result', combo_image)
# plt.imshow(canny)

#cv2.waitKey(0)
# pyplot already contains the function imshow
# so we can use plt in case of cv2.imshow

#plt.imshow(canny)
# plt.show()
cap = cv2.VideoCapture("test2.mp4")
while (cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_canny = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1 )
    cv2.imshow('result', combo_image)
    if cv2.waitKey(1) == ord('q'):
        break
cap.realease()
cv2.destroyAllWindows()
