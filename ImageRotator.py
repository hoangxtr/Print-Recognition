import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# def getSkewAngle(cvImage) -> float:
#     # Prep image, copy, convert to gray scale, blur, and threshold
#     newImage = cvImage.copy()
#     h,w = newImage.shape[:2]

#     x_percent=0.25
#     y_percent=0.2
#     newImage = newImage[int(h*y_percent):int(h*(1-y_percent)), int(w*x_percent): int(w*(1-x_percent))]

#     gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (3, 3), 0)
#     thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

#     # Apply dilate to merge text into meaningful lines/paragraphs.
#     # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
#     # But use smaller kernel on Y axis to separate between different blocks of text
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
#     dilate = cv2.dilate(thresh, kernel, iterations=5)

#     # Find all contours
#     contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     contours = sorted(contours, key = cv2.contourArea, reverse = True)

#     # Find largest contour and surround in min area box
#     largestContour = contours[0]
#     minAreaRect = cv2.minAreaRect(largestContour)
#     print(minAreaRect)

#     # Determine the angle. Convert it to the value that was originally used to obtain skewed image
#     angle = minAreaRect[-1]
#     angle = 90-angle
#     print(angle)

#     box = cv2.boxPoints(minAreaRect)
#     box = np.int0(box)

#     cv2.drawContours(newImage,[box],0,(0,0,255),2)
#     if minAreaRect[1][0] > minAreaRect[1][1]: angle += 90
#     # if angle < -45:
#     #     angle = 90 + angle
#     return angle, newImage

# # Rotate the image around its center
# def rotateImage(cvImage, angle: float):
#     newImage = cvImage.copy()
#     (h, w) = newImage.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, angle, 1.0, )
#     newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#     return newImage

# # Deskew image
# def deskew(cvImage):
#     angle, newImage = getSkewAngle(cvImage)
#     return rotateImage(cvImage, -1.0 * angle), rotateImage(newImage, -1.0 * angle), angle



class ImageRotator:
    def __init__(self) -> None:
        pass

    def get_rotated_angle(self, image) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
        image = image.copy()
        h,w = image.shape[:2]

        x_percent=0.25
        y_percent=0.2
        cropped_image = image[int(h*y_percent):int(h*(1-y_percent)), int(w*x_percent): int(w*(1-x_percent))]

        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Apply dilate to merge text into meaningful lines/paragraphs.
        # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
        # But use smaller kernel on Y axis to separate between different blocks of text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        dilate = cv2.dilate(thresh, kernel, iterations=5)

        # Find all contours
        contours, _ = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)

        # Find largest contour and surround in min area box
        largestContour = contours[0]
        minAreaRect = cv2.minAreaRect(largestContour)

        # Determine the angle. Convert it to the value that was originally used to obtain skewed image
        angle = minAreaRect[-1]
        angle = 90-angle

        box = cv2.boxPoints(minAreaRect)
        box = np.int0(box)

        if minAreaRect[1][0] > minAreaRect[1][1]: angle += 90
        return angle

    def rotate_image(self, image, angle: float):
        image = image.copy()
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0, )
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_AREA)
        return image
    
    def process(self, image):
        angle = self.get_rotated_angle(image)
        return self.rotate_image(image, -1.0 * angle)

all_paths = glob.glob('blue/*.tif')

m = -np.inf
rotator = ImageRotator()
for i in range(10):
    print('=========')
    img_path = all_paths[np.random.randint(len(all_paths))]
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    ret = rotator.process(img)

    cv2.imshow('ret', ret)
    cv2.imshow('image', img)

    cv2.waitKey(0)

cv2.destroyAllWindows()
