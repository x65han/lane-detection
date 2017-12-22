# import packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
import utility
from moviepy.editor import VideoFileClip
from IPython.display import HTML

# pipeline
input_dir = "test_images/"
output_dir = "test_images_output/"

def process_image(img, isRGB=True):
    ysize = img.shape[0]
    xsize = img.shape[1]
    # convert to gray scale
    gray_img = utility.grayscale(img)
    # apply Gaussian Blur
    kernel_size = 5
    blur_img = utility.gaussian_blur(gray_img, kernel_size)
    # Canny Edge detection
    low_threshold = 60
    high_threshold = 150
    edge_img = utility.canny(blur_img, low_threshold, high_threshold)
    # region filter
    vertices = np.array([[(0,ysize),(xsize/2-20, ysize/2 + 60), (xsize/2 + 20, ysize/2 + 60), (xsize,ysize)]], dtype=np.int32)
    mask_img = utility.region_of_interest(edge_img, vertices)
    # Hough Line parameters
    rho = 3              # distance resolution in pixels of the Hough grid
    theta = np.pi/180    # angular resolution in radians of the Hough grid
    threshold = 70       # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 70 # minimum number of pixels making up a line
    max_line_gap = 250   # maximum gap in pixels between connectable line segments
    # Hough space
    if isRGB == True:
        line_img = utility.hough_lines(mask_img, rho, theta, threshold, min_line_length, max_line_gap, isRGB=True)
    else:
        line_img = utility.hough_lines(mask_img, rho, theta, threshold, min_line_length, max_line_gap, isRGB=False)
    # overlap image
    result = utility.weighted_img(line_img, img, 0.6, 1, 0)
    return result

# process images
for index, img_name in enumerate(os.listdir(input_dir)):
    img = cv2.imread(input_dir + img_name)
    result = process_image(img, False)
    cv2.imwrite(output_dir + img_name, result)
    print("saved " + output_dir + img_name)

# process video
white_output = 'test_videos_output/solidWhiteRight.mp4'
clip1 = VideoFileClip('test_videos/solidWhiteRight.mp4')
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

yellow_output = 'test_videos_output/solidYellowLeft.mp4'
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(yellow_output, audio=False)

challenge_output = 'test_videos_output/challenge.mp4'
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
challenge_clip.write_videofile(challenge_output, audio=False)