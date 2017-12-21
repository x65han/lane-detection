import numpy as np
import cv2

# helper functions

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255, ) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    lines = lines[0]
    left = []
    right = []
    for line in lines:
        x1 = line[0]
        y1 = line[1]
        x2 = line[2]
        y2 = line[3]
        m = float(y2 - y1) / float(x2 - x1)
        if abs(m) < 0.5:
            continue
        if m < 0:
            left.append(line)
        else:
            right.append(line)

    # convert list to array
    left = np.asarray(left)
    right = np.asarray(right)
    if len(left.shape) != 2 or len(right.shape) != 2:
        return

    left_line = [left.mean(axis=0)[0], left.mean(axis=0)[1], left.mean(axis=0)[2], left.mean(axis=0)[3]]
    left_line = np.asarray(left_line).astype(int)
    # left_line = [left.min(axis=0)[0], left.min(axis=0)[1], left.max(axis=0)[2], left.max(axis=0)[3]]
    cv2.line(img, (left_line[0], left_line[1]), (left_line[2], left_line[3]), color, thickness)

    right_line = [right.mean(axis=0)[0], right.mean(axis=0)[1], right.mean(axis=0)[2], right.mean(axis=0)[3]]
    right_line = np.asarray(right_line).astype(int)
    # right_line = [right.max(axis=0)[0], right.max(axis=0)[1], right.min(axis=0)[2], right.min(axis=0)[3]]
    cv2.line(img, (right_line[0], right_line[1]), (right_line[2], right_line[3]), color, thickness)
        

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, isRGB=True):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if isRGB == True:
        draw_lines(line_img, lines)
    else:
        draw_lines(line_img, lines, [0, 0, 255])
    return line_img


# Python 3 has support for cool math symbols.


def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * alpha + img * beta + gamma
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)
