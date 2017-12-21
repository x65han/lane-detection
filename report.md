# **Finding Lane Lines on the Road** 

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* apply the pipeline on videos
* Reflect on your work in a written report

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline is made up of the following steps...
- convert image to gray scale
- gaussian blur the image with kernel_size of 5
- apply canny edge algorithm on image
- apply region filter to only consider the lower triangle where lane could be seen
- Utilize hough_lines to get the white line
- apply the highlighted lane image with the original image

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...
- Calculate slope and use slope to determine if the line is left or right
- filter out all slope below 0.5 (or nearly horizontal line)
- then average the left lane and then average the right lane

### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when the lane is curved. The red highlight could be inaccurate in this case. 

Another shortcoming is that the lane is a bit short at the moment. Improvement might be needed to draw longer lines.

### 3. Suggest possible improvements to your pipeline

Modify the draw_lines() function to draw better lines with the given information from hough_lines() function

### How to run this project?
- You can use Jupyter notebook
- Or use `python lane.py`