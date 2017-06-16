# **Finding Lane Lines on the Road** 

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My lane-detection peline consisted of 7 steps. 

__Step__ __1__: Appply gray scale
 
__Step__ __2__: Apply gaussian blur

__Step__ __3__: Find canny edges

__Step__ __4__: Find area of interest in an image

__Step__ __5__: Find lines using hough transformation

__Step__ __6__: Draw the output from (5) on top of the original input image

__Step__ __7__: Draw a pair of left and right lines that represent the corresponding lanes on the road.


Steps 1 to 6 are achieved using the helper functions provided with the project, but
choosing (tuning) appropriate parameters.

__Drawing an extended left and right lane__
Step 7 required modification of the draw_lines util function (provided with the project) to extrapolate the line segments (found using hough transformation (Step 5)) into two single lines the represent the left and right lane in the image.
A brief overview of the above extrapolation is provided below.

draw_lines now makes use of a newly added helper function - 
def extrapolate(x_size, y_size, lines):
_where_:
x_size _is the length of the image_
y_size _is the height of the image_
lines _represent the set of segments (output from hough transformation)_

__NOTE__: _hough_lines method was modified to return a tuple (output image, set of lines) instead of only returning the output image._

Below is a description of how extrapolation works.
A straight line can be represented by __y = mx + b__ where m is the _slope_ and b 
is the _intercept on the x axis_. In our context (image from road), the left and
right lanes have +ve and -ve slope respectively. 
__Goal:__ _Deduce the end-points of lanes that shall represent each lane_.

In the input set of lines, each line is represented by (x1, y1, x2, y2)

__NOTE__: The _origin_ (0,0) in the standard coordinate axis is placed at the left bottom of the image, instead of top left. The y coordinate in the image is adjusted by _y1 = y_size - y1_, so that we can operate in the coordinate system with origin (0,0) at left bottom (and not top left). 

__Steps to form the left and right lanes__
Given a set of lines (or line segments) output from hough transformation.

a) Classify each line as being part of left lane (+ve slope) or right lane (-ve slope)

b) Find min(x), max(x), min(y) and max(y) for left anf right lanes

c) Find slope of left and right lanes using standard formula 
    slope = (y2 - y1)/(x2 - x1) 
    slope left lane = (max(y) - min(y))/(max(x) - min(x))
    slope right lane = (max(y) - min(y))/(min(x) - max(x))
    
d) Use equation _y = mx + b_ to derive the respective point where each lane meets the x axis, thus giving the starting points for each lane on the x axis.

e) Find other end-point for each lane:
    Left lane: max(x), max(y)
    Rightlane: min(x), max(y)    
    
f) Since we changed the coordinate axis in the beginning, translate the above back into the original image axis (where origin is top left)

g) Return the starting and ending points for each lane. 


### 2. Identify potential shortcomings with your current pipeline

Potential shortcomings:

a) __Parallel lanes__: the left and right lanes have an identical slope (both are (almost)parallel to each other). The individual coordinates (x1, y1, x2, y2) can be used to classify the line segment
    
b) __Curvy lanes__: Here the lane represents a higher degree polynomial instead of a straight line. The submitted solution needs ot be generalized to fit curves and not just lines. 

c) __Fluctuating end-points__: The left and right lanes for an image are computed  
    independently without taking into account the results from previous image. 
    This results in slight shift in the left and right lanes that gives a wobbling 
    effect to each lane when observed in a video. This can be fixed to produce 
    smooth transitions and moreover change the left and right lanes if the delta
    exceeds a threshold.       


### 3. Suggest possible improvements to your pipeline

Fixing the shortcomings in Section (2) would be possible improvements in the pipeline