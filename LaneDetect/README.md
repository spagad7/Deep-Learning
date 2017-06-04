# **Finding Lane Lines on the Road** 


The goal of this project is to detect and annotate lane markes on the road.


[//]: # (Image References)

[image1]: ./Images/imgGray.jpg "Grayscale Image"
[image2]: ./Images/imgBlur.jpg "Blurred Image"
[image3]: ./Images/imgEdge.jpg "Edge Image"
[image4]: ./Images/imgMasked.jpg "Masked Image"
[image5]: ./Images/imgLines.jpg "Lane Markers Annotated Image"

---

### Reflection

### 1. Steps

The pipeline I used for this project consists of 5 main steps:

a. Extract frame from video and convert it to grayscale image.
![alt Grayscale Image][image1]

b. The grayscale image is smoothend using gaussian kernel. Smoothing reduces noise in image.
![alt Blurred Image][image2]

c. Extract edges in an image using Canny edge detector.
![alt Edge Image][image3]

d. Apply region of interest mask to remove all the edges outside the region of interest.
![alt Masked Image][image4]

e. Apply Hough transform to find straight lines in the edge image
![alt Lane Markers Annotated Image][image5]

### Functions

1.	main:
	This function is the starting point of the program. It extracts frames from video and passes the frames to processImage function.

2.	processImage:
	This function converts input image to grayscale, smoothens it, extracts edges and masks the image to get edges in region of interest. The processed image is passed to drawLines function.

3.  drawLines:
	This fuction applies Hough transform on the masked edge image to find all the straight lines in the image. After finding the straight lines, the slope of each line is calculated and categorized as positive or negative slope. Positive slope corresponds to left line and negetive slope to right line. Out of all the calculated slopes for the left line, one of them is chosen to draw extrapolated single line corresponding to left lane marker. Extrapolation is done using the slope intercept formula: y = mx + c



### 2. Shortcomings

One of the potential shortcoming for this program is that it cant detect very curved lanes. This is because the progam tries to find and fit straight lines in a given image; a atraight line can't be fit on a curved lane. Another shortcoming is that edge detection fails when lane lines are not distinctly marked or when the contrast of color between lane marker and road is not high enough. And since hough transform is contingent on detection of straight lines in edges, it fails to find straight lines in such images.


### 3. Possible improvements

A possible improvement for this project is: In extrapolating lines, only one of the slope(first one in the list) is considered. A better solution is to consider the most common or average slope in the list so that the lines aren't shaky.