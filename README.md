# A GUI tool for manual parameters choice of Canny and HoughLinesP OpenCV functions

This is a slightly modified version of 
https://github.com/maunesh/opencv-gui-helper-tool
and is made for the first project of the Udacity Autonomous Car ND
https://github.com/udacity/CarND-LaneLines-P1

I have added a window for the output of the OpenCV HoughLinesP function.
The output directory **output_images** must be created in the project directory.
The using of the tool is as usually:

`python3 find_edges.py <file name>`

[Example output picture](highway-hough.png)

# Version 2
for Advanced Lane Finder

`python3 find_edges2.py <file name>`

Additional hyperparameters:
- HLS color space tresholds
- Color choice
- Left, right, top, bottom margins
- Perspective transform matrix

