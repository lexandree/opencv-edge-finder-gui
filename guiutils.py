import cv2
import numpy as np


class EdgeFinder:
    def __init__(   self, image, filter_size=7, threshold1=200, threshold2=230, 
                    rho = 4, theta = np.pi/180, min_votes = 25, min_line_length = 40, max_line_gap = 40, 
                    cv_color=cv2.COLOR_RGB2GRAY, color=[255, 0, 0], thickness=4, mean_rate = 0.9, 
                    pos_intersect_ratio=10, neg_intersect_ratio=10):
                    
        self.orig_image = image
        self._cv_color = cv_color #cv2.COLOR_BGR2GRAY or cv2.COLOR_RGB2GRAY
        self.image = self._grayscale(image)
        self._filter_size = filter_size
        self._threshold1 = threshold1
        self._threshold2 = threshold2

        self._rho = rho # distance resolution in pixels of the Hough grid
        self._theta = theta # angular resolution in radians of the Hough grid
        self._min_votes = min_votes     # minimum number of votes (intersections in Hough grid cell)
        self._min_line_length = min_line_length #minimum number of pixels making up a line
        self._max_line_gap = max_line_gap    # maximum gap in pixels between connectable line segments

        self._color = color
        self._thickness = thickness
        self._pos_intersect_ratio = pos_intersect_ratio
        self._neg_intersect_ratio = neg_intersect_ratio
        
        def onchangeThreshold1(pos):
            self._threshold1 = pos
            self._render()

        def onchangeThreshold2(pos):
            self._threshold2 = pos
            self._render()

        def onchangeFilterSize(pos):
            self._filter_size = pos
            self._filter_size += (self._filter_size + 1) % 2
            self._render()

        def onchangeRho(pos):
            self._rho = np.max([pos, 1])
            self._render()

        def onchangeTheta(pos):
            self._theta = np.max([pos*np.pi/180.0, np.pi/180.0])
            self._render()

        def onchangeMinVotes(pos):
            self._min_votes = pos
            self._render()

        def onchangeMinLineLength(pos):
            self._min_line_length = pos
            self._render()

        def onchangeMaxLineGap(pos):
            self._max_line_gap = pos
            self._render()

        cv2.namedWindow('edges')
        cv2.namedWindow('hough')

        cv2.createTrackbar('threshold1', 'edges', self._threshold1, 255, onchangeThreshold1)
        cv2.createTrackbar('threshold2', 'edges', self._threshold2, 255, onchangeThreshold2)
        cv2.createTrackbar('filter_size', 'edges', self._filter_size, 20, onchangeFilterSize)
        
        cv2.createTrackbar('rho', 'hough', self._rho, 100, onchangeRho)
        cv2.createTrackbar('theta', 'hough', 1, 12, onchangeTheta)
        cv2.createTrackbar('min_votes', 'hough', self._min_votes, 100, onchangeMinVotes)
        cv2.createTrackbar('min lenght', 'hough', self._min_line_length, 300, onchangeMinLineLength)
        cv2.createTrackbar('max gap', 'hough', self._max_line_gap, 200, onchangeMaxLineGap)

        self._render()

        print("Adjust the parameters as desired.  Hit any key to close.")

        cv2.waitKey(0)

        cv2.destroyWindow('edges')
        cv2.destroyWindow('smoothed')
        cv2.destroyWindow('hough')

    def _grayscale(self, img):
        return cv2.cvtColor(img, self._cv_color)

    def _draw_lines(self, img, lines):

        for line in lines:
            for x1,y1,x2,y2 in line:
                if abs(x2-x1) < 0.01:
                    continue
                slope = round((y2-y1)/(x2-x1),1)
                if np.isnan(slope) or np.isinf(slope):
                    continue
                intercept = int(round(y1 - slope*x1))
                if slope > 0 and ((x1+x2<img.shape[1]) or(intercept<img.shape[0]//self._pos_intersect_ratio)):
                    cv2.line(img, (x1, y1), (x2, y2), [0, 255, 255], self._thickness)
                elif slope < 0 and ((x1+x2>img.shape[1]) or(intercept<(img.shape[0] + img.shape[0]//self._neg_intersect_ratio))):
                    cv2.line(img, (x1, y1), (x2, y2), [0, 255, 255], self._thickness)
                elif abs(slope) < 0.5 or abs(slope) > 0.85:
                    cv2.line(img, (x1, y1), (x2, y2), [0, 255, 255], self._thickness)
                else:
                    cv2.line(img, (x1, y1), (x2, y2), self._color, self._thickness)
                    

    def _hough_lines(self, img):
        """
        `img` should be the output of a Canny transform.
        Returns an image with hough lines drawn.
        """
        lines = cv2.HoughLinesP(img, self._rho, self._theta, self._min_votes, np.array([]), 
            minLineLength=self._min_line_length, maxLineGap=self._max_line_gap)
            
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        self._draw_lines(line_img, lines)
        return line_img

    def threshold1(self):
        return self._threshold1

    def threshold2(self):
        return self._threshold2

    def filterSize(self):
        return self._filter_size
        
    def rho(self):
        return self._rho
        
    def theta(self):
        return self._theta
        
    def min_votes(self):
        return self._min_votes

    def min_line_length(self):
        return self._min_line_length

    def max_line_gap(self):
        return self._max_line_gap
        
    def edgeImage(self):
        return self._edge_img

    def smoothedImage(self):
        return self._smoothed_img

    def houghImage(self):
        return self._hough_img

    def _render(self):
        self._smoothed_img = cv2.GaussianBlur(self.image, (self._filter_size, self._filter_size), sigmaX=0, sigmaY=0)
        self._edge_img = cv2.Canny(self._smoothed_img, self._threshold1, self._threshold2)
        self._hough_img = self._hough_lines(self._edge_img)
        # Draw the lines on the edge image
        self._hough_img = cv2.addWeighted(self._hough_img, 0.8, self.orig_image, 1., 0.) #[:,:,0]
        
        cv2.imshow('smoothed', self._smoothed_img)
        cv2.imshow('edges', self._edge_img)
        cv2.imshow('hough', self._hough_img)
