import cv2
import numpy as np
import pickle


class EdgeFinder:
    def __init__(   self, image, filter_size=7, threshold1=200, threshold2=230, 
                    rho = 4, theta = np.pi/180, min_votes = 25, min_line_length = 40, max_line_gap = 40, 
                    s_thresh_lower=57, s_thresh_upper=205, sx_thresh_lower=26, sx_thresh_upper=62,
                    cv_color=cv2.COLOR_RGB2GRAY, color=[255, 0, 0], thickness=4, mean_rate = 0.9, 
                    pos_intersect_ratio=10, neg_intersect_ratio=10, draw_all=False, dist_mtx_file='undist.p', M_transform_file='m_trans.p'):

        self._orig_image = image.copy()
        self._orig_image[-100:,:,:] = 0 # hide motor hood on the picture
        self._undist = False
        self._const_image = self._orig_image.copy()
        self._filter_size = filter_size
        self._threshold1 = threshold1
        self._threshold2 = threshold2

        self._rho = rho # distance resolution in pixels of the Hough grid
        self._theta = theta # angular resolution in radians of the Hough grid
        self._min_votes = min_votes     # minimum number of votes (intersections in Hough grid cell)
        self._min_line_length = min_line_length #minimum number of pixels making up a line
        self._max_line_gap = max_line_gap    # maximum gap in pixels between connectable line segments
        
        self._s_thresh_lower = s_thresh_lower
        self._s_thresh_upper = s_thresh_upper
        self._sx_thresh_lower = sx_thresh_lower
        self._sx_thresh_upper = sx_thresh_upper
        self._r_color = 0

        self._cv_color = cv_color #cv2.COLOR_BGR2GRAY or cv2.COLOR_RGB2GRAY
        self._color = color # color for detected and approximated lanes
        self._thickness = thickness # hough lines thickness
        self._mean_rate = mean_rate  # means*mean_rate + current_values*(1 - mean_rate)
        self._draw_all = draw_all   # switch all hough lines besides approximated on and off
        self._pos_intersect_ratio = pos_intersect_ratio # pos_intersect_treshold = - img.heght * pos_intersect_ratio
        self._neg_intersect_ratio = neg_intersect_ratio # neg_intersect_treshold = img.heght + img.heght * pos_intersect_ratio
        self._FLOAT_TRESHOLD = 0.01
        self._mtx = None
        self._dist = None
        self._M = None
        self._dist_mtx_file = dist_mtx_file
        self._M_transform_file = M_transform_file
        
        self._means = np.zeros((2,2), dtype=float) # slope_pos, slope_neg, intersect_pos, intersect_neg
        self.read_dist_mtx(self._dist_mtx_file)
        self.read_M_transform(self._M_transform_file)
        self._orig_image = self.undistort_img(self._orig_image)
        self._means_y0 = self._orig_image.shape[0]//2 + self._orig_image.shape[0]//9

        self._offset = 300
        self._alt_m = False
        self._alt_pipeline = False
        self._offset_y = 0 # hide motor hood on the picture 
        # polynomials
        self._left_fit = []
        self._right_fit = []
        self._left_fit_real = []
        self._right_fit_real = []
        
        self.ym_per_pix = 55/self._orig_image.shape[0] # meters per pixel in y dimension
        self.xm_per_pix = 3.7/(self._orig_image.shape[1]-200) # meters per pixel in x dimension - offset
        
        self.curv_left, self.curv_right = 32000, 32000

        def onchangeR_color(pos):
            self._r_color = pos
            self._render()

        def onchangeS_thresh_lower(pos):
            self._s_thresh_lower = pos
            self._render()

        def onchangeS_thresh_upper(pos):
            self._s_thresh_upper = pos
            self._render()

        def onchangeSx_thresh_lower(pos):
            self._sx_thresh_lower = pos
            self._render()

        def onchangeSx_thresh_upper(pos):
            self._sx_thresh_upper = pos
            self._render()

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
        cv2.namedWindow('pipeline')

        cv2.createTrackbar('threshold1', 'edges', self._threshold1, 255, onchangeThreshold1)
        cv2.createTrackbar('threshold2', 'edges', self._threshold2, 255, onchangeThreshold2)
        cv2.createTrackbar('filter_size', 'edges', self._filter_size, 20, onchangeFilterSize)
        
        cv2.createTrackbar('rho', 'hough', self._rho, 100, onchangeRho)
        cv2.createTrackbar('theta', 'hough', 1, 12, onchangeTheta)
        cv2.createTrackbar('min_votes', 'hough', self._min_votes, 100, onchangeMinVotes)
        cv2.createTrackbar('min lenght', 'hough', self._min_line_length, 300, onchangeMinLineLength)
        cv2.createTrackbar('max gap', 'hough', self._max_line_gap, 200, onchangeMaxLineGap)

        cv2.createTrackbar('s_thresh_lower', 'pipeline', self._s_thresh_lower, 255, onchangeS_thresh_lower)
        cv2.createTrackbar('s_thresh_upper', 'pipeline', self._s_thresh_upper, 255, onchangeS_thresh_upper)
        cv2.createTrackbar('sx_thresh_lower', 'pipeline', self._sx_thresh_lower, 255, onchangeSx_thresh_lower)
        cv2.createTrackbar('sx_thresh_upper', 'pipeline', self._sx_thresh_upper, 255, onchangeSx_thresh_upper)
        cv2.createTrackbar('r_color', 'pipeline', self._r_color, 2, onchangeR_color)

        self._render()

        print("Adjust the parameters as desired.  Hit q to close.")

        #cv2.waitKey(0)
        self._do_loop()

        cv2.destroyWindow('edges')
        #cv2.destroyWindow('smoothed')
        cv2.destroyWindow('hough')
        cv2.destroyWindow('pipeline')

    def _grayscale(self, img):
        return cv2.cvtColor(img, self._cv_color)
        
    def get_radius(self, poly, x):
        
        res = np.power((1+np.square(2*poly[0]*x+poly[1])), 3/2)/np.abs(2*poly[0])
        return res
        
    def _img_overlay_color(self, back, fore, x, y, opaque=True):
        # for small image use cv.resize in main programm:
        # overlay = cv2.resize(self._edge_img, None, fx=0.2, fy=0.2, interpolation = cv2.INTER_AREA)

        if len(fore.shape)==2:
            overlay = cv2.cvtColor(fore, cv2.COLOR_GRAY2RGB)
        else:
            overlay = fore
        rows, cols, channels = overlay.shape  
        if opaque:
            overlay_copy = overlay
        else:
            trans_indices = overlay[...,2] != 0 # Where not transparent
            overlay_copy = back[y:y+rows, x:x+cols] 
            overlay_copy[trans_indices] = overlay[trans_indices]
            
        out_img = back.copy()
        out_img[y:y+rows, x:x+cols] = overlay_copy
        return out_img

    def read_dist_mtx(self, file_name):
        with open(file_name, 'rb') as config_dictionary_file:
            dist_pickle = pickle.load(config_dictionary_file)
        self._mtx = dist_pickle["mtx"]
        self._dist = dist_pickle["dist"]

    def read_M_transform(self, file_name):
        with open(file_name, 'rb') as m_file:
            self._M = pickle.load(m_file)
        #self._M = dict_pickle["M"]

    def transform(self, img):
        '''
        perspective transformation of img with self._M matrix
        '''
        img_size = (img.shape[1], img.shape[0])
        #self._transformed = cv2.warpPerspective(img, self._M, img_size)
        return cv2.warpPerspective(img, self._M, img_size)
        
    def switch_img(self):
        if self._undist:
            self._undist = False
            return self._const_image
        else:
            self._undist = True
            return cv2.undistort(self._const_image, self._mtx, self._dist, None, self._mtx)

    def undistort_img(self, img):
        return cv2.undistort(img, self._mtx, self._dist, None, self._mtx)

    def gaussian_blur(self, img):
        """Applies a Gaussian Noise kernel"""
        return cv2.GaussianBlur(img, (self._filter_size, self._filter_size), 0)

    def canny(self, img):
        """Applies the Canny transform"""
        return cv2.Canny(img, self._threshold1, self._threshold2)

    def region_of_interest(self, img):
        """
        Applies an image mask.
        
        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        `vertices` should be a numpy array of integer points.
        """
        #defining a blank mask to start with
        imshape = img.shape
        vertices = np.array([[(0,imshape[0]),(0, imshape[0]/2+imshape[0]*0.1), 
                                (imshape[1], imshape[0]/2+imshape[0]*0.1),
                                (imshape[1],imshape[0])]], 
                                dtype=np.int32)
        mask = np.zeros_like(img)   
        
        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
            
        #filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        
        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def _draw_means(self, img, lines):
        """Computes slope and intersect means, approximates and drows lines"""
        means = np.zeros((2,2), dtype=float)
        counts = np.zeros(2, dtype=int)
        
        for line in lines:
            x1,y1,x2,y2,sl,intcpt = line
            means[int(sl < 0)] += sl,intcpt
            counts[int(sl < 0)] += 1
        
        for i in range(2):
            if counts[i]>0:
                means[i] /= counts[i]
        if abs(self._means.all()) < self._FLOAT_TRESHOLD:
            self._means = means
            #print('means init')
        for i in range(2):
            if counts[i]>0:
                self._means[i] = self._means[i]*self._mean_rate + means[i]*(1 - self._mean_rate)

            if np.isnan(self._means[i]).any():
                continue
            if abs(self._means[i,0])<self._FLOAT_TRESHOLD:
                continue
            y1 = self._means_y0 #img.shape[0]//2 + img.shape[0]//10
            y2 = img.shape[0]-1
            x1 = int((y1 - self._means[i,1])/self._means[i,0])
            x2 = int((y2 - self._means[i,1])/self._means[i,0])
            cv2.line(img, (x1, y1), (x2, y2), self._color, self._thickness)
        return img

    def _select_lines(self, img, lines, color=[255, 255, 0], thickness=4):
        """Removes Hough lines they are off criteria from the mean list
            can draw these lines for control"""
        _lines = []
        _short_line = []
        for line in lines:
            for x1,y1,x2,y2 in line:
                if abs(x2-x1) < 1:
                    continue
                slope = round((y2-y1)/(x2-x1),1)
                if np.isnan(slope) or np.isinf(slope):
                    continue
                intercept = int(round(y1 - slope*x1))
                if slope > 0 and ((x1+x2<img.shape[1]) or(intercept< -img.shape[0]//self._pos_intersect_ratio)): 
                    if self._draw_all:
                        cv2.line(img, (x1, y1), (x2, y2), [0, 255, 255], thickness)
                elif slope < 0 and ((x1+x2>img.shape[1]) or(intercept<(img.shape[0] + img.shape[0]//self._neg_intersect_ratio))):
                    if self._draw_all:
                        cv2.line(img, (x1, y1), (x2, y2), [0, 255, 255], thickness)
                elif abs(slope) < 0.5 or abs(slope) > 0.85:
                    if self._draw_all:
                        cv2.line(img, (x1, y1), (x2, y2), [0, 255, 255], thickness)
                else:
                    if self._draw_all:
                        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
                    _short_line = [round(x1), round(y1), round(x2), round(y2), slope, intercept]
                    _lines.append(_short_line)
        return _lines            
                    

    def _hough_lines(self, img):
        """
        `img` should be the output of a Canny transform.
        Returns an image with hough lines drawn.
        """
        lines = cv2.HoughLinesP(img, self._rho, self._theta, self._min_votes, np.array([]), 
            minLineLength=self._min_line_length, maxLineGap=self._max_line_gap)
            
        #line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        #self._draw_lines(line_img, lines)
        return lines

    def find_lane_pixels(self, binary_warped):
        y_base_position = binary_warped.shape[0] - self._offset_y
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[y_base_position - binary_warped.shape[0]//5:y_base_position,:], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(y_base_position//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = y_base_position - (window+1)*window_height
            win_y_high = y_base_position - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2) 
            
            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img

    def fit_poly(self, img_shape, leftx, lefty, rightx, righty):
        # Generate y values for plotting
        ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
        right_fitx = 1*ploty**2 + 1*ploty
        left_fitx = right_fitx
        if len(leftx):
            left_fit = np.polyfit(lefty, leftx, 2)
            left_fit_real = np.polyfit(lefty*self.ym_per_pix, leftx*self.xm_per_pix, 2)
            if len(self._left_fit)==0:
                self._left_fit = left_fit
                self._left_fit_real = left_fit_real
            else:
                for i in range(len(self._left_fit)):
                    self._left_fit[i] = self._left_fit[i]*self._mean_rate + left_fit[i]*(1 - self._mean_rate)
                    self._left_fit_real[i] = self._left_fit_real[i]*self._mean_rate + left_fit_real[i]*(1 - self._mean_rate)
            
            if len(self._left_fit):
                #try:
                left_fitx = self._left_fit[0]*ploty**2 + self._left_fit[1]*ploty + self._left_fit[2]
                #except TypeError:
                    # Avoids an error if `left` and `right_fit` are still none or incorrect
                #    print('The function failed to fit the left line!')
                #    left_fitx = 1*ploty**2 + 1*ploty
        if len(rightx):
            right_fit = np.polyfit(righty, rightx, 2)
            right_fit_real = np.polyfit(righty*self.ym_per_pix, rightx*self.xm_per_pix, 2)
            if len(self._right_fit)==0:
                self._right_fit = right_fit
                self._right_fit_real = right_fit_real
            else:
                for i in range(len(self._right_fit)):
                    self._right_fit[i] = self._right_fit[i]*self._mean_rate + right_fit[i]*(1 - self._mean_rate)
                    self._right_fit_real[i] = self._right_fit_real[i]*self._mean_rate + right_fit_real[i]*(1 - self._mean_rate)

            if len(self._right_fit):
                #try:
                right_fitx = self._right_fit[0]*ploty**2 + self._right_fit[1]*ploty + self._right_fit[2]
                #except TypeError:
                    # Avoids an error if `left` and `right_fit` are still none or incorrect
                #    print('The function failed to fit the right line!')
                #    right_fitx = 1*ploty**2 + 1*ploty
        
        #print(left_fitx.shape)
        #print(left_fitx.shape)
        left_fitx[left_fitx>=img_shape[1]] = img_shape[1] - 1
        right_fitx[right_fitx>=img_shape[1]] = img_shape[1] - 1
        left_fitx[left_fitx < 0] = 0
        right_fitx[right_fitx < 0] = 0

        return np.int16(left_fitx), np.int16(right_fitx), np.int16(ploty)

    def fit_polynomial(self, binary_warped):
        # Find our lane pixels first
        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(binary_warped)

        # Fit a second order polynomial to each using `np.polyfit`
        left_fitx, right_fitx, ploty = self.fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

        ## Visualization ##
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        # Plots the left and right polynomials on the lane lines
        out_img[ploty, left_fitx] = [255, 255, 0]
        out_img[ploty, right_fitx] = [255, 255, 0]

        #plt.plot(left_fitx, ploty, color='yellow')
        #plt.plot(right_fitx, ploty, color='yellow')

        return out_img

    def pipeline(self, inp_img): # , s_thresh=(170, 255), sx_thresh=(20, 100)
        s_thresh = (self._s_thresh_lower, self._s_thresh_upper)
        sx_thresh = (self._sx_thresh_lower, self._sx_thresh_upper)
        img = np.copy(inp_img)
        # Convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

        # Sobel R
        sobelr = cv2.Sobel(img[:,:,self._r_color], cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelr = np.absolute(sobelr) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobelr = np.uint8(255*abs_sobelr/np.max(abs_sobelr))
        # Threshold R gradient
        srbinary = np.zeros_like(scaled_sobelr)
        srbinary[(scaled_sobelr >= sx_thresh[0]) & (scaled_sobelr <= sx_thresh[1])] = 1
        
        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        # Stack each channel
        #color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
        color_binary = np.dstack((srbinary, sxbinary, s_binary)) * 255
        return color_binary

    def search_around_poly(self, binary_warped):
        # HYPERPARAMETER
        # Choose the width of the margin around the previous polynomial to search
        # The quiz grader expects 100 here, but feel free to tune on your own!
        margin = 100

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        ### TO-DO: Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        ### Hint: consider the window areas for the similarly named variables ###
        ### in the previous quiz, but change the windows to our new search area ###
        left_lane_inds = ((nonzerox>(self._left_fit[0]*(nonzeroy**2)+self._left_fit[1]*nonzeroy+
                            self._left_fit[2]-margin))&
                            (nonzerox<(self._left_fit[0]*(nonzeroy**2)+self._left_fit[1]*nonzeroy+
                            self._left_fit[2]+margin)))
        right_lane_inds = ((nonzerox>(self._right_fit[0]*nonzeroy**2+self._right_fit[1]*nonzeroy+
                            self._right_fit[2]-margin))&
                            (nonzerox<(self._right_fit[0]*nonzeroy**2+self._right_fit[1]*nonzeroy+
                            self._right_fit[2]+margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit new polynomials
        left_fitx, right_fitx, ploty = self.fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
        
        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                  ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                  ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        
        # Plots the left and right polynomials on the lane lines
        result[ploty, left_fitx] = [255, 255, 0]
        result[ploty, right_fitx] = [255, 255, 0]

        # Plot the polynomial lines onto the image
        #plt.plot(left_fitx, ploty, color='yellow')
        #plt.plot(right_fitx, ploty, color='yellow')
        ## End visualization steps ##
        
        return result


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

    def draw_lanes(self, img):
        """Pipeline for finding lanes"""
        tmp_img = self.grayscale(image)

        tmp_img = self.gaussian_blur(img)
        tmp_img = self.canny(tmp_img)
        tmp_img = self.region_of_interest(tmp_img)
        _lines = self._hough_lines(tmp_img)
        
        if _lines is None: return None
        
        lane_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        _lines = self._select_lines(lane_img, _lines)
        tmp_img = self._draw_means(lane_img, _lines)

        return cv2.addWeighted(tmp_img, 0.8, img, 1., 0.) #[:,:,0]

    def measure_curvature(self, pict_shape):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        # Define conversions in x and y from pixels space to meters
        # ploty = np.linspace(0, pict_shape[0]-1, num=pict_shape[0])# to cover same y-range as image
        # ploty *= ym_per_pix
        # left_poly = np.polyfit(ploty, left_x*xm_per_pix, 2)
        # right_poly = np.polyfit(ploty, right_x*xm_per_pix, 2)
        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image

        y_eval = pict_shape[0] * self.ym_per_pix
        left_curverad = self.get_radius(self._left_fit_real, y_eval)
        right_curverad = self.get_radius(self._right_fit_real, y_eval)
        return left_curverad, right_curverad
        
    def trans_back(self, warped, orig_img):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = warp_zero #np.dstack((warp_zero, warp_zero, warp_zero))
        
        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
        left_fitx = self._left_fit[0]*ploty**2 + self._left_fit[1]*ploty + self._left_fit[2]
        right_fitx = self._right_fit[0]*ploty**2 + self._right_fit[1]*ploty + self._right_fit[2]
        left_fitx[left_fitx>=warped.shape[1]] = warped.shape[1] - 1
        right_fitx[right_fitx>=warped.shape[1]] = warped.shape[1] - 1
        left_fitx[left_fitx < 0] = 0
        right_fitx[right_fitx < 0] = 0

       # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self._M, (warped.shape[1], warped.shape[0]),
             flags=cv2.WARP_INVERSE_MAP) 
        #cv2.CV_INTER_LINEAR+cv2.CV_WARP_FILL_OUTLIERS+cv2.WARP_INVERSE_MAP
        # Combine the result with the original image
        return cv2.addWeighted(orig_img, 1, newwarp, 0.3, 0)
    
    def get_position(self):
    
        self._hough_img
        ploty = self._hough_img.shape[0]
        left_line = self._left_fit[0]*ploty**2 + self._left_fit[1]*ploty + self._left_fit[2]
        right_line = self._right_fit[0]*ploty**2 + self._right_fit[1]*ploty + self._right_fit[2]
        current_pos = (ploty - right_line + left_line)//2 * self.xm_per_pix
        return current_pos

    def _render(self):
        # if not self._M is None:
            # self.transform(self._orig_image)
            # _gray_img = self._grayscale(self._transformed)
        # else:
            # _gray_img = self._grayscale(self._orig_image)
        _gray_img = self._grayscale(self._orig_image)

        self._smoothed_img = self.gaussian_blur(_gray_img)
        self._edge_img = self.canny(self._smoothed_img)
        tmp_img = self.region_of_interest(self._edge_img)
        _lines = self._hough_lines(tmp_img)
        
        lane_img = np.zeros((self._orig_image.shape[0], self._orig_image.shape[1], 3), dtype=np.uint8)
        _lines = self._select_lines(lane_img, _lines)
        tmp_img = self._draw_means(lane_img, _lines)
        self._hough_img = cv2.addWeighted(tmp_img, 0.8, self._orig_image, 1., 0.) #[:,:,0]
        
        overlay = cv2.resize(self._edge_img,None,fx=0.2, fy=0.2, interpolation = cv2.INTER_AREA)
        self._hough_img = self._img_overlay_color(self._hough_img, overlay, 5, 5)  
        diff = np.abs(self.curv_left - self.curv_right)/(self.curv_left + self.curv_right)
        if (self._M is None) or diff > 0.09:
            pipeline_img = self.pipeline(self._orig_image)
            polynomial_img = self.fit_polynomial(self._grayscale(pipeline_img))
        else:
            self._transformed = self.transform(self._orig_image)
            if self._alt_pipeline:
                pipeline_img = self.transform(self._edge_img)
                polynomial_img = self.fit_polynomial(pipeline_img)
            else:
                pipeline_img = self.pipeline(self._transformed)
                polynomial_img = self.fit_polynomial(self._grayscale(pipeline_img))
            #print('transform polynomial_img')
            self._transformed = self.transform(self._orig_image)
            #cv2.imshow('transformed', self._transformed)
            overlay = cv2.resize(self._transformed, None, fx=0.2, fy=0.2, interpolation = cv2.INTER_AREA)
            self._hough_img = self._img_overlay_color(self._hough_img, overlay, 6+self._edge_img.shape[1]//5, 5)        
        if self._alt_pipeline:
            around = self.search_around_poly(pipeline_img)
        else:
            around = self.search_around_poly(self._grayscale(pipeline_img))
        #around = self.search_around_poly(self._grayscale(self._transformed))
        overlay = cv2.resize(around, None, fx=0.2, fy=0.2, interpolation = cv2.INTER_AREA)
        self._hough_img = self._img_overlay_color(self._hough_img, overlay, (6+self._edge_img.shape[1]//5)*2, 5)        

        self._hough_img = self.trans_back(self._transformed, self._hough_img)
        cv2.imshow('pipeline', polynomial_img)
        #cv2.imshow('smoothed', self._smoothed_img)
        cv2.imshow('edges', self._edge_img)
        
        curv_left, curv_right = np.int32(self.measure_curvature(pipeline_img.shape))
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        position = self.get_position()
        if position<0:
            str_pos = f"Vehicle is {np.abs(position):.2f}m left of center"
        else:
            str_pos = f"Vehicle is {position:.2f}m right of center"
        cv2.putText(self._hough_img, f'offset x(a-d):{self._offset},offset y(w-x):{self._offset_y},alt.M(f):{self._alt_m},left {curv_left},right {curv_right}, alt.p(g):{self._alt_pipeline} ', 
                    (10,30), font, fontScale, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(self._hough_img, f'Radius of Curvature {np.int_(curv_left+curv_right)//2}(m)', 
                    (10,65), font, fontScale, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(self._hough_img, str_pos, 
                    (10,100), font, fontScale, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow('hough', self._hough_img)
        
    def create_M(self):
        offset = self._offset
        y1 = self._means_y0 #img.shape[0]//2 + img.shape[0]//10
        y2 = self._orig_image.shape[0] - 50 # offset 
        x1 = int((y1 - self._means[1,1])/self._means[1,0])
        x2 = int((y1 - self._means[0,1])/self._means[0,0]) #+ 10
        x3 = int((y2 - self._means[0,1])/self._means[0,0])
        x4 = int((y2 - self._means[1,1])/self._means[1,0])
        #print(np.int16([[x1,y1],[x2,y1],[x3,y2],[x4,y2]]))
        corners = np.float32([[x1,y1],[x2,y1],[x3,y2],[x4,y2]])
        # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
        #Note: you could pick any four of the detected corners 
        # as long as those four corners define a rectangle
        #One especially smart way to do this would be to use four well-chosen
        # corners that were automatically detected during the undistortion steps
        #We recommend using the automatic detection of corners in your code
        # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
        if self._alt_m:
            dst = np.float32([[offset, y1],
                [self._orig_image.shape[1] - offset, y1],
                [self._orig_image.shape[1] - offset, y2],
                [offset, y2]])
        else:
            dst = np.float32([[offset, 0],
                [self._orig_image.shape[1] - offset, 0],
                [self._orig_image.shape[1] - offset, self._orig_image.shape[0] - 0],
                [offset, self._orig_image.shape[0] - 0]])
        #print(np.int16(dst))
        # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
        self._M = cv2.getPerspectiveTransform(corners, dst)
        #M_pickle = {}
        #M_pickle['M'] = self._M
        with open(self._M_transform_file, 'wb') as m_file:
            pickle.dump(self._M, m_file)
            print('transform matrix saved to {}'.format(self._M_transform_file))
        
        #self._M = self.read_M_transform(self._M_transform_file)
        return self._M

    def _do_loop(self):
        while True:
            k = cv2.waitKey(10)
            # Press q to break
            if k == ord('q'):
                break
            # press a to increase self._offset by 1
            if k == ord('a'):
                self._offset += 1
                if self._offset >=300:
                    self._offset = 300
            # press d to decrease self._offset by 1
            elif k== ord('d'):
                self._offset -= 1
                if self._offset <= 0:
                    self._offset = 0
            elif k== ord('w'):
                self._offset_y += 1
                if self._offset_y >=200:
                    self._offset_y = 200
            elif k== ord('x'):
                self._offset_y -= 1
                if self._offset_y <= 0:
                    self._offset_y = 0
            elif k== ord('m'):
                self.create_M()
            elif k== ord('s'):
                self._orig_image = self.switch_img()
            elif k== ord('f'):
                self._alt_m = not self._alt_m
            elif k== ord('g'):
                self._alt_pipeline = not self._alt_pipeline
            self._render()
        return 0

