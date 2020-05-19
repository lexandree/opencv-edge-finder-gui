"""
How to run:
python find_edges.py <image path>
"""

import argparse
import cv2
import os
import numpy as np

from guiutils_p2 import EdgeFinder


def main():
    parser = argparse.ArgumentParser(description='Visualizes the line for hough transform.')
    parser.add_argument('filename')

    args = parser.parse_args()
    print(args.filename)

    #img = cv2.imread(args.filename, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(args.filename)
    print(img.shape)
    print('current dir: ', os.getcwd())
    

    edge_finder = EdgeFinder(img, color=[0, 0, 255], filter_size=7, threshold1=200, threshold2=215, pos_intersect_ratio=20, 
                            mean_rate=0.4, draw_all=True)

    print("Edge parameters:")
    print("GaussianBlur Filter Size: %d" % edge_finder.filterSize())
    print("Threshold1: %d" % edge_finder.threshold1())
    print("Threshold2: %d" % edge_finder.threshold2())

    print("Rho: %d" % edge_finder.rho())
    print("Theta: %f" % edge_finder.theta())
    print("Min Votes: %d" % edge_finder.min_votes())
    print("min_line_length: %d" % edge_finder.min_line_length())
    print("max_line_gap: %d" % edge_finder.max_line_gap())


    (head, tail) = os.path.split(args.filename)
    print('head & tail:', head, tail)
    (root, ext) = os.path.splitext(tail)
    print('root & ext:', root, ext)

    smoothed_filename = os.path.join("output_images", root + "-smoothed" + ext)
    edge_filename = os.path.join("output_images", root + "-edges" + ext)
    hough_filename = os.path.join("output_images", root + "-hough" + ext)

    cv2.imwrite(smoothed_filename, edge_finder.smoothedImage())
    cv2.imwrite(edge_filename, edge_finder.edgeImage())
    cv2.imwrite(hough_filename, edge_finder.houghImage())

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
