import unittest
import measure_width as mw

import shapely
import cv2
import numpy as np
import math
import imutils
import fiona
import sys
import pandas as pd
sys.path.append(r'D:\Code\StreetView\gsv_pano')
sys.path.append(r'E:\USC_OneDrive\OneDrive - University of South Carolina\StreetView\gsv_pano')

from pano import GSV_pano
from PIL import Image
import time
import os
import glob
import multiprocessing as mp
# from label_centerlines import get_centerline
from shapely.geometry import Point, Polygon, mapping, LineString, MultiLineString
from shapely import speedups
speedups.disable()
# stackoverflow.com/questions/62075847/using-qgis-and-shaply-error-geosgeom-createlinearring-r-returned-a-null-pointer


class MyTestCase(unittest.TestCase):


    # def test_check_touched(self):
    #     # img_file = r'D:\Research\sidewalk_wheelchair\DC_DOMs\XzB9K8BHqMpZVKZR-E9MBw_DOM_0.05.tif'
    #     img_file = r'D:\Research\sidewalk_wheelchair\DC_DOMs\zhJf6faax0FFmR67jKzoKA_DOM_0.05.tif'
    #     img_pil = Image.open(img_file)
    #     img_np = np.array(np.array(img_pil))
    #     img_cv = cv2.merge([img_np])
    #     col = 49
    #     row = 203
    #     # col = 192   # False
    #     # row = 137
    #     category_list = [58, 255]  # car: 58, other: 255
    #     mask_np = mw.create_mask(img_np, category_list)
    #     is_touched = mw.check_touched(col, row, mask_np)
    #     self.assertEqual(is_touched, True)

    # def test_create_mask(self):
    #     img_file = r'D:\Research\sidewalk_wheelchair\DC_DOMs\XzB9K8BHqMpZVKZR-E9MBw_DOM_0.05.tif'
    #     img_pil = Image.open(img_file)
    #     img_np = np.array(np.array(img_pil))
    #     img_cv = cv2.merge([img_np])
    #     category_list = [58, 255]  # car: 58, other: 255
    #     img_file_catetory_sum = 96284
    #     mask_np = mw.create_mask(img_np, category_list)
    #     img_mask = mask_np.astype(np.uint8)
    #     img_mask = np.where(img_mask > 0, 255, 0)
    #     # cv2.imshow("mask", cv2.merge([img_mask.astype(np.uint8)]))
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()
    #     self.assertEqual(mask_np.sum(), img_file_catetory_sum)

    # def test_get_forward_backward_pano(self):
    #     json_file = r'D:\Research\sidewalk_wheelchair\json\Ww21SnxYVOMQMveHrmZFQw.json'
    #     pano = GSV_pano(json_file=json_file)
    #     json_dir = os.path.dirname(json_file)
    #     # print(pano.panoId)
    #     forward_pano, backward_pano = mw.get_forward_backward_pano(pano, json_dir=json_dir)
    #     # print(forward_pano.panoId)
    #     self.assertEqual('7luda3PAd0jn7QqNppv8sw', backward_pano.panoId)
    #     self.assertEqual('6w0LpXxQ2p18n9fBDgks1w', forward_pano.panoId)

    def test_get_pano_neighbors(self):
        json_file = r'D:\Research\sidewalk_wheelchair\json\Ww21SnxYVOMQMveHrmZFQw.json'
        # pano = GSV_pano(json_file=json_file)
        # json_dir = os.path.dirname(json_file)
        # print(pano.panoId)
        forward_pano, backward_pano = \
            mw.get_pano_neighbors(json_file=json_file, neighbor_order=1)

        # print(forward_pano.panoId)
        self.assertEqual('7luda3PAd0jn7QqNppv8sw', backward_pano.panoId)
        self.assertEqual('6w0LpXxQ2p18n9fBDgks1w', forward_pano.panoId)

        forward_pano, backward_pano, forward_pano_forward, backward_pano_backward = \
            mw.get_pano_neighbors(json_file=json_file, neighbor_order=2)
        # print(forward_pano.panoId)
        # self.assertEqual('7luda3PAd0jn7QqNppv8sw', backward_pano.panoId)
        # self.assertEqual('6w0LpXxQ2p18n9fBDgks1w', forward_pano.panoId)

        # print(forward_pano.panoId)
        self.assertEqual('7luda3PAd0jn7QqNppv8sw', backward_pano.panoId)
        self.assertEqual('6w0LpXxQ2p18n9fBDgks1w', forward_pano.panoId)
        self.assertEqual('DwcbiZpc7-1pSUEA-t3gWg', forward_pano_forward.panoId)
        self.assertEqual('rtCfertMVLeOdcZlTG9Mng', backward_pano_backward.panoId)

if __name__ == '__main__':
    unittest.main()