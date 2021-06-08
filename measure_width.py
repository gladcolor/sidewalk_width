import shapely
import cv2
import numpy as np
import math
import imutils
import fiona
import sys
import subprocess

from tqdm import tqdm
import pandas as pd
sys.path.append(r'D:\Code\StreetView\gsv_pano')
sys.path.append(r'E:\USC_OneDrive\OneDrive - University of South Carolina\StreetView\gsv_pano')
sys.path.append(r'K:\OneDrive_USC\OneDrive - University of South Carolina\StreetView\gsv_pano')

import geopandas as gpd
from pano import GSV_pano
import utils
from numpy import linalg as LA
from PIL import Image
import time
import os
import glob
import multiprocessing as mp
import logging
import fiona
import rasterio.features

# import gdal_array
from skimage import measure
import numpy as np
import json
import os
import datetime
from shapely.geometry import Polygon

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from label_centerlines import get_centerline
from shapely.geometry import Point, Polygon, mapping, LineString, MultiLineString
from shapely import speedups
speedups.disable()
# stackoverflow.com/questions/62075847/using-qgis-and-shaply-error-geosgeom-createlinearring-r-returned-a-null-pointer

logging.basicConfig(filename='measurementing.log', level=logging.ERROR)

LINE_COUNT = 160

SIDEWALK_INTERVAL = 0.25 # meter
COVER_RATIO_MINIMUM = 0.85

def cv_img_rotate_bound(image, angle, flags=cv2.INTER_NEAREST):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), flags=flags)

def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
    '''
    rows, cols = np.where(x == 1) # .T sets Fortran order down-then-right
    run_lengths = []
    run_rows = []
    prev = -2
    for idx, b in enumerate(cols):
        if (b != prev+1):
            run_lengths.extend((b, 0))
            run_rows.extend((rows[idx],))
        # else:  #(b < prev),  new line
        #     pass
        run_lengths[-1] += 1
        prev = b
    return run_lengths, run_rows



def cal_witdh_from_list(img_list, crs_local=6847):

    # img_path = r'ZXyk9lKhL5siKJglQPqfMA_DOM_0.05.tif'
    # img_list = [img_path]
    total_cnt = len(img_list)
    start_time = time.perf_counter()
    # print("total_cnt: ", os.getpid(),  total_cnt)
    cnt = 0

    yaw_csv_file = r'H:\USC_OneDrive\OneDrive - University of South Carolina\Research\sidewalk_wheelchair\vectors\near_compassA.csv'
    # yaw_csv_file = r'E:\USC_OneDrive\OneDrive - University of South Carolina\Research\sidewalk_wheelchair\vectors\near_compassA.csv'
    df_yaw = pd.read_csv(yaw_csv_file)
    df_yaw = df_yaw.set_index("panoId")



    while len(img_list) > 0:
        try:
            img_path = img_list.pop()
            cnt = total_cnt - len(img_list)
            cnt += 1
            cal_witdh(img_path, df_yaw, crs_local=crs_local)
            if cnt % 100 == 0:
                print(cnt, img_path)
        except Exception as e:
            print("Error in cal_witdh_from_list():", e)
            continue

def read_worldfile(file_path):
    try:
        f = open(file_path, "r")
        lines = f.readlines()
        f.close()
        lines = [line[:-1] for line in lines]
        resolution = float(lines[0])
        upper_left_x = float(lines[4])
        upper_left_y = float(lines[5])
        return resolution, upper_left_x, upper_left_y
    except Exception as e:
        print("Error in read_worldfile(), return Nones:", e)
        return None, None, None


def cal_witdh(img_path, df_yaw, crs_local=6847):


    basename = os.path.basename(img_path)
    dirname = os.path.dirname(img_path)
    panoId = basename[:22]
    json_file = os.path.join(dirname, panoId + '.json')
    pano1 = GSV_pano(json_file=json_file, crs_local=6847)
    # print(pano1.jdata)


    pano_yaw_deg = pano1.jdata['Projection']['pano_yaw_deg']
    street_yaw = df_yaw.loc[panoId]['CompassA']
    if abs(street_yaw - pano_yaw_deg) > 150:
        pano_yaw_deg = street_yaw + 180
    else:
        pano_yaw_deg = street_yaw

    img_pil = Image.open(img_path)
    img_np = np.array(img_pil)
    # im_cv = cv2.imread(img_path)
    target_ids = [8, 12]

    class_idx = img_np
    target_np = np.zeros(img_np.shape)
    for i in target_ids:
        target_np = np.logical_or(target_np, class_idx == i)



    # AOI = np.where(AOI == 0, 0, 1).astype(np.uint8)

    morph_kernel_open  = (5, 5)
    morph_kernel_close = (10, 10)
    g_close = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_close)
    g_open  = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_open)

    target_np = target_np.astype(np.uint8)



    yaw_deg =  -pano_yaw_deg
    # print("yaw_deg:", yaw_deg)

    cv2_closed = cv2.morphologyEx(target_np, cv2.MORPH_CLOSE, g_close) # fill small gaps
    cv2_opened = cv2.morphologyEx(cv2_closed, cv2.MORPH_OPEN, g_open)

    cv2_opened = np.where(cv2_opened == 0, 0, 255).astype(np.uint8)

    opened_color = cv2.merge((cv2_opened, cv2_opened, cv2_opened))
    img_rotated = cv_img_rotate_bound(cv2_opened, yaw_deg)

    # cv2.imshow("img_rotated", np.where(img_rotated == 0, 0, 255).astype(np.uint8))


    # draw lines
    line_cnt = LINE_COUNT
    img_h, img_w = img_rotated.shape
    start_x = 0
    end_x = img_w -1
    interval = int(img_h / line_cnt)
    line_ys = range(interval, img_h, interval)
    line_thickness = 1


    to_RLE = img_rotated[line_ys]
    run_lengths, run_rows = rle_encoding(to_RLE)
    # print("rung_lengths, rows:\n", run_lengths[::2], "\n", run_rows, "\n", run_lengths[1::2])


    # cv2.imshow("Raw image", AOI.astype(np.uint8))

    # cv2_closed = np.where(cv2_closed == 0, 0, 255).astype(np.uint8)
    # cv2.imshow("cv2_closed", cv2_closed.astype(np.uint8))

    # for y in line_ys:
    #     cv2.line(opened_color, (start_x, y), (end_x, y), (0, 255, 0), thickness=line_thickness)

    angle_deg = 0
    angle_rad = math.radians(angle_deg)
    lengths = run_lengths[1::2]
    lengths = np.array(lengths)
    new_lengths = lengths.copy()
    new_run_cols = run_lengths[::2].copy()
    pen_lengths = lengths * math.cos(angle_rad)
    to_x = (pen_lengths * math.cos(angle_rad)).astype(int)
    to_y = (pen_lengths * math.sin(angle_rad)).astype(int)
    max_width_meter = 30
    pix_resolution = 0.05
    max_width_pix = int(max_width_meter / pix_resolution)
    for idx, col in enumerate(run_lengths):
        if idx % 2 == 0:
            idx2 = int(idx / 2)
            row = run_rows[idx2] * interval + interval - 1
            radius = 5
            # print(row, col)
            length = run_lengths[idx + 1]
            new_run_cols[idx2] = col
            if (length > max_width_pix) and (idx2 > 0):
                length = new_lengths[idx2 - 1]
                new_run_cols[idx2] = new_run_cols[idx2 - 1]
                print("long length!")
                # print("length, new_lengths[idx2], max_width_pix:", length, new_lengths[idx2], max_width_pix)

            new_lengths[idx2] = length
            new_run_cols[idx2] = new_run_cols[idx2]

            # print("col, new_run_cols[idx2]:", col, new_run_cols[idx2])
            col = new_run_cols[idx2]
            to_x[idx2] = new_lengths[idx2]
            # to_y[idx2] = row

            end_x = col + to_x[idx2]
            end_y = row + to_y[idx2]

            # cv2.line(opened_color, (col, row), (end_x, end_y), (0, 0, 255), thickness=line_thickness)
            # cv2.circle(opened_color, (col, row), radius, (0, 255, 0), line_thickness)


    # cv2.imshow("cv2_opened", opened_color)

    # find contour
    raw_contours, hierarchy = cv2.findContours(img_rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # contours = [np.squeeze(cont) for cont in raw_contours]
    #
    # start_time = time.perf_counter()
    # centerlines = get_polygon_centline(contours[0:])
    # print("Time used to get centerline: ", time.perf_counter() - start_time)
    # p = Polygon([[0, 0], [2, 2], [2, 0]])
    # pts = centerlines.reshape((-1,1,2))
    # all_pair_list = seg_contours(raw_contours[5:6], opened_color)
    img_rotated_color = cv2.merge((img_rotated, img_rotated, img_rotated))

    category_list = [58, 255]
    raw_img_rotated = cv_img_rotate_bound(img_np, yaw_deg)
    # cv2.imshow("raw_img_rotated", raw_img_rotated)

    car_mask_np = create_mask(raw_img_rotated, category_list=category_list)

    # for l in centerlines:
    #     pts = l.coords.xy
    #     pts = np.array(pts).T.reshape((-1, 1, 2)).astype(np.int32)
    #     # pts = np.array(pts).T.
    #     cv2.polylines(img_rotated_color, [pts], False, (255), 2)

    # all_pair_list = seg_contours(raw_contours[:], img_rotated_color)
    all_pair_list = seg_contours(raw_contours[:], car_mask_np, img_rotated)
    # print(all_pair_list)
    end_points = np.zeros((len(all_pair_list) * 2, 2))

    for idx, pair in enumerate(all_pair_list):
        x = int(pair[1])
        y = int(pair[2])
        end_x = int(pair[3])
        end_y = int(pair[4])
        cv2.line(img_rotated_color, (x, y), (end_x, end_y), (255, 0, 0), thickness=line_thickness)
        end_points[idx * 2] = np.array([x, y])
        end_points[idx * 2 + 1] = np.array([end_x, end_y])

    # cv2.imshow("img_rotated added pairs", img_rotated.astype(np.uint8))

    # end_points = np.hstack((end_points, np.ones((end_points.shape[0], 1))))
    tx = img_rotated.shape[0] / 2
    ty = img_rotated.shape[1] / 2
    # print("tx, ty:", tx, ty)
    end_points_transed = points_2D_translation(end_points, tx, ty)
    # print("end_points_transed:", end_points_transed[0])

    tx = target_np.shape[0] / 2
    ty = target_np.shape[1] / 2
    #
    # print("tx, ty:", tx, ty)

    end_points_rotated = points_2D_rotated(end_points_transed, yaw_deg)
    # print("end_points_rotated:", end_points_rotated[0])
    end_points_transed = points_2D_translation(end_points_rotated, -tx, ty)

    # print("final end_points_transed:", end_points_transed.astype(int))
    line_thickness = 1
    radius = 2
    raw_AOI_color = np.where(target_np == 0, 0, 255).astype(np.uint8)
    raw_AOI_color = cv2.merge((raw_AOI_color, raw_AOI_color, raw_AOI_color))
    line_cnt = len(end_points_transed)
    line_cnt = int(line_cnt)
    end_points_transed = end_points_transed.astype(int)

    dom_path = r'AZK1jDGIZC1zmuooSZCzEg.tif'
    dom_path = r'-ft2bZI1Ial4C6N_iwmmvw_DOM_0.05.tif'
    # im_dom = cv2.imread(dom_path)
    for idx in range(0, line_cnt, 2):
        col = end_points_transed[idx][0]
        row = end_points_transed[idx][1]
        # to_y[idx2] = row

        end_x = end_points_transed[idx + 1][0]
        end_y = end_points_transed[idx + 1][1]
        # cv2.line(opened_color, (col, row), (end_x , end_y), (0, 0, 255), thickness=line_thickness)
        # cv2.line(im_dom,       (col, row), (end_x , end_y), (0, 0, 255), thickness=line_thickness)
        # cv2.circle(raw_AOI_color, (end_x, end_y), radius, (0, 255, 0), line_thickness)
        # cv2.circle(raw_AOI_color, (col, row), radius, (0, 255, 0), line_thickness)

    # write txt
    saved_path = r'D:\Research\sidewalk_wheelchair\DC_DOMs_measuremens2'
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    # saved_path = r'H:\Research\sidewalk_wheelchair\DC_DOMs_measuremens'
    new_name = os.path.join(saved_path, f'{panoId}_widths.csv')
    worldfile_ext = img_path[-3] + img_path[-1] + 'w'
    worldfile_path = img_path[:-3] + worldfile_ext
    wf_resolution, wf_x, wf_y = read_worldfile(worldfile_path)
    # print("wf_resolution, wf_x, wf_y:", wf_resolution, wf_x, wf_y)
    f = open(new_name, 'w')
    f.writelines('panoId,contour_num,center_x,center_y,length,start_x,start_y,end_x,end_y,cover_ratio,is_touched\n')
    # print("new_name:", new_name)
    for idx in range(0, line_cnt, 2):
        col = end_points_transed[idx][0] * wf_resolution + wf_x
        row = wf_y - end_points_transed[idx][1] * wf_resolution
        # to_y[idx2] = row

        end_x = end_points_transed[idx + 1][0] * wf_resolution + wf_x
        end_y = wf_y - end_points_transed[idx + 1][1] * wf_resolution

        center_x = (end_x + col) / 2
        center_y = (end_y + row) / 2

        idx2 = int(idx/2)
        length = all_pair_list[idx2][3] - all_pair_list[idx2][1]
        contour_num = all_pair_list[idx2][0]
        length = length * wf_resolution
        cover_ratio = all_pair_list[idx2][5]
        is_touched = all_pair_list[idx2][6]

        f.writelines(f'{panoId},{contour_num},{center_x:.3f},{center_y:.3f},{length:.3f},{col:.3f},{row:.3f},{end_x:.3f},{end_y:.3f},{cover_ratio:.3f},{int(is_touched)}\n')
        # print("center_x, center_x:", f'{center_x},{center_y},{length},{col},{row},{end_x},{end_y}\n')
        # f.writelines(f'{center_x},{center_y},{length},{col},{row},{end_x},{end_y}\n')
    f.close()

    # cv2.imshow("opened_color", opened_color)
    # cv2.imshow("im_dom", im_dom)
    # cv2.imshow("img_rotated_color", img_rotated_color)

    # end_points_transed = end_points_transed[:, 0:2]


    # rotated = imutils.rotate_bound(opened_color, -45)....


    # to_RLE = np.where(to_RLE == 0, 0, 255).astype(np.uint8)
    # cv2.imshow("to_RLE", to_RLE.astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def cal_witdh0(img_path):
    # img_path = r'AZK1jDGIZC1zmuooSZCzEg_DOM.tif'
    # # img_path = r'Ld-CMATy8ZxKap6VAtZTEg_DOM.tif'
    # # img_path = r'v-VR9FB7kCxU1eDLEtFiJQ_DOM.tif'
    # img_path = r'-0D29S37SnmRq9Dju9hkqQ_DOM.tif'
    img_path = r'ZXyk9lKhL5siKJglQPqfMA_DOM_0.05.tif'

    # img_path = r'H:\USC_OneDrive\OneDrive - University of South Carolina\Research\sidewalk_wheelchair\DC_DOMs\-ft2bZI1Ial4C6N_iwmmvw_DOM_0.05.tif'


    panoId_2019 = r'ZXyk9lKhL5siKJglQPqfMA'
    pano1 = GSV_pano(panoId=panoId_2019, crs_local=6847, saved_path=os.getcwd())
    # print(pano1.jdata)
    pano_yaw_deg = pano1.jdata['Projection']['pano_yaw_deg']

    target_ids = [244]
    im_cv = cv2.imread(img_path)

    red_channel = im_cv[:, :, 2]
    class_idx = red_channel
    AOI = np.zeros((len(class_idx), len(class_idx)))
    for i in target_ids:
        AOI = np.logical_or(AOI, class_idx == i)
        # print(AOI)
    # AOI = np.where(AOI == 0, 0, 1).astype(np.uint8)
    # print(AOI)

    morph_kernel_open  = (5, 5)
    morph_kernel_close = (10, 10)
    g_close = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_close)
    g_open  = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_open)

    raw_AOI = np.where(AOI == 0, 0, 1).astype(np.uint8)

    # cv2.imshow("Raw image", np.where(raw_AOI == 0, 0, 255).astype(np.uint8) )

    # yaw_deg =  226.4377593994141 - 90
    # yaw_deg =  92.53645324707031
    yaw_deg =  -pano_yaw_deg
    # print("yaw_deg:", yaw_deg)
    #
    cv2_closed = cv2.morphologyEx(raw_AOI, cv2.MORPH_CLOSE, g_close) # fill small gaps
    cv2_opened = cv2.morphologyEx(cv2_closed, cv2.MORPH_OPEN, g_open)

    cv2_opened = np.where(cv2_opened == 0, 0, 255).astype(np.uint8)

    opened_color = cv2.merge((cv2_opened, cv2_opened, cv2_opened))
    img_rotated = imutils.rotate_bound(cv2_opened, yaw_deg)

    # draw lines
    line_cnt = 160
    img_h, img_w = img_rotated.shape
    start_x = 0
    end_x = img_w -1
    interval = int(img_h / line_cnt)
    line_ys = range(interval, img_h, interval)
    line_thickness = 1


    to_RLE = img_rotated[line_ys]
    run_lengths, run_rows = rle_encoding(to_RLE)
    # print("rung_lengths, rows:\n", run_lengths[::2], "\n", run_rows, "\n", run_lengths[1::2])

    AOI = np.where(AOI == 0, 0, 255).astype(np.uint8)

    # cv2.imshow("Raw image", AOI.astype(np.uint8))

    # cv2_closed = np.where(cv2_closed == 0, 0, 255).astype(np.uint8)
    # cv2.imshow("cv2_closed", cv2_closed.astype(np.uint8))

    for y in line_ys:
        cv2.line(opened_color, (start_x, y), (end_x, y), (0, 255, 0), thickness=line_thickness)

    angle_deg = 0
    angle_rad = math.radians(angle_deg)
    lengths = run_lengths[1::2]
    lengths = np.array(lengths)
    new_lengths = lengths.copy()
    new_run_cols = run_lengths[::2].copy()
    pen_lengths = lengths * math.cos(angle_rad)
    to_x = (pen_lengths * math.cos(angle_rad)).astype(int)
    to_y = (pen_lengths * math.sin(angle_rad)).astype(int)
    max_width_meter = 30
    pix_resolution = 0.05
    max_width_pix = int(max_width_meter / pix_resolution)
    for idx, col in enumerate(run_lengths):
        if idx % 2 == 0:
            idx2 = int(idx / 2)
            row = run_rows[idx2] * interval + interval - 1
            radius = 5
            # print(row, col)
            length = run_lengths[idx + 1]
            new_run_cols[idx2] = col
            if (length > max_width_pix) and (idx2 > 0):
                length = new_lengths[idx2 - 1]
                new_run_cols[idx2] = new_run_cols[idx2 - 1]
                print("long length!")
                # print("length, new_lengths[idx2], max_width_pix:", length, new_lengths[idx2], max_width_pix)

            new_lengths[idx2] = length
            new_run_cols[idx2] = new_run_cols[idx2]

            # print("col, new_run_cols[idx2]:", col, new_run_cols[idx2])
            col = new_run_cols[idx2]
            to_x[idx2] = new_lengths[idx2]
            # to_y[idx2] = row

            end_x = col + to_x[idx2]
            end_y = row + to_y[idx2]

            # cv2.line(opened_color, (col, row), (end_x, end_y), (0, 0, 255), thickness=line_thickness)
            # cv2.circle(opened_color, (col, row), radius, (0, 255, 0), line_thickness)


    # cv2.imshow("cv2_opened", opened_color)

    # find contour
    raw_contours, hierarchy = cv2.findContours(img_rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # contours = [np.squeeze(cont) for cont in raw_contours]
    #
    # start_time = time.perf_counter()
    # centerlines = get_polygon_centline(contours[0:])
    # print("Time used to get centerline: ", time.perf_counter() - start_time)
    # p = Polygon([[0, 0], [2, 2], [2, 0]])
    # pts = centerlines.reshape((-1,1,2))
    # all_pair_list = seg_contours(raw_contours[5:6], opened_color)
    img_rotated_color = cv2.merge((img_rotated, img_rotated, img_rotated))

    # for l in centerlines:
    #     pts = l.coords.xy
    #     pts = np.array(pts).T.reshape((-1, 1, 2)).astype(np.int32)
    #     # pts = np.array(pts).T.
    #     cv2.polylines(img_rotated_color, [pts], False, (255), 2)

    all_pair_list = seg_contours(raw_contours[:], img_rotated_color)
    end_points = np.zeros((len(all_pair_list) * 2, 2))
    for idx, pair in enumerate(all_pair_list):
        x = int(pair[1])
        y = int(pair[2])
        end_x = int(pair[3])
        end_y = int(pair[4])
        cv2.line(img_rotated_color, (x, y), (end_x, end_y), (255, 0, 0), thickness=line_thickness)
        end_points[idx * 2] = np.array([x, y])
        end_points[idx * 2 + 1] = np.array([end_x, end_y])

    # cv2.imshow("img_rotated added pairs", img_rotated.astype(np.uint8))

    end_points = np.hstack((end_points, np.ones((end_points.shape[0], 1))))
    tx = img_rotated.shape[0] / 2
    ty = img_rotated.shape[1] / 2
    # print("tx, ty:", tx, ty)
    end_points_transed = points_2D_translation(end_points, tx, ty)
    # print("end_points_transed:", end_points_transed[0])

    tx = raw_AOI.shape[0] / 2
    ty = raw_AOI.shape[1] / 2
    #
    # print("tx, ty:", tx, ty)

    end_points_rotated = points_2D_rotated(end_points_transed, yaw_deg)
    # print("end_points_rotated:", end_points_rotated[0])
    end_points_transed = points_2D_translation(end_points_rotated, -tx, ty)

    # print("final end_points_transed:", end_points_transed.astype(int))
    line_thickness = 1
    radius = 2
    raw_AOI_color = np.where(raw_AOI == 0, 0, 255).astype(np.uint8)
    raw_AOI_color = cv2.merge((raw_AOI_color, raw_AOI_color, raw_AOI_color))
    line_cnt = len(end_points_transed)
    line_cnt = int(line_cnt)
    end_points_transed = end_points_transed.astype(int)

    # dom_path = r'AZK1jDGIZC1zmuooSZCzEg.tif'
    # dom_path = r'-ft2bZI1Ial4C6N_iwmmvw_DOM_0.05.tif'
    # im_dom = cv2.imread(dom_path)
    for idx in range(0, line_cnt, 2):
        col = end_points_transed[idx][0]
        row = end_points_transed[idx][1]
        # to_y[idx2] = row

        end_x = end_points_transed[idx + 1][0]
        end_y = end_points_transed[idx + 1][1]
        cv2.line(opened_color, (col, row), (end_x , end_y), (0, 0, 255), thickness=line_thickness)
        # cv2.line(im_dom,       (col, row), (end_x , end_y), (0, 0, 255), thickness=line_thickness)
        # cv2.circle(raw_AOI_color, (end_x, end_y), radius, (0, 255, 0), line_thickness)
        # cv2.circle(raw_AOI_color, (col, row), radius, (0, 255, 0), line_thickness)
    # cv2.imshow("opened_color", opened_color)
    # cv2.imshow("im_dom", im_dom)
    # cv2.imshow("img_rotated_color", img_rotated_color)

    # end_points_transed = end_points_transed[:, 0:2]


    # rotated = imutils.rotate_bound(opened_color, -45)....


    # cv2.imshow("rotated", rotated)
    # to_RLE = np.where(to_RLE == 0, 0, 255).astype(np.uint8)
    # cv2.imshow("to_RLE", to_RLE.astype(np.uint8))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def points_2D_translation_math(points, tx, ty):
    points = np.hstack((points, np.ones((points.shape[0], 1))))
    t_mat = np.array([[1, 0, -tx],
                      [0, 1, -ty],
                      [0, 0, 1]])
    results = points.dot(t_mat.T)
    return results[:, 0:2]

def points_2D_translation(points, tx, ty):
    points = np.hstack((points, np.ones((points.shape[0], 1))))
    t_mat = np.array([[1, 0, -tx],
                      [0, -1, ty],
                      [0, 0, 1]])
    results = points.dot(t_mat.T)
    return results[:, 0:2]

def points_2D_rotated(points, angle_deg):
    points = np.hstack((points, np.ones((points.shape[0], 1))))
    angle = math.radians(angle_deg)
    r_mat = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1]])
    results = points.dot(r_mat.T)
    return results[:, 0:2]



def seg_contours(raw_contours, mask_np,  img_rotated, interval_pix=10, max_width_pix=50):
    contours = [np.squeeze(cont) for cont in raw_contours]

    # con = cv2.drawContours(opened_color_img, raw_contours, -1, (0, 255, 0), 2)
    # cv2.imshow("raw_contours", opened_color_img)

    centeral_col = mask_np.shape[1] / 2
    centeral_col = int(centeral_col)

    all_pair_list = []

    for idx, contour in enumerate(contours):
        Xs = contour[:, 0]
        Ys = contour[:, 1]
        y_min = Ys.min()
        y_max = Ys.max()
        x_min = Xs.min()
        x_max = Xs.max()
        h = y_max - y_min
        cut_row_cnt = np.ceil(h / interval_pix).astype(int) + 1
        cut_rows = np.array((range(cut_row_cnt))) * interval_pix +  y_min
        cut_rows[-1] = y_max
        pairs = np.zeros((cut_row_cnt, 2))
        for idx2, row in enumerate(cut_rows):
            cols = contour[contour[:, 1] == row][:, 0]
            cols = np.sort(cols)

            pair = get_pair_col(cols, centeral_col)

            length = pair[1] - pair[0]
            # Huan  !!
            # if (length > max_width_pix) and (idx2 > 0):
            #     pair = pairs[idx2 - 1]

            pairs[idx2] = pair

            is_touched = line_ends_touched(pair[0], row, pair[1], row, mask_np, width=13, height=1, threshold=6)

            cover_ratio = -1

            target_mask_np = np.logical_or(img_rotated, mask_np)

            cover_ratio = get_cover_ratio(int(np.mean(pair[0:2])), int(row), target_mask_np, width=length, height=length)
            # cover_ratio = 0
            cover_ratio_threshold = 0.5

            all_pair_list.append((idx, pair[0], row, pair[1], row, cover_ratio, is_touched))
            # pairs.append(pair)
        # all_pair_list
    # print("all_pair_list:", all_pair_list)

    return all_pair_list

def get_cover_ratio(col, row, mask_np, width, height):

    try:
        cover_ratio = -1
        if width * height == 0:
            return cover_ratio
        mask_w, mask_h = mask_np.shape
        col_start = max(0, int(col - width/2))
        col_end = min(mask_w - 1, int(col + width/2))

        row_start = max(0, int(row - height/2))
        row_end = min(mask_h - 1, int(row + height/2))

        samples = mask_np[row_start:row_end, col_start:col_end]
        # samples = samples / 255

        cover_ratio = samples.sum() / float(width * height)

        #  show image
        # cv_mask_color = cv2.merge([np.where(mask_np > 0, 255, 0).astype(np.uint8)])
        # cv_mask_color = cv2.merge([cv_mask_color, cv_mask_color, cv_mask_color])
        # top_left = (col_start, row_start)
        # bottom_right = (col_end, row_end)
        # cv2.rectangle(cv_mask_color, top_left, bottom_right, color=(0, 255, 0), thickness=2)
        # cv2.imshow(f"cover_ratio: {cover_ratio:.3f}", cv_mask_color)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return cover_ratio
    except Exception as e:
        print("Error in get_cover_ratio():", e)
        return cover_ratio

def line_ends_touched(x1, y1, x2, y2, mask_np, width=13, height=1, threshold=6):
    is_touched1 = check_touched(x1, y1, mask_np, width=width, height=height, threshold=threshold)
    is_touched2 = check_touched(x2, y2, mask_np, width=width, height=height, threshold=threshold)
    #
    # cv2.imshow("mask_np", cv2.merge([np.where(mask_np > 0, 255, 0).astype(np.uint8)]))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # re = (is_touched1 or is_touched2)

    return (is_touched1 or is_touched2)

def create_mask(img_np, category_list):
    '''
    create a mask, where True indicates the taget categories.
    :param img_np:
    :param category_list:
    :return:
    '''
    mask_np = np.zeros(shape=img_np.shape, dtype=bool)
    class_idx = img_np
    for category in category_list:
        mask_np = np.logical_or(mask_np, img_np == category)

    return mask_np

def check_touched(col, row, mask_np, width=6, height=1, threshold=6):
    '''
    check whethe a sidewalk touchs cars. car: 58, other: 255
    :param col:
    :param row:
    :param img_np:
    :param threshold:
    :return:
    '''
    mask_w, mask_h = mask_np.shape
    col_start = max(0, int(col - width/1))
    col_end = min(mask_w - 1, int(col + width/1))

    row_start = max(0, int(row - height/1))
    row_end = min(mask_h - 1, int(row + height/1))

    samples = mask_np[row_start:row_end, col_start:col_end]

    is_touched = samples.sum() > threshold

    return is_touched

def get_pair_col(cols, central_col):
    '''
    Find out the start and end col from cols intersecting the horizontal row.
    :param cols: numpy 1-D array
    :return:
    '''

    left = cols.min()
    right = cols.max()
    length = right - left

    # Case 1: one line in the top/bottom
    if length == (len(cols) - 1):
        pass

    # Case 2: two cols only
    if (len(cols) == 2) and (length > 2):
        pass

    # Case 3: several parts, like intersecting with two fingers.
    # step a: find segments
    starts = []
    prev = -2
    for col in cols:
        if col > (prev+1):
            starts.extend((col,))
        prev = col
    if len(starts) == 1:
        starts.append(cols[-1])

    if starts[0] > central_col:
        left = starts[0]
        right = starts[1]
    else:
        left = starts[-2]
        right = starts[-1]

    return left, right


def get_centerline_from_img(img_path):
    img_pil = Image.open(img_path)
    img_np = np.array(img_pil)


    im_cv = cv2.imread(img_path)
    target_ids = [12]

    class_idx = img_np
    target_np = np.zeros(img_np.shape)
    for i in target_ids:
        target_np = np.logical_or(target_np, class_idx == i)

    # img_cv = cv2.cvtColor(np.asarray(img_pil),cv2.IMREAD_GRAYSCALE)
    target_np = target_np.astype(np.uint8)
    target_cv = cv2.merge([target_np])
    raw_contours, hierarchy = cv2.findContours(target_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [np.squeeze(cont) for cont in raw_contours]
    centerlines = get_polygon_centline(contours)

    print("OK")

    # centerline1 = get_centerline(polygon, segmentize_maxlen=8, max_points=3000, simplification=0.05, smooth_sigma=5)

def get_polygon_centline(contours, world_coords=[], segmentize_maxlen=1, max_points=3000, simplification=0.5, smooth_sigma=5):
    if not isinstance(contours, list):
        contours = [contours]
    results = []
    for contour in contours:
        try:
            polygon = Polygon(contour)
            print(polygon)
            centerline1 = get_centerline(polygon, segmentize_maxlen=segmentize_maxlen, max_points=max_points, simplification=simplification, smooth_sigma=smooth_sigma)
            print(centerline1)
            results.append(centerline1)
        except Exception as e:
            print("Error in get_polygon_centline():", e)
            results.append(0)
            continue

    if len(world_coords) > 0:
        # coords =
        # centerlines = [np.c_[cont, np.ones((len(cont),))] for cont in contours]
        results = [shapely.affinity.affine_transform(cont, world_coords) for cont in results]
        print(results)

    return results

# utest passed.
def get_forward_backward_pano(pano, json_dir='', forward=True, backward=True):
    try:
        forward_pano = None
        backward_pano = None
        Links = pano.jdata['Links']
        pano_yaw_deg = pano.jdata['Projection']['pano_yaw_deg']
        forward_pano_link, backward_pano_link = utils.find_forward_bacwark_link(Links, pano_yaw_deg)

        backward_panoId = backward_pano_link['panoId']
        forward_panoId  =  forward_pano_link['panoId']

        if forward:
            if json_dir != "":
                forward_pano_json_file  = os.path.join(json_dir,  forward_panoId + ".json")
                forward_pano = GSV_pano(json_file=forward_pano_json_file)
            else:
                forward_pano = GSV_pano(panoId=forward_panoId)

        if backward:
            backward_panoId = backward_pano_link['panoId']
            if json_dir != "":
                backward_pano_json_file = os.path.join(json_dir, backward_panoId + ".json")
                backward_pano = GSV_pano(json_file=backward_pano_json_file)
            else:
                backward_pano = GSV_pano(panoId=backward_panoId)

        if backward and forward:
            return forward_pano, backward_pano

        if forward:
            return forward_pano

        if backward:
            return backward_pano

    except Exception as e:
        logging.error("Error in get_forward_pano(): %s" % (e,))



# utest passed.
def get_pano_neighbors(panoId="", json_file='', neighbor_order=1):
    neighbor_order = int(neighbor_order)
    if neighbor_order > 2:
        logging.error("Error in get_pano_neighbors: neighbor_order %s > 2. Support 1, and 2 only." % (neighbor_order, ), exc_info=True)
    neighbors = []
    try:
        json_dir = os.path.dirname(json_file)
        if json_file != "":
            if os.path.exists(json_file):
                pano = GSV_pano(json_file=json_file)
            else:
                logging.error("json_file is no get_pano_neighbors(): %s" % json_file)
        if panoId != "":
            pano = GSV_pano(panoId=panoId)

        forward_pano, backward_pano = get_forward_backward_pano(pano, json_dir=json_dir)


        if neighbor_order == 1:
            return (forward_pano, backward_pano)

        if neighbor_order == 2:
            forward_pano_forward  = get_forward_backward_pano(forward_pano, json_dir=json_dir, backward=False)
            backward_pano_backward = get_forward_backward_pano(backward_pano, json_dir=json_dir, forward=False)

            return (forward_pano, backward_pano, forward_pano_forward, backward_pano_backward)

    except Exception as e:
        logging.error("Error in get_pano_neighbors: %s" % (e, ), exc_info=True)

def read_neighbor_measurements(neighbors, measure_dir):
    measurements_dfs = []
    for n in neighbors:
        measurements_file = os.path.join(measure_dir, n.panoId + "_widths.csv")
        if os.path.exists(measurements_file):
            df = pd.read_csv(measurements_file)
            measurements_dfs.append(df)
        else:
            measurements_dfs.append(None)
    return measurements_dfs

def filter_measurement(df):
    df1 = df[df['cover_ratio'] > COVER_RATIO_MINIMUM]
    df1 = df1[df1['is_touched'] == 0]
    df1 = df1[df1['length'] > 0.3]
    return df1

def rotate_pano_center(pano, tx, ty, yaw_deg):
    pano_center = points_2D_translation_math(np.array([[pano.x, pano.y]]), tx, ty)
    pano_center = points_2D_rotated(pano_center, yaw_deg)[0]
    return pano_center

def draw_measurements(ax, measure_df, rotated=True, color='red'):
    for idx, row in measure_df.iterrows():
        if not rotated:
            Xs = (row['col'], row['end_x'])
            Ys = (row['row'], row['end_y'])

        else:
            Xs = (row['rotated_start_x'], row['rotated_end_x'])
            Ys = (row['rotated_start_y'], row['rotated_end_y'])
        ax.plot(Xs, Ys, color=color)
        # l = mlines.Line2D(Xs, Ys)
        # ax.add_line(l)
    return ax

def draw_pano_center(ax, pano, tx=0, ty=0, yaw_deg=0, color='red'):
    pano_center = points_2D_translation_math(np.array([[pano.x, pano.y]]), tx, ty)
    pano_center = points_2D_rotated(pano_center, yaw_deg)[0]
    ax.scatter(pano_center[0], pano_center[1], color=color)
    text_y_offset = 0.5
    ax.text(pano_center[0], pano_center[1] + text_y_offset, s=pano.panoId, color=color)
    return ax

def get_random_RGB_color(cnt):
    return [[np.random.rand(3,)] for i in range(cnt)]

def draw_neighors(ax, panos, measurement_dfs, tx=0, ty=0, yaw_deg=0):
    colors = get_random_RGB_color(max(len(panos), len(measurement_dfs)))
    first_colors = ['red', 'green', 'blue', 'purple', 'black']
    colors = first_colors + colors
    for idx, pano in enumerate(panos):
        draw_pano_center(ax, pano, tx, ty, yaw_deg, color=colors[idx])
        if yaw_deg == 0:
            draw_measurements(ax, measurement_dfs[idx], rotated=False, color=colors[idx])
        else:
            draw_measurements(ax, measurement_dfs[idx], rotated=True, color=colors[idx])
            pass

def cut_measurements(df, column=r''):
    pass

def find_sidewalk_segments(pano_XYs_rotated, measurement_dfs):

    try:
        top = pano_XYs_rotated[1][1] / 2
        bottom = pano_XYs_rotated[2][1] / 2

        interval_cnt = int((top - bottom) / SIDEWALK_INTERVAL)

        cutoff_dfs = [df[df['rotated_center_y'].between(bottom, top)] for df in measurement_dfs]

        medians = get_4_segments_from_measurements(cutoff_dfs)

        print("widths: ", np.array(medians)[:, 2])

        upper_right_medians, bottom_right_medians, upper_left_medians, bottom_left_medians = medians
        right_x = (upper_right_medians[0] + bottom_right_medians[0]) / 2
        left_x = (upper_left_medians[0] + bottom_left_medians[0]) / 2

        right_point = (right_x, 0)
        left_point = (left_x, 0)

        # uppter_right_point = (, top)

        corners = get_4_corners(pano_XYs_rotated, measurement_dfs)
        upper_right_point = (corners[0][0], top)
        bottom_right_point = (corners[1][0], bottom)
        upper_left_point = (corners[2][0], top)
        bottom_left_point = (corners[3][0], bottom)

        upper_right_segment = [right_point, upper_right_point, medians[0][2]]
        bottom_right_segment = [bottom_right_point, right_point, medians[1][2]]
        bottom_left_segment = [bottom_left_point, left_point, medians[2][2]]
        upper_left_segment = [left_point, upper_left_point, medians[3][2]]


        return [upper_right_segment, bottom_right_segment, bottom_left_segment, upper_left_segment]

    except Exception as e:
        logging.error("Error in find_sidewalk_segments(): %s" % e)

def add_more_measurement(df, measurement_dfs, right=True, upper=True):
    if len(df) < 1:
        df = pd.concat(measurement_dfs[1:3], axis=0)
        if right:
            df = df[df['rotated_center_x'] > 0]
        else:
            df = df[df['rotated_center_x'] < 0]

        if upper:
            df = df[df['rotated_center_y'] > 0]
        else:
            df = df[df['rotated_center_y'] < 0]

    if len(df) < 1:
        df = pd.concat(measurement_dfs[3:5], axis=0)
        if right:
            df = df[df['rotated_center_x'] > 0]
        else:
            df = df[df['rotated_center_x'] < 0]

        if upper:
            df = df[df['rotated_center_y'] > 0]
        else:
            df = df[df['rotated_center_y'] < 0]

    return df

def get_4_segments_from_measurements(measurement_dfs):
    try:
        pano_df = measurement_dfs[0]
        left_df = pano_df[pano_df['rotated_center_x'] < 0]
        right_df = pano_df[pano_df['rotated_center_x'] > 0]

        upper_right_df = right_df[right_df['rotated_center_y'] > 0]
        upper_right_df = add_more_measurement(upper_right_df, measurement_dfs, right=True, upper=True)
        upper_right_medians = get_measurement_medians(upper_right_df, columns=['rotated_center_x', 'rotated_center_y', 'length'])


        bottom_right_df = right_df[right_df['rotated_center_y'] < 0]
        bottom_right_df = add_more_measurement(bottom_right_df, measurement_dfs, right=True, upper=False)
        bottom_right_medians = get_measurement_medians(bottom_right_df, columns=['rotated_center_x', 'rotated_center_y', 'length'])


        upper_left_df = left_df[left_df['rotated_center_y'] > 0]
        upper_left_df = add_more_measurement(upper_left_df, measurement_dfs, right=False, upper=True)
        upper_left_medians = get_measurement_medians(upper_left_df, columns=['rotated_center_x', 'rotated_center_y', 'length'])

        bottom_left_df = left_df[left_df['rotated_center_y'] < 0]
        bottom_left_df = add_more_measurement(bottom_left_df, measurement_dfs, right=False, upper=False)
        bottom_left_medians = get_measurement_medians(bottom_left_df, columns=['rotated_center_x', 'rotated_center_y', 'length'])

        return [upper_right_medians, bottom_right_medians, bottom_left_medians, upper_left_medians]
    except Exception as e:
        logging.error("Error in get_4_segments_from_measurements(): %s" % e)

def get_4_corners(pano_XYs_rotated, measurement_dfs):

    center_xy = pano_XYs_rotated[0]
    upper_xy = pano_XYs_rotated[1]
    bottom_xy = pano_XYs_rotated[2]

    center_df = measurement_dfs[0]
    upper_df = measurement_dfs[1]
    bottom_df = measurement_dfs[2]

    measurement_all = pd.concat(measurement_dfs[:3])
    measurement_all = measurement_all[measurement_all['rotated_center_y'] < upper_xy[1]]
    measurement_all = measurement_all[measurement_all['rotated_center_y'] > bottom_xy[1]]

    measure_right_df = measurement_all[measurement_all['rotated_center_x'] > 0]
    measure_left_df = measurement_all[measurement_all['rotated_center_x'] < 0]

    measure_upper_right_df = measure_right_df[measure_right_df['rotated_center_y'] > 0]
    measure_bottom_right_df = measure_right_df[measure_right_df['rotated_center_y'] < 0]

    measure_upper_left_df = measure_left_df[measure_left_df['rotated_center_y'] > 0]
    measure_bottom_left_df = measure_left_df[measure_left_df['rotated_center_y'] < 0]

    upper_right_medians = get_measurement_medians(measure_upper_right_df,
                                                  columns=['rotated_center_x', 'rotated_center_y', 'length'])

    bottom_right_medians = get_measurement_medians(measure_bottom_right_df,
                                                   columns=['rotated_center_x', 'rotated_center_y', 'length'])

    upper_left_medians = get_measurement_medians(measure_upper_left_df,
                                                 columns=['rotated_center_x', 'rotated_center_y', 'length'])

    bottom_left_medians = get_measurement_medians(measure_bottom_left_df,
                                                  columns=['rotated_center_x', 'rotated_center_y', 'length'])

    return [upper_right_medians, bottom_right_medians, upper_left_medians, bottom_left_medians]

def get_measurement_medians(df, columns=['rotated_center_x', 'rotated_center_y', 'length']):
    medians = []
    try:
        for column in columns:
            if column == 'length':
                medians.append(df[column].quantile(0.15))
            else:
                medians.append(df[column].median())


        return medians
    except Exception as e:
        logging.error("Error in get_measurement_medians(): %s" % e)

def buffer_line(linestring, distance):
    assert(isinstance(linestring, LineString))
    distance = float(distance)
    left_line = linestring.parallel_offset(distance=distance, side="left", join_style=2)
    right_line = linestring.parallel_offset(distance=distance, side="right", join_style=2)

    left_xy = left_line.xy
    right_xy = right_line.xy
    polygon_xy = (left_xy[0] + right_xy[0], left_xy[1] + right_xy[1])
    polygon_points = list(zip(polygon_xy[0], polygon_xy[1]))
    polygon_points.append(polygon_points[0])

    polygon = Polygon(polygon_points)

    return polygon

def  sidewalk_connect():
    measure_dir = r'D:\Research\sidewalk_wheelchair\DC_DOMs_measuremens2'
    json_dir = r'D:\Research\sidewalk_wheelchair\json'
    json_dir = r'D:\Research\sidewalk_wheelchair\DC_DOMs'
    # measure_list = [r'D:\Research\sidewalk_wheelchair\DC_DOMs_measuremens2\QbwFEBFshO-mIu3eL5Dlcg_widths.csv']
    measure_list = [r'D:\Research\sidewalk_wheelchair\DC_DOMs_measuremens2\Ww21SnxYVOMQMveHrmZFQw_widths.csv']
    measure_list = [r'D:\Research\sidewalk_wheelchair\DC_DOMs_measuremens2\NkuNqaZlKfRpbAFs1GhcGg_widths.csv']
    measure_list = [r'D:\Research\sidewalk_wheelchair\DC_DOMs_measuremens2\EDV0ojtKFiyMM1IdoZTNQw_widths.csv']
    measure_list = [r'D:\Research\sidewalk_wheelchair\DC_DOMs_measuremens2\pdYGZm3hSfSdwgsh1a6dlQ_widths.csv']
    measure_list = [r'D:\Research\sidewalk_wheelchair\DC_DOMs_measuremens2\O6IyeKOf5--It2_-O6arcA_widths.csv']


    measure_list = glob.glob(os.path.join(measure_dir, "*_widths.csv"))[:]

    measure_list = [r'D:\Research\sidewalk_wheelchair\DC_DOMs_measuremens2\9HHwEtLAJWLFXT3-nQBa5w_widths.csv']

    local_crs = 6487
    transformer = utils.epsg_transform(in_epsg=4326, out_epsg=local_crs)

    yaw_csv_file = r'H:\USC_OneDrive\OneDrive - University of South Carolina\Research\sidewalk_wheelchair\vectors\near_compassA.csv'
    yaw_csv_file = r'E:\USC_OneDrive\OneDrive - University of South Carolina\Research\sidewalk_wheelchair\vectors\near_compassA.csv'
    df_yaw = pd.read_csv(yaw_csv_file)
    df_yaw = df_yaw.set_index("panoId")

    lines = []
    df_sidewalk_seg = pd.DataFrame(columns={"slope_panoId": str,
                                            "width": float,
                                            "start_x": float,
                                            "start_y": float,
                                            "end_x": float,
                                            "end_y": float,
                                            "slope": float,
                                            "distance_to_pano": float
                                            # "involved_pano1": str,
                                            # "involved_pano2": str,
                                            # "involved_pano3": str,
                                            # "linestring": object
                                            }
                                   )

    total_cnt = len(measure_list)
    lines = []
    while len(measure_list) > 0:
        try:

            meaure_file = measure_list.pop()
            print("")
            print(f"Finished {total_cnt - len(measure_list)} / {total_cnt}. Processing: ", meaure_file)
            df = pd.read_csv(meaure_file)
            df = df[df['cover_ratio'] > 0.85]
            df = df[df['is_touched'] == 0]
            panoId = os.path.basename(meaure_file)[:22]
            json_file = os.path.join(json_dir, panoId + ".json")

            pano = GSV_pano(json_file=json_file)
            # pano.x, pano.y = transformer.transform(pano.lat, pano.lon)
            pano.calculate_xy(transformer=transformer)


            pano_yaw_deg = pano.jdata['Projection']['pano_yaw_deg']

            street_yaw = df_yaw.loc[panoId]['CompassA']
            angle_diff = utils.degree_difference(street_yaw, pano_yaw_deg)
            if angle_diff > 150:
                pano_yaw_deg = (street_yaw + 180) % 360
            else:
                pano_yaw_deg = street_yaw


            forward_pano, backward_pano, forward_pano_forward, backward_pano_backward = \
                get_pano_neighbors(json_file=json_file, neighbor_order=2)

            pano_neighbors = [pano, forward_pano, backward_pano, forward_pano_forward, backward_pano_backward]
            pano_XYs = [p.calculate_xy(transformer) for p in pano_neighbors]
            pano_XYs_rotated = [rotate_pano_center(p, pano.x, pano.y, pano_yaw_deg) for p in pano_neighbors]


            measure_neighbor_dfs = read_neighbor_measurements(pano_neighbors, measure_dir)

            rotated_measurements = [rotate_measurement_df(df,  pano.x, pano.y, pano_yaw_deg) for df in measure_neighbor_dfs]
            filtered_measurements = [filter_measurement(df) for df in rotated_measurements]




            # note the direction of y-axis needs to be clarified. The y-axis of geographic coordinate sysmem towarks up,
            # but the image row direction towards to bottom!

            # connect
            # determine the segment length

            sidewalk_segments = find_sidewalk_segments(pano_XYs_rotated, filtered_measurements)

            # draw rotated measurements
            draw_rotated = False
            # draw_rotated = True
            if draw_rotated:

                plt.subplots(figsize=(15, 30))
                ax = plt.gca()
                draw_neighors(ax, pano_neighbors, filtered_measurements, tx=pano.x, ty=pano.y, yaw_deg=pano_yaw_deg)
                for segment in sidewalk_segments:
                    if not math.isnan(segment[2]):
                        ax.plot((segment[0][0], segment[1][0]), (segment[0][1], segment[1][1]))
                        linestring = LineString([segment[0][:2], segment[1][:2]])
                        polyon = buffer_line(linestring, segment[2]/2)
                        ax.plot(polyon.exterior.xy[0], polyon.exterior.xy[1])
                        plt.axis('scaled')
                        plt.tight_layout()
                        plt.show()

            # draw raw measurements
            draw_raw = True
            draw_raw = False
            if draw_raw:
                plt.subplots(figsize=(15, 30))
                ax = plt.gca()
                for segment in sidewalk_segments:
                    draw_neighors(ax, pano_neighbors, filtered_measurements, tx=0, ty=0, yaw_deg=0)
                    if not math.isnan(segment[2]):
                        linestring = LineString([segment[0][:2], segment[1][:2]])
                        linestring = shapely.affinity.rotate(linestring, -pano_yaw_deg, origin=(0, 0))
                        linestring = shapely.affinity.translate(linestring, pano.x, pano.y)
                        polyon = buffer_line(linestring, segment[2] / 2)
                        ax.plot(polyon.exterior.xy[0], polyon.exterior.xy[1])
                        print(linestring.xy)
                        print(list(zip(linestring.xy[0], linestring.xy[1])))
                        plt.axis('scaled')
                        plt.tight_layout()
                        plt.show()


            for segment in sidewalk_segments:
                is_nan = has_nan(segment)
                # if not math.isnan(segment[2]):
                if not is_nan:
                    linestring = LineString([segment[0][:2], segment[1][:2]])
                    linestring = shapely.affinity.rotate(linestring, -pano_yaw_deg, origin=(0, 0))
                    linestring = shapely.affinity.translate(linestring, pano.x, pano.y)
                    # polyon = buffer_line(linestring, segment[2] / 2)

                    line = {"slope_panoId": pano.panoId,
                            "width": segment[2],
                            "slope": 90.0 - float(pano.jdata['Projection']['tilt_yaw_deg']),
                            "distance_to_pano": abs(segment[0][0]),
                            # "width": linestring.xy[0][0],
                            "start_x": linestring.xy[0][0],
                            "start_y": linestring.xy[0][1],
                            "end_x": linestring.xy[1][0],
                            "end_y": linestring.xy[1][1],
                            # "linestring": linestring
                            }
                    lines.append(linestring)
                    df_sidewalk_seg = df_sidewalk_seg.append(line, ignore_index=True)

            # for m in medians:
            #     ax.scatter(m[0], m[1], s=m[2] * 200, marker='^')
            # ax.scatter(0, right_top)
            # ax.scatter(0, right_bottom)


            pano_apex_deg = get_pano_apex(transformer=transformer, pano=pano, json_dir=json_dir)

            print("pano_apex_deg, street_yaw:", pano_apex_deg, street_yaw)





        except Exception as e:
            # print("Error in sidewalk_connect():", e, meaure_file)
            logging.error("Error in sidewalk_connect(): %s, %s" % (e, meaure_file), exc_info=True)
            continue
    df_sidewalk_seg.to_csv(r'E:\USC_OneDrive\OneDrive - University of South Carolina\Research\sidewalk_wheelchair\results\sidewalks_edge_centerline.csv', index=False)
    gdf = gpd.GeoDataFrame(df_sidewalk_seg, geometry=lines)
    gdf.to_file(r'E:\USC_OneDrive\OneDrive - University of South Carolina\Research\sidewalk_wheelchair\results\sidewalks_edge_centerline.shp')

def has_nan(segment):
    a = (segment[2], 0)
    b = np.array([segment[0], segment[1], a])
    is_nan = np.isnan(b).any()
    return is_nan

def rotate_measurement_df(df, tx, ty, rotated_deg):
    points = np.array([df['col'], df['row']]).T
    # points = np.hstack((points, np.ones((points.shape[0], 1))))
    points = points_2D_translation_math(points, tx, ty)
    points = points_2D_rotated(points, rotated_deg)
    df['rotated_start_x'] = points[:, 0]
    df['rotated_start_y'] = points[:, 1]

    points = np.array([df['center_x'], df['center_y']]).T
    # points = np.hstack((points, np.ones((points.shape[0], 1))))
    points = points_2D_translation_math(points, tx, ty)
    points = points_2D_rotated(points, rotated_deg)
    df['rotated_center_x'] = points[:, 0]
    df['rotated_center_y'] = points[:, 1]

    points = np.array([df['end_x'], df['end_y']]).T
    # points = np.hstack((points, np.ones((points.shape[0], 1))))
    points = points_2D_translation_math(points, tx, ty)
    points = points_2D_rotated(points, rotated_deg)
    df['rotated_end_x'] = points[:, 0]
    df['rotated_end_y'] = points[:, 1]

    return df

def test1():
    # path
    path = r'ZXyk9lKhL5siKJglQPqfMA_DOM_0.05.tif'



    # Reading an image in default
    # mode
    image = cv2.imread(path)

    # Window name in which image is
    # displayed
    window_name = 'Image'

    # Polygon corner points coordinates
    pts = np.array([[25, 70], [25, 160],
                    [110, 200], [200, 160],
                    [200, 70], [110, 20]],
                   np.int32)

    pts = pts.reshape((-1, 1, 2))

    isClosed = True

    # Blue color in BGR
    color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 2

    # Using cv2.polylines() method
    # Draw a Blue polygon with
    # thickness of 1 px
    image = cv2.polylines(image, [pts],
                          isClosed, color, thickness)
# not finished
def get_pano_apex(transformer, panoId="", json_dir="", pano=None):

    Links = pano.jdata["Links"]
    if len(Links) < 2:
        # print(f"Error in draw_panorama_apex(): {pano.panoId} has no 2 panoramas in Links.")
        return None
    try:

        json_file_0 = os.path.join(json_dir, Links[0]['panoId'] + ".json")
        json_file_1 = os.path.join(json_dir, Links[1]['panoId'] + ".json")

        # print("json_file_0:", json_file_0)

        pano_0 = GSV_pano(json_file=json_file_0)
        pano_1 = GSV_pano(json_file=json_file_1)

        # print("pano_1.panoId:", pano_1.panoId)

        if (pano_1.panoId == 0) or (pano_0.panoId == 0):
            # print("Error in Links:")
            return None
        # pano_1 = GSV_pano(panoId=Links[1]['panoId'])

        # print("Line 532")

        # print("Line 572")
        xy = transformer.transform(pano.lat, pano.lon)
        xy0 = transformer.transform(pano_0.lat, pano_0.lon)
        xy1 = transformer.transform(pano_1.lat, pano_1.lon)
        pts = np.array([xy0, xy, xy1])
        # print("Line 577")

        # calculate angle
        a = (xy[1] - xy0[1], xy[0] - xy0[0])
        a = np.array(a)
        b = (xy[1] - xy1[1], xy[0] - xy1[0])
        b = np.array(b)
        angle = np.arccos(np.dot(a, b) / (LA.norm(a) * LA.norm(b)))
        angle_deg = np.degrees(angle)
        return angle_deg
    except Exception as e:
        print("Error in gt_pano_apex():", e)

def get_all_widths():
    DOM_dir = r'D:\Research\sidewalk_wheelchair\DC_DOMs'
    DOM_dir = r'H:\Research\sidewalk_wheelchair\DC_DOMs'
    img_list = glob.glob(os.path.join(DOM_dir, '*DOM*.tif'))
    # img_list = [r'D:\Research\sidewalk_wheelchair\DC_DOMs\XzB9K8BHqMpZVKZR-E9MBw_DOM_0.05.tif']
    # img_list = [r'D:\Research\sidewalk_wheelchair\DC_DOMs\Jk7bEuZo5fzWeax42a0bSw_DOM_0.05.tif']
    # img_list = [r'D:\Research\sidewalk_wheelchair\DC_DOMs\Jk091JHY-Fnhv_ho_1Qqmg_DOM_0.05.tif']
    # img_list = [r'D:\Research\sidewalk_wheelchair\DC_DOMs\JKABMj3d30_i7hL9T00JjA_DOM_0.05.tif']
    # img_list = [r'D:\Research\sidewalk_wheelchair\DC_DOMs\2vaa4vNnHbgayTqpJZvKrg_DOM_0.05.tif']
    # img_list = [r'H:\Research\sidewalk_wheelchair\DC_DOMs\2UzX00nEfTL8jXMfnhoTWw_DOM_0.05.tif']

    skip = 0

    img_list_mp = mp.Manager().list()
    for img in img_list[skip:]:
        img_list_mp.append(img)

    process_cnt = 10

    if (process_cnt == 1) or (len(img_list) < process_cnt * 3):

        cal_witdh_from_list(img_list)

    if process_cnt > 1:

        pool = mp.Pool(processes=process_cnt)

        for i in range(process_cnt):
            print(i)
            pool.apply_async(cal_witdh_from_list, args=(img_list_mp,))
        pool.close()
        pool.join()


def multi_process(func, args, process_cnt=6):
    print("Done")


def get_line(series):
    p = Point(series['col'], series['row'])
    p1 = Point(series['end_x'], series['end_y'])
    line = LineString([p1, p])
    return line

def measurements_to_shapefile(widths_files='', saved_path=''):


    # for idx, f in enumerate(widths_files):
    total_cnt = len(widths_files)
    while len(widths_files) > 0:
        f = widths_files.pop(0)
        processed_cnt = total_cnt -len(widths_files)
        try:
            if processed_cnt % 1000 == 0:
                print("Processing: ", processed_cnt, f)
            df = pd.read_csv(f)

            lines = df.apply(get_line, axis=1)

            gdf = gpd.GeoDataFrame(df, geometry=lines)
            basename = os.path.basename(f).replace(".csv", ".shp")
            new_name = os.path.join(saved_path, basename)
            gdf.to_file(new_name)
        except Exception as e:
            print("Error in measurements_to_shapefile():", e, f)
            continue


def measurements_to_shapefile_mp(width_dir='', saved_path=''):
    width_dir = r'D:\Research\sidewalk_wheelchair\DC_DOMs_measuremens2'
    saved_path = r'D:\Research\sidewalk_wheelchair\DC_DOMs_width_shapes'

    widths_files = glob.glob(os.path.join(width_dir, r'*.csv'))

    widths_files_mp = mp.Manager().list()

    for f in widths_files[10000:]:
        widths_files_mp.append(f)

    # measurements_to_shapefile(widths_files_mp, saved_path)

    process_cnt = 10
    pool = mp.Pool(processes=process_cnt)
    for i in range(process_cnt):
        pool.apply_async(measurements_to_shapefile, args=(widths_files_mp, saved_path))
    pool.close()
    pool.join()


def DOM_to_shapefile(DOM_list, class_idxs, saved_path):

    total = len(DOM_list)
    while len(DOM_list[:]) > 0:
        print(f"Processing {total - len(DOM_list)} / {total} files.")

        try:
            img_path = DOM_list.pop(0)
            basename = os.path.basename(img_path)
            dirname = os.path.dirname(img_path)
            panoId = basename[:22]

            img_pil = Image.open(img_path)
            img_np = np.array(img_pil)
            # im_cv = cv2.imread(img_path)
            target_ids = class_idxs

            class_idx = img_np
            target_np = np.zeros(img_np.shape)
            for i in target_ids:
                target_np = np.logical_or(target_np, class_idx == i)

            # AOI = np.where(AOI == 0, 0, 1).astype(np.uint8)

            morph_kernel_open = (5, 5)
            morph_kernel_close = (10, 10)
            g_close = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_close)
            g_open = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_open)

            target_np = target_np.astype(np.uint8)

            cv2_closed = cv2.morphologyEx(target_np, cv2.MORPH_CLOSE, g_close)  # fill small gaps
            cv2_opened = cv2.morphologyEx(cv2_closed, cv2.MORPH_OPEN, g_open)

            cv2_opened = np.where(cv2_opened == 0, 0, 255).astype(np.uint8)

            # raw_contours, hierarchy = cv2.findContours(cv2_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # read the world file
            worldfile_ext = img_path[-3] + img_path[-1] + 'w'
            worldfile_path = img_path[:-3] + worldfile_ext

            worldfile_lines = open(worldfile_path, 'r').readlines()
            worldfile_lines = [float(line[:-1] )for line in worldfile_lines]
            affinity_matrix = np.array(worldfile_lines)

            cv2_opened = np.where(cv2_opened == 0, 0, 1).astype(np.uint8)

            affinity_matrix_rasterio = np.array([affinity_matrix[0], affinity_matrix[2], \
                                                 affinity_matrix[4], affinity_matrix[1], \
                                                 affinity_matrix[3], affinity_matrix[5]])
            polygons = rasterio.features.shapes(cv2_opened, transform=affinity_matrix_rasterio)

            # polygons = binaryMask2Polygon(cv2_opened)
            shapely_polygons = []
            polygon_cnt = 0
            for idx, polygon in enumerate(polygons):
                # using rasterio
                if polygon[1] == 1:
                    shapely_polygon = shapely.geometry.shape(polygon[0])
                    shapely_polygon = shapely_polygon.simplify(0.05)
                    shapely_polygons.append(shapely_polygon)
                    polygon_cnt += 1


                # using skimage
                # # print(type(polygon), polygon.size)
                # if polygon.size > 7:
                #     polygon = Polygon(np.squeeze(polygon))
                #     polygon = shapely.affinity.affine_transform(polygon, affinity_matrix)
                #     shapely_polygons.append(polygon)

            # print(polygon)
            df = pd.DataFrame([panoId] * polygon_cnt)
            df.columns = ['panoId']
            gdf = gpd.GeoDataFrame(df, geometry=shapely_polygons)
            # print(gdf)
            new_name = os.path.join(saved_path, basename.replace(".tif", ".shp"))
            gdf.to_file(new_name)
        except Exception as e:
            print(e)
            logging.error("Error in processing %s, %s" % (img_path, str(e)))

            continue

def DOM_to_shapefile_mp(DOM_dir, class_idxs, saved_path):

    DOM_files = glob.glob(os.path.join(DOM_dir, "*_DOM_0.05.tif"))
    DOM_to_shapefile(DOM_files, class_idxs, saved_path)

    DOM_files_mp = mp.Manager().list()
    for file in DOM_files[4000:]:
        DOM_files_mp.append(file)

    process_cnt = 10
    pool = mp.Pool(processes=process_cnt)
    for i in range(process_cnt):
        pool.apply_async(DOM_to_shapefile, args=(DOM_files_mp, class_idxs, saved_path))
    pool.close()
    pool.join()


def binaryMask2Polygon( binaryMask):
    polygons =[]

    padded_binary_mask = np.pad(binaryMask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)

    def closeContour(contour):
        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack((contour, contour[0]))
        return contour

    for contour in contours:
        contour = closeContour(contour)
        contour = measure.approximate_polygon(contour, 1)

        if len(contour)<3:
            continue
        contour = np.flip(contour, axis=1)
        # segmentation = contour.ravel().tolist()
        #
        # # after padding and subtracting 1 we may get -0.5 points in our segmentation
        # segmentation = [0 if i < 0 else i for i in segmentation]
        # polygons.append(segmentation)
        polygons.append(contour)
    return polygons

def merge_shp(shp_dir):
    files = glob.glob(os.path.join(shp_dir, "*.shp"))
    gdf_list = []
    for idx, file in tqdm(enumerate(files)):
        try:
            gdf = gpd.read_file(file)
            gdf_list.append(gdf)
        except Exception as e:
            print("Error: ", str(e))
            continue

    print("Concatinng gdfs...")

    all_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs=gdf_list[0].crs)
    all_gdf.to_file(r'H:\Research\sidewalk_wheelchair\DC_DOMs_shapefile_merged.shp')

    print("Finished.")

def split_DOM_by_points(img_file, shp_file, buffer_distance, saved_path, name_column, img_ext=".tif"):
    # note image and shp file should have the same projection!
    # example: gdal_translate -projwin 406935.7972999995 135400.57989997734 406975.84730000043 135360.52989997427 -of GTiff "E:/USC_OneDrive/OneDrive - University of South Carolina/Research/sidewalk_wheelchair/DC_sidewalk_raster/sidwalk_dc.img" H:/OUTPUT.tif
    if os.path.exists(img_file) and os.path.exists(shp_file) and os.path.exists(saved_path):
        gdf = gpd.read_file(shp_file)
        for idx, row in tqdm(enumerate(gdf.iterrows())):
            # print("idx:", idx)
            # print("row:\n", row)
            basename = str(row[1][name_column]) + img_ext
            x, y = row[1].geometry.xy
            left = x[0] - buffer_distance
            right = x[0] + buffer_distance
            top = y[0] + buffer_distance
            bottom = y[0] - buffer_distance
            saved_file = os.path.join(saved_path, basename)
            gdal_cmd = f'gdal_translate -projwin {left} {top} {right} {bottom} -of GTiff "{img_file}" "{saved_file}" -co "TFW=YES" -co compress=lzw'
            result = subprocess.run(gdal_cmd, stdout=subprocess.PIPE)
            print(result.stdout)


if __name__ == "__main__":
    # test1()
    # cal_witdh()
    # get_all_widths()
    # sidewalk_connect()
    # measurements_to_shapefile_mp()
    # img_path = r'ZXyk9lKhL5siKJglQPqfMA_DOM_0.05.tif'

    # split_DOM_by_points(img_file=r'E:/USC_OneDrive/OneDrive - University of South Carolina/Research/sidewalk_wheelchair/DC_sidewalk_raster/sidwalk_dc.img', \
    #                     shp_file=r'E:\USC_OneDrive\OneDrive - University of South Carolina\Research\sidewalk_wheelchair\vectors\panoramas_CRS6487.shp', \
    #                     buffer_distance=20, \
    #                     saved_path=r'H:\Research\sidewalk_wheelchairs\DC_sidewalk_clip_tiles2', \
    #                     name_column='panoId')

    split_DOM_by_points(img_file=r'E:/USC_OneDrive/OneDrive - University of South Carolina/Research/sidewalk_wheelchair/DC_sidewalk_raster/sidwalk_dc.img', \
                        shp_file=r'E:\USC_OneDrive\OneDrive - University of South Carolina\Research\sidewalk_wheelchair\vectors\DC_Roadway_Block-shp\Roadway_Block_EPSG6487_mid_point_50m.shp', \
                        buffer_distance=25, \
                        saved_path=r'H:\Research\sidewalk_wheelchairs\DC_road_split_tiles', \
                        name_column='ORIG_FID')

    # cal_witdh_from_list([])

    # get_centerline_from_img(img_path)

    # DOM_to_shapefile_mp(DOM_dir=r"H:\Research\sidewalk_wheelchair\DC_DOMs", class_idxs=[8, 12], saved_path=r"H:\Research\sidewalk_wheelchair\DC_DOMs_shapefile")
    # merge_shp(shp_dir=r'H:\Research\sidewalk_wheelchair\DC_DOMs_shapefile')