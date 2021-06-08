import cv2
import numpy as np

def cal_witdh():
    img_path = r'AZK1jDGIZC1zmuooSZCzEg_DOM.tif'
    target_ids = [244]
    im_cv = cv2.imread(img_path)

    red_channel = im_cv[:, :, 2]
    class_idx = red_channel
    AOI = np.zeros((len(class_idx), len(class_idx)))
    for i in target_ids:
        AOI = np.logical_or(AOI, class_idx == i)
        print(AOI)
    AOI = np.where(AOI == 0, 0, 255)
    print(AOI)
    # AOI = cv2.cvtColor(AOI, cv2.COLOR)
    cv2.imshow("Raw image", AOI.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cal_witdh()
