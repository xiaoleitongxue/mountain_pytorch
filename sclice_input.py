import cv2
import numpy as np


def slice_input(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    image_top_left = np.zeros(image.shape)
    image_top_right = np.zeros(image.shape)
    image_bottom_left = np.zeros(image.shape)
    image_bottom_right = np.zeros(image.shape)
    # 获取图像的高度和宽度
    height, width = image.shape[:2]

    # 计算中心点坐标
    center_x = width // 2
    center_y = height // 2

    # 将图像切分成四个部分
    top_left = image[0:center_y, 0:center_x]
    image_top_left[0:top_left.shape[0], 0:top_left.shape[1]] = top_left

    top_right = image[0:center_y, center_x:width]
    image_top_right[0:top_right.shape[0], 0:top_right.shape[1]] = top_right

    bottom_left = image[center_y:height, 0:center_x]
    image_bottom_left[0:bottom_left.shape[0], 0:bottom_left.shape[1]] = bottom_left

    bottom_right = image[center_y:height, center_x:width]
    image_bottom_right[0:bottom_right.shape[0], 0:bottom_right.shape[1]] = bottom_right

    image_list = []
    image_list.append(image_top_left)
    image_list.append(image_top_right)
    image_list.append(image_bottom_left)
    image_list.append(image_bottom_right)
    return image_list
    # # 显示切分后的图像
    # cv2.imshow('Top Left', top_left)
    # cv2.imshow('Top Right', top_right)
    # cv2.imshow('Bottom Left', bottom_left)
    # cv2.imshow('Bottom Right', bottom_right)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
