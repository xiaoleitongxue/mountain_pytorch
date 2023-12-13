
import model
import torch
import io
def inference(model: model.Test_model.Model, img):
    output = model.inference(img)
    print(output)
    return output

class Coordinate:
    x = 0
    y = 0

    def __init__(self, x_, y_):
        x = x_
        y = y_

def local_coordinates2global_coordinates(offset_x, offset_y, coordinate):
    coordinate.x = coordinate.x + offset_x
    coordinate.y = coordinate.y + offset_y
    return coordinate


def batch_transformer(coordinates):
    list = []
    for coordinate in coordinates:
        transed_coor = local_coordinates2global_coordinates(offset_x, offset_y, coordinate)
        list.append(transed_coor)
    return list


import cv2
import numpy as np
from torchvision.io.image import read_image
import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.io.image import write_png


def slice_input(image_path) -> []:
    # 读取图像
    image = read_image(image_path)
    image_top_left = torch.zeros_like(image)
    image_top_right = torch.zeros_like(image)
    image_bottom_left = torch.zeros_like(image)
    image_bottom_right = torch.zeros_like(image)
    # 获取图像的高度和宽度
    height, width = image.shape[1:3]

    # 计算中心点坐标
    center_x = width // 2
    center_y = height // 2

    # 将图像切分成四个部分
    top_left = image[:, 0:center_y, 0:center_x]
    image_top_left[:, 0:top_left.shape[1], 0:top_left.shape[2]] = top_left

    top_right = image[:, 0:center_y, center_x:width]
    image_top_right[:, 0:top_right.shape[1], 0:top_right.shape[2]] = top_right

    bottom_left = image[:, center_y:height, 0:center_x]
    image_bottom_left[:, 0:bottom_left.shape[1],
                      0:bottom_left.shape[2]] = bottom_left

    bottom_right = image[:, center_y:height, center_x:width]
    image_bottom_right[:, 0:bottom_right.shape[1],
                       0:bottom_right.shape[2]] = bottom_right

    # save image to file
    write_png(input=image_top_left,
              filename='./output_images/image_top_left.png', compression_level=0)
    write_png(input=image_top_right,
              filename='./output_images/image_top_right.png', compression_level=0)
    write_png(input=image_bottom_left,
              filename='./output_images/image_bottom_left.png', compression_level=0)
    write_png(input=image_bottom_right,
              filename='./output_images/image_bottom_right.png', compression_level=0)

    image_list = []
    image_list.append(image_top_left)
    image_list.append(image_top_right)
    image_list.append(image_bottom_left)
    image_list.append(image_bottom_right)

    return image_list


def tensor2bytes(tensor : torch.Tensor) -> bytes:
    stream = io.BytesIO()
    torch.save(tensor, stream)
    return stream.getvalue()


def bytes2tensor(b : bytes) -> torch.Tensor:
    stream = io.BytesIO(b)
    return torch.load(stream)