from torchvision.io.image import read_image


from torchvision.models import resnet50, ResNet50_Weights
from model import Test_model
import torch

import utils
if __name__ == "__main__":
    image_list = utils.slice_input('./images/dog.jpg')
    model = Test_model.Model()
    output_filenames = ['output_top_left.png', 'output_top_right.png', 'output_bottom_left.png', 'output_bottom_right.png']
    for image, output_filename in zip(image_list, output_filenames):
        output = model.inference(image)
        # output.save('./output_images/' + output_filename)