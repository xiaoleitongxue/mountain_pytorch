from torchvision.io.image import read_image
from torchvision.io.image import write_png
from torchvision.io.image import write_file
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import torch
from PIL import Image
import PIL
class Model():

    def __init__(self) -> None:
        self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=self.weights, box_score_thresh=0.9)
        self.model.eval()

    def inference(self, img) -> torch.Tensor:
        # Step 2: Initialize the inference transforms
        preprocess = self.weights.transforms()

        # Step 3: Apply inference preprocessing transforms
        batch = [preprocess(img)]

        # Step 4: Use the model and visualize the prediction
        prediction = self.model(batch)[0]
        labels = [self.weights.meta["categories"][i]
                  for i in prediction["labels"]]
        box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                                  labels=labels,
                                  colors="red",
                                  width=4, font_size=30)
        # im = to_pil_image(box.detach())
        return box
        
