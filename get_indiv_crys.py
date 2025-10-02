import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

image_folder = r"path to folder with full images, like the 1920x1080px"
masks_folder = r"path to folder with the segmentation masks, downloaded from CVAT"
save_cropped_images_folder = r"path to save individual crystal images"
save_cropped_masks_folder = r"path to save individual crystal masks"
# save_cropped_masks_folder is used if you want to retrain the model to measure 
# crystal sizes and these images are used to train the "size measuring" model (Segmentation Model).
# You can ignore "save_cropped_masks_folder" if you want.

images_list = sorted(os.listdir(image_folder))
masks_list = sorted(os.listdir(masks_folder))
os.makedirs(save_cropped_images_folder, exist_ok=True)
os.makedirs(save_cropped_masks_folder, exist_ok=True)
print(len(masks_list))
for i in range(len(masks_list)):
    orig_img = Image.open(image_folder + "/" + images_list[i]).convert("RGB")
    mask = Image.open(masks_folder + "/" + masks_list[i]).convert("L")

    mask = np.array(mask)
    obj_ids = np.unique(mask)
    obj_ids = obj_ids[1:]
    num_objs = len(obj_ids)
    masks = np.zeros((num_objs , mask.shape[0] , mask.shape[1] ))
    for k in range(num_objs):
        masks[k] = np.where(mask == obj_ids[k], True, 0)

    boxes = []
    for j in range(num_objs):
        pos = np.where(masks[j])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        if xmin == xmax:
            xmin += -1
        if ymin == ymax:
            ymin += -1
        
        print((xmin, ymin, xmax, ymax))
        mask = Image.fromarray(np.uint8(masks[j])*255)
        cropped_mask = mask.crop((xmin, ymin, xmax, ymax))
        cropped_mask.save(save_cropped_masks_folder + "/" + images_list[i] + "_mask_" + str(j) + ".png")

        orig_img_crop = orig_img.crop((xmin, ymin, xmax, ymax))
        orig_img_crop.save(save_cropped_images_folder + "/" + images_list[i] + "_mask_" + str(j) + ".png")

        boxes.append([xmin , ymin , xmax , ymax])
        
        print("Image: " + str(i) + " Mask: " + str(j))
        #print("Boxes: " + str(boxes))