import os
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
import numpy as np


ori_val = "datasets/Water Bodies/val/Images"
ori_mask = "datasets/Water Bodies/val/Masks"
save_pth = "Paper Implementation/U-Net/saved_img"
mask_pth = os.path.join(save_pth, "mask")
pred_pth = os.path.join(save_pth, "pred")
val_pth = os.path.join(save_pth, "val")

transform = A.Compose([
    A.Resize(300, 300),
    A.CenterCrop(256, 256)
])


def save_crop(scan_pth, save_name):
    idx = 0
    for (root, dirs, files) in os.walk(scan_pth, topdown=True):
        for name in files:
            image_pth = os.path.join(root, name)
            s_pth = os.path.join(save_pth, save_name, str(save_name) + "_" + str(idx) + ".jpg")
            im = np.array(Image.open(image_pth))
            al = transform(image=im)
            im = al["image"]
            im = Image.fromarray(im)
            im.save(s_pth)
            idx += 1


n_examples = 5


def show_ex(n_examples):
    fig, axs = plt.subplots(n_examples, 3, figsize=(14, n_examples * 7), constrained_layout=True)
    for index in range(n_examples):
        pred_img = os.path.join(pred_pth, "pred_" + str(index) + ".jpg")
        mask_img = os.path.join(mask_pth, "mask_" + str(index) + ".jpg")
        val_img = os.path.join(val_pth, "val_" + str(index) + ".jpg")
        pred = Image.open(pred_img)
        mask = Image.open(mask_img)
        val = Image.open(val_img)
        axs[index, 0].imshow(val)
        axs[index, 0].axis('off')
        axs[index, 0].set_title("Satellite Image")
        axs[index, 1].imshow(mask)
        axs[index, 1].axis('off')
        axs[index, 1].set_title("Ground Truth")
        axs[index, 2].imshow(pred)
        axs[index, 2].axis('off')
        axs[index, 2].set_title("Predicted")

    plt.subplot_tool()
    plt.show()


# save_crop(ori_mask, "mask")
show_ex(n_examples)
