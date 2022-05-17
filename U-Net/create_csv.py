import csv
import os


train_images = "datasets/Water Bodies/train/Images"
train_masks = "datasets/Water Bodies/train/Masks"
train_csv = "datasets/Water Bodies/train/train_csv.csv"

val_images = "datasets/Water Bodies/val/Images"
val_masks = "datasets/Water Bodies/val/Masks"
val_csv = "datasets/Water Bodies/val/val_csv.csv"


def create_csv(csv_file, images, masks):
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Index', 'Image', 'Mask'])
        for (root, dirs, files) in os.walk(images, topdown=True):
            for name in files:
                index = name[11:-4]
                image = os.path.join(root, name)
                mask = os.path.join(masks, name)
                writer.writerow([index, image, mask])


if __name__ == "__main__":
    create_csv(train_csv, train_images, train_masks)
