import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class YOLODataset(Dataset):
    def __init__(self, images_dir, labels_dir, transforms=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms
        self.images = [img for img in os.listdir(images_dir) if img.endswith(('jpg', 'jpeg', 'png'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        
        image = Image.open(img_path).convert("RGB")
        image = T.ToTensor()(image)
        height, width = image.shape[:2]

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                parts = line.strip().split()
                for line in f:
                    if len(parts) != 5:
                        print(f"Warning: incorrect label format in {label_path}: {line.strip()}")
                        continue
                    cls, x_center, y_center, w, h = map(float, line.strip().split())

                    x_min = (x_center - w / 2) * width
                    y_min = (y_center - h / 2) * height
                    x_max = (x_center + w / 2) * width
                    y_max = (y_center + h / 2) * height

                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(int(cls))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels}

        if self.transforms:
            transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            target['boxes'] = torch.tensor(transformed['bboxes'], dtype=torch.float32)

        return image, target