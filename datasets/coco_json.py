import os
import json
import torch
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset

class CocoDataset(Dataset):
    def __init__(self, images_dir, annotations_file, transforms=None):

        self.images_dir = images_dir
        self.transforms = transforms

        with open(annotations_file, 'r') as f:
            self.coco_data = json.load(f)

        self.image_id_to_filename = {img["id"]: img["file_name"] for img in self.coco_data["images"]}

        self.annotations = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            bbox = ann["bbox"]
            category_id = ann["category_id"]

            x_min, y_min, w, h = bbox
            x_max = x_min + w
            y_max = y_min + h

            if img_id not in self.annotations:
                self.annotations[img_id] = {"boxes": [], "labels": []}

            self.annotations[img_id]["boxes"].append([x_min, y_min, x_max, y_max])
            self.annotations[img_id]["labels"].append(category_id)

        self.image_ids = list(self.image_id_to_filename.keys()) 

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_name = self.image_id_to_filename[img_id]
        img_path = os.path.join(self.images_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        image = T.ToTensor()(image)

        boxes = self.annotations.get(img_id, {}).get("boxes", [])
        labels = self.annotations.get(img_id, {}).get("labels", [])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        return image, target

def collate_fn(batch):
    images, targets = [], []
    for img, target in batch:
        images.append(img)

        # Обрабатываем случай, когда нет объектов
        if len(target["boxes"]) == 0:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)

        targets.append(target)

    return torch.stack(images), targets

