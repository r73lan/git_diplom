import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_yolo(model, dataset, optimizer, custom_loss_fn, device='cuda', epochs=10, batch_size=16):
    model.to(device)
    model.train()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, targets in pbar:
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            preds = model(images)

            loss = custom_loss_fn(preds, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")


def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)
    return images, targets
