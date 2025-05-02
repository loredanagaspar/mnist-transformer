import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import wandb
import os
from time import time
from itertools import islice
from tqdm import tqdm

from model import TransformerMNIST, greedy_decode
from mnist_generator import TiledMNISTDataset, label_to_index, index_to_label, collate_fn


if __name__ == "__main__":

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TiledMNISTDataset(split="train")
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

    test_dataset = TiledMNISTDataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=64,
                             shuffle=False, collate_fn=collate_fn)

    BATCH_SIZE = 64
    NUM_EPOCHS = 5
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 0.1
    LEARNING_RATE_DECAY = 0.9
    train = True
    CONTEXT_SIZE = 6
    VOCAB_SIZE = 13

    model = TransformerMNIST(vocab_size=VOCAB_SIZE).to(device)
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=LEARNING_RATE_DECAY)

    wandb.init(project="mnist-transformer", config={
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "lr_decay": LEARNING_RATE_DECAY,
        "vocab_size": VOCAB_SIZE,
        "device": str(device)
    })

start = time()
for epoch in range(NUM_EPOCHS):
    model.train()
    total_train_loss = 0

    for images, input_seq, target_seq in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
        images, input_seq, target_seq = images.to(
            device), input_seq.to(device), target_seq.to(device)

        logits = model(images, input_seq)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(
            B * T, V), target_seq.view(B * T), ignore_index=label_to_index["<pad>"], label_smoothing=0.1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # === Eval ===
    model.eval()
    total_test_loss = 0
    total_test_correct = 0
    total_test_tokens = 0

    with torch.no_grad():
        for images, input_seq, target_seq in tqdm(test_loader, desc=f"Epoch {epoch+1} Training"):
            images, input_seq, target_seq = images.to(
                device), input_seq.to(device), target_seq.to(device)

            logits = model(images, input_seq)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(
                B * T, V), target_seq.view(B * T), ignore_index=label_to_index["<pad>"], label_smoothing=0.1)
            total_test_loss += loss.item()

            preds = logits.argmax(dim=2)
            mask = target_seq != label_to_index["<pad>"]
            correct = ((preds == target_seq) & mask).float().sum().item()
            total_test_correct += correct
            total_test_tokens += mask.sum().item()

    avg_test_loss = total_test_loss / len(test_loader)
    test_accuracy = total_test_correct / total_test_tokens

    wandb.log({
        "train_loss": avg_train_loss,
        "test_loss": avg_test_loss,
        "test_accuracy": test_accuracy
    })

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Acc: {test_accuracy:.4f}")

    # Save model checkpoint
    model_path = f"model/{wandb.run.name}/transformer_epoch{epoch}.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)

    # Greedy decode preview
    test_img, _, test_target = test_dataset[0]
    pred_tokens = greedy_decode(model, test_img.to(device))
    pred_labels = [index_to_label[t] for t in pred_tokens]
    target_labels = [index_to_label[t.item()] for t in test_target]

    plt.imshow(to_pil_image(test_img), cmap="gray")
    plt.axis("off")
    plt.title(
        f"Pred: {' '.join(pred_labels)}\nTrue: {' '.join(target_labels)}")
    plt.show()

    scheduler.step()

print(f"Training complete in {round(time() - start)}s")
