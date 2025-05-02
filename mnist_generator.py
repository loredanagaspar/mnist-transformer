import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import datasets
import torchvision.transforms as transforms
import os
import random
os.makedirs("samples", exist_ok=True)

label_to_index = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "<start>": 10,
    "<end>": 11,
    "<pad>": 12
}

index_to_label = {v: k for k, v in label_to_index.items()}


class MNISTDataset(Dataset):
    def __init__(self, split="train"):
        self.dataset = datasets.load_dataset("ylecun/mnist")[split]
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.transform(self.dataset[idx]["image"]), torch.tensor(self.dataset[idx]["label"], dtype=torch.long)


class TiledMNISTDataset(Dataset):
    def __init__(self, split="train", tile_size=4, allow_blanks=True):
        self.dataset = datasets.load_dataset("ylecun/mnist")[split]
        self.tile_size = tile_size  # fixed 4 for 3 x 2 grid
        self.transform = transforms.ToTensor()
        self.allow_blanks = allow_blanks

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        num_digits = self.tile_size if not self.allow_blanks else torch.randint(
            0, self.tile_size + 1, (1,)).item()
        # Sample real digits
        indices = torch.randint(0, len(self.dataset), (num_digits,)).tolist()
        images = [self.transform(self.dataset[i]["image"]) for i in indices]
        labels = [self.dataset[i]["label"] for i in indices]

        # Add blank images if needed
        blank = torch.zeros((1, 28, 28))
        while len(images) < self.tile_size:
            images.append(blank)
            labels.append(None)  # Use None for blanks

        # Shuffle so blanks aren't always at the end
        combined = list(zip(images, labels))
        random.shuffle(combined)
        images, labels = zip(*combined)

        # Create 2x2 tiled image
        row1 = torch.cat(images[:2], dim=2)  # concat width-wise
        row2 = torch.cat(images[2:], dim=2)
        tile = torch.cat([row1, row2], dim=1)  # concat height-wise

        # Create input and target sequences with tokens
        in_seq = [label_to_index["<start>"]] + \
            [label_to_index[str(l)] for l in labels if l is not None]
        out_seq = [label_to_index[str(
            l)] for l in labels if l is not None] + [label_to_index["<end>"]]

        return tile, torch.tensor(in_seq, dtype=torch.long), torch.tensor(out_seq, dtype=torch.long)


class ScatteredMNISTDataset(Dataset):
    def __init__(self, split="train", max_n=4):
        self.dataset = datasets.load_dataset("ylecun/mnist")[split]
        self.max_n = max_n
        self.transform = transforms.ToTensor()
        self.digit_size = 28  # MNIST digits are 28x28
        self.canvas_size = 128

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        n = torch.randint(0, self.max_n + 1, (1,)).item()

        canvas = torch.zeros((1, self.canvas_size, self.canvas_size))

        positions = []
        labels = []

        # Create a pool to randomly pick positions from (top right corner of the digit)
        trimmed_canvas_size = self.canvas_size - self.digit_size + 1
        pixel_pool = torch.arange(trimmed_canvas_size ** 2)
        # Create a mask to exclude positions that are already occupied
        mask = torch.ones((trimmed_canvas_size ** 2), dtype=torch.bool)
        for _ in range(n):
            idx = torch.randint(0, len(self.dataset), (1,)).item()
            # Randomly pick a position from the pool
            masked_pool = pixel_pool.masked_select(mask)
            pos = torch.randint(0, masked_pool.shape[0], (1,)).item()
            pos = masked_pool[pos]
            x = pos % trimmed_canvas_size
            y = pos // trimmed_canvas_size
            # Place the digit on the canvas
            canvas[:, y:y+self.digit_size, x:x +
                   self.digit_size] = self.transform(self.dataset[idx]["image"])
            positions.append((x, y))
            labels.append(self.dataset[idx]["label"])
            # Update the mask to exclude all the positions that are occupied
            mask.view((trimmed_canvas_size, trimmed_canvas_size))[y:min(
                y+self.digit_size, trimmed_canvas_size), x:min(x+self.digit_size, trimmed_canvas_size)] = False

        # Create input and target sequences
        labels = [label_to_index["<start>"]] + \
            sorted(labels) + [label_to_index["<end>"]]
        labels = labels + [label_to_index["<pad>"]] * \
            max(self.max_n + 2 - len(labels), 0)
        input_seq = torch.tensor(labels[:self.max_n+1], dtype=torch.long)
        target_seq = torch.tensor(labels[1:], dtype=torch.long)

        return canvas, input_seq, target_seq


def collate_fn(batch):
    MAX_TOKENS = 6
    images, input_seqs, target_seqs = zip(*batch)

    images = torch.stack(images)

    input_seqs = pad_sequence(
        input_seqs, batch_first=True, padding_value=label_to_index["<pad>"])
    target_seqs = pad_sequence(
        target_seqs, batch_first=True, padding_value=label_to_index["<pad>"])

    # Optional: Clip or pad to MAX_TOKENS
    input_seqs = input_seqs[:, :MAX_TOKENS]
    target_seqs = target_seqs[:, :MAX_TOKENS]

    return images, input_seqs, target_seqs


if __name__ == "__main__":
    import os
    from torchvision.transforms.functional import to_pil_image
    # torch.manual_seed(42)
    # dataset = MNISTDataset()
    # print(dataset[0])
    # tiled_dataset = TiledMNISTDataset()
    # sample = tiled_dataset[0]
    # print(sample[0].shape)
    # print(sample[1])
    # print(sample[2])
    # assert (sample[1][1:] == sample[2][:-1]).all()
    os.makedirs("samples", exist_ok=True)
    torch.manual_seed(42)
    scattered_dataset = ScatteredMNISTDataset()
    for i in range(1000):
        sample = scattered_dataset[i]
        print(f"sample {i}: {sample[1]}")
        assert sample[1].shape == sample[2].shape == (5,)
        assert (sample[1][1:] == sample[2][:-1]).all()
        to_pil_image(sample[0]).save(f"samples/scattered_sample_{i}.png")
