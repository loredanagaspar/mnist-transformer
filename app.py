from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import torch
import torchvision.transforms as transforms
import base64
import re
from model import TransformerMNIST, greedy_decode, label_to_index, index_to_label
from PIL import ImageFilter, ImageOps
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from mnist_generator import TiledMNISTDataset

app = FastAPI()


@app.get("/")
def root():
    return {"message": "MNIST Transformer API is running!"}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow frontend dev
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerMNIST(vocab_size=13).to(device)
model_path = "model/robust-spaceship-6/transformer_epoch4.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Inference endpoint


class ImageRequest(BaseModel):
    image_base64: str


@app.post("/predict/")
def predict(req: ImageRequest):
    # Convert base64 to PIL image
    image_data = re.sub('^data:image/.+;base64,', '', req.image_base64)
    image = Image.open(BytesIO(base64.b64decode(image_data))).convert("L")
    image = image.resize((56, 56))
    image = image.filter(ImageFilter.GaussianBlur(radius=1))

    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to tensor and scales to [0,1]
        # transforms.Normalize((0.5,), (0.5,))  # If you used this during training
    ])
    image_tensor = transform(image).to(device)  # (1, 56, 56)

    # Visualize what the model sees
    plt.imshow(image_tensor.squeeze().cpu().numpy(), cmap="gray")
    plt.title("What the model sees")
    plt.show()

    # Predict
    pred_tokens = greedy_decode(model, image_tensor, max_len=6)
    pred_labels = [index_to_label[t] for t in pred_tokens]

    return {"prediction": pred_labels}


@app.get("/test-tiles")
def test_tiled_mnist():
    tiled = TiledMNISTDataset(split="test", allow_blanks=True)
    image, input_seq, target_seq = tiled[0]

    image = transforms.Resize((56, 56))(image).to(device)

    pred_tokens = greedy_decode(model, image, max_len=6)
    pred_labels = [index_to_label[t] for t in pred_tokens]
    true_labels = [index_to_label[t.item()] for t in target_seq]

    return {
        "predicted_labels": pred_labels,
        "true_labels": true_labels
    }
