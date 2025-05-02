import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from time import time
from mnist_generator import TiledMNISTDataset, label_to_index, index_to_label

# === Token dictionary ===
label_to_index = {
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4,
    "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "<start>": 10, "<end>": 11, "<pad>": 12
}
index_to_label = {v: k for k, v in label_to_index.items()}

# === Constants ===
PATCH_SIZE = 14
NUM_PATCHES = 16
N_EMBD = 64
NUM_HEADS = 4
NUM_BLOCKS = 4
DROPOUT = 0.1
CONTEXT_SIZE = 6
CHANNELS = 1  # grayscale 1 channel

# === Model Components (PatchEmbedding, Attention, MLP etc.) ===

# - PatchEmbedding


class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        # example Input of Raw MNIST images (Batch_size=32, Chanel=1, W=56, h=56)
        self._conv2d = nn.Conv2d(
            in_channels=CHANNELS,  # channel=1
            out_channels=N_EMBD,  # n_embd=64 - the dimensional embedding vector
            kernel_size=PATCH_SIZE,  # defines the size of each patch 14x14
            # moves the kernel by 14 pixels = no overlap between patches
            stride=PATCH_SIZE, padding=0)
        # after conv2d each of 16 patches (4x4gid) mapped to 64 features (nr of embeddings)=> (Batch size=32, Emb=64, 4, 4)
        # Flattens [B, 64, 4, 4] → [B, 64, 16]
        self._flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):
        x = self._conv2d(x)  # (B, 1,56,56)->(B,64,4,4)
        x = self._flatten(x)  # -> (B, 64, 16)
        # after permuting the (32,64,4,4) we have (32=Batch,16=patches,64=embedning vector in embedding space) = > Final ViT input: token sequence
        return x.permute(0, 2, 1)  # (B, 16, 16)

# - AttentionHead


class AttentionHead(nn.Module):  # Head
    """ one head of self-attention """
    # Constructor

    def __init__(self, head_size, is_decoder):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(  # trils is a lower triangular matrix used for masking future tokens in decoder self-attebtion, not needed in Vit or encoder
            torch.ones(CONTEXT_SIZE, CONTEXT_SIZE)))  # shape is T, T
        self.dropout = nn.Dropout(DROPOUT)
        self._is_decoder = is_decoder

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # SHAPE(B T C)
        q = self.query(x)  # SHAPE(B T C)s
        wei = q @ k.transpose(-2, -1) * C**-0.5
        if self._is_decoder:  # only in decoder blocks
            wei = wei.masked_fill(
                self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out  # (B, T, head_size)

# - MultiHeadAttention
# 4 heads where the head size per head is 16 (Embed dim/num of heads)


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, head_size, is_decoder=True):
        super().__init__()

        self.heads = nn.ModuleList(
            [AttentionHead(head_size, is_decoder) for _ in range(NUM_HEADS)])
        self.proj = nn.Linear(head_size * NUM_HEADS, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):

        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

# - CrossAttentionHead


class CrossAttentionHead(nn.Module):
    """ One head of cross-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, query_input, encoder_output):
        """
        query_input: shape (B, T_dec, N_EMBD) ← decoder input
        encoder_output: shape (B, T_enc, N_EMBD) ← encoder output (patch tokens)
        """
        B, T_dec, _ = query_input.shape
        T_enc = encoder_output.size(1)

        k = self.key(encoder_output)       # (B, T_enc, head_size)
        q = self.query(query_input)        # (B, T_dec, head_size)
        v = self.value(encoder_output)     # (B, T_enc, head_size)

        # Attention scores
        wei = q @ k.transpose(-2, -1) * (k.size(-1)
                                         ** -0.5)  # (B, T_dec, T_enc)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # Weighted sum of values
        out = wei @ v  # (B, T_dec, head_size)
        return out

# - MultiHeadCrossAttention


class MultiHeadCrossAttention(nn.Module):
    """ Multi-head cross-attention for decoder """

    def __init__(self, head_size):
        super().__init__()
        self.heads = nn.ModuleList([
            CrossAttentionHead(head_size) for _ in range(NUM_HEADS)
        ])
        self.proj = nn.Linear(head_size * NUM_HEADS, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, query_input, encoder_output):
        # Each head processes the inputs in parallel
        out = torch.cat([h(query_input, encoder_output)
                        for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

# - MLP


class MLP(nn.Module):  # FeedForward where clasification takes place
    """ a simple linear layer followed by a non-linearity """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # first fully connected layer. Expands the feature space (eg from 64 to 256) allowing the network to learn more complex intermediate represenations
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),  # ViT paper uses GELU # Activation fuctions -Adds non-linearity
            # Projects back down to original embedding dim eg 256 ->64
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        # passes the input x of shape B x T x N-Embed through the MLP block. Output has the same shape
        return self.net(x)

# - SelfAttentionBlock


class SelfAttentionBlock(nn.Module):
    def __init__(self, is_decoder=False):
        super().__init__()
        head_size = N_EMBD//NUM_HEADS
        self.attn = MultiHeadAttention(head_size, is_decoder=is_decoder)
        self.mlp = MLP()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.ln2 = nn.LayerNorm(N_EMBD)

    def forward(self, x):
        x = x+self.attn(self.ln1(x))
        x = x+self.mlp(self.ln2(x))
        return x

# - CrossAttentionBlock


class CrossAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = N_EMBD // NUM_HEADS
        self.cross_attn = MultiHeadCrossAttention(head_size)
        self.mlp = MLP()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.ln2 = nn.LayerNorm(N_EMBD)

    def forward(self, x, encoder_output):
        x = x + self.cross_attn(self.ln1(x), encoder_output)
        x = x + self.mlp(self.ln2(x))
        return x

# - TransformerEncoder


class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbedding()  # outputs (B, 16, 64)
        self.pos_embed = nn.Parameter(torch.randn(
            1, NUM_PATCHES, N_EMBD))  # (1, 16, 64)

        self.blocks = nn.Sequential(
            *[SelfAttentionBlock(is_decoder=False) for _ in range(NUM_BLOCKS)]
        )
        self.ln_final = nn.LayerNorm(N_EMBD)

    def forward(self, x):
        x = self.patch_embed(x)  # (B, 16, 64)
        x = x + self.pos_embed   # Add positional encoding
        x = self.blocks(x)       # Apply transformer encoder blocks
        x = self.ln_final(x)
        return x  # (B, 16, 64)

# - TransformerDecoder


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, N_EMBD)
        self.pos_embed = nn.Embedding(CONTEXT_SIZE, N_EMBD)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                SelfAttentionBlock(is_decoder=True),
                CrossAttentionBlock()
            ) for _ in range(NUM_BLOCKS)
        ])

        self.ln_final = nn.LayerNorm(N_EMBD)
        self.output_layer = nn.Linear(N_EMBD, vocab_size)

    def forward(self, x, encoder_output):
        B, T = x.shape
        token_embeddings = self.token_embed(x)  # (B, T, 64)
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        position_embeddings = self.pos_embed(positions)  # (1, T, 64)
        x = token_embeddings + position_embeddings

        for block in self.blocks:
            self_attn, cross_attn = block
            x = self_attn(x)
            x = cross_attn(x, encoder_output)

        x = self.ln_final(x)
        logits = self.output_layer(x)  # (B, T, vocab_size)
        return logits

# - TransformerMNIST


class TransformerMNIST(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = TransformerEncoder()
        self.decoder = TransformerDecoder(vocab_size)

    def forward(self, image, input_seq):
        encoder_output = self.encoder(image)  # (B, 16, 64)
        logits = self.decoder(input_seq, encoder_output)  # (B, T, vocab_size)
        return logits


# === Greedy Decoder to evaluate the predictions===

def greedy_decode(model, image, max_len=6):
    model.eval()
    image = image.unsqueeze(0).to(next(model.parameters()).device)

    input_seq = [label_to_index["<start>"]]
    input_tensor = torch.tensor(
        input_seq, dtype=torch.long).unsqueeze(0).to(image.device)

    for _ in range(max_len - 1):
        with torch.no_grad():
            logits = model(image, input_tensor)
        next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
        input_seq.append(next_token)
        if next_token == label_to_index["<end>"]:
            break
        input_tensor = torch.tensor(
            input_seq, dtype=torch.long).unsqueeze(0).to(image.device)

    return input_seq


if __name__ == "__main__":
    VOCAB_SIZE = len(label_to_index)

    dummy_immage = torch.randn(3, 1, 56, 56)
    image_embedding = PatchEmbedding()
    assert image_embedding(dummy_immage).shape == torch.Size([3, 16, 64])

    # AttentionHead
    x = torch.randn(2, 6, 64)
    attn = AttentionHead(16, is_decoder=True)
    assert attn(x).shape == (2, 6, 16)

    # MultiHeadAttention
    mha = MultiHeadAttention(16)
    assert mha(x).shape == (2, 6, 64)

    # CrossAttentionHead
    q = torch.randn(2, 6, 64)
    k = torch.randn(2, 16, 64)
    cross = CrossAttentionHead(16)
    assert cross(q, k).shape == (2, 6, 16)

    # MultiHeadCrossAttention
    mhca = MultiHeadCrossAttention(16)
    assert mhca(q, k).shape == (2, 6, 64)

    # MLP
    mlp = MLP()
    assert mlp(x).shape == (2, 6, 64)

    # SelfAttentionBlock
    block = SelfAttentionBlock(is_decoder=True)
    assert block(x).shape == (2, 6, 64)

    # CrossAttentionBlock
    cab = CrossAttentionBlock()
    assert cab(q, k).shape == (2, 6, 64)

    # TransformerEncoder
    encoder = TransformerEncoder()
    enc_out = encoder(torch.randn(2, 1, 56, 56))
    assert enc_out.shape == (2, 16, 64)

    # TransformerDecoder
    decoder = TransformerDecoder(VOCAB_SIZE)
    seq = torch.randint(0, VOCAB_SIZE, (2, 6))
    assert decoder(seq, enc_out).shape == (2, 6, VOCAB_SIZE)

    # TransformerMNIST
    model = TransformerMNIST(VOCAB_SIZE)
    image = torch.randn(2, 1, 56, 56)
    assert model(image, seq).shape == (2, 6, VOCAB_SIZE)

    print("All asserts passed ✅")
