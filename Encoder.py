import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=256):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"Image size ({img_size}) must be divisible by patch size ({patch_size})")
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x2 = self.self_attn(x, x, x)[0]
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        x2 = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=256, n_layers=12, n_heads=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.img = img_size
        self.patch_size = patch_size
        self.emb = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 1025, embed_dim))  # 학습 가능한 포지셔널 인코딩
         
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(min(0.1, dropout))  # 드롭아웃 비율 조정
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, n_heads, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])
        
    def forward(self, x):
        n_samples = x.shape[0]
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(n_samples, -1, -1)  # (n_samples, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # (n_samples, n_patches + 1, embed_dim)
        x = x + self.pos_embed
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return x[:, 0]  # CLS 토큰의 출력을 사용

# 사용 예시
# 모델 초기화 및 입력 데이터 준비
model = TransformerEncoder()
input_tensor = torch.randn(8, 3, 512, 512)  # 가상의 이미지 데이터
output = model(input_tensor)
print("done")