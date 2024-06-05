import numpy as np
import torch
from sklearn.manifold import TSNE
import wandb
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def visualize_latent_space(model, dataloader, device, epoch, perplexity=5):
    model.eval()
    latents = []
    labels = []

    with torch.no_grad():
        for data in dataloader:
            input_images, target_images = data
            input_images = input_images.to(device)
            mu, logvar = model.encode(input_images)
            z = model.reparameterize(mu, logvar)
            latents.append(z.cpu().numpy())
            labels.append(np.argmax(target_images.cpu().numpy(), axis=1))  # Assuming labels are one-hot encoded
    
    latents = np.concatenate(latents, axis=0)
    labels = np.concatenate(labels, axis=0)

    # t-SNE를 사용하여 2차원으로 차원 축소
    tsne = TSNE(n_components=2, perplexity=perplexity)
    latents_2d = tsne.fit_transform(latents)

    # 등고선 플롯 그리기
    x = latents_2d[:, 0]
    y = latents_2d[:, 1]
    z = labels

    plt.figure(figsize=(10, 8))
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    
    zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='cubic')
    
    plt.contourf(xi, yi, zi, levels=14, cmap='viridis')
    plt.colorbar()
    plt.title(f'Latent Space Contour Plot at Epoch {epoch}')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')


    # 플롯을 이미지로 저장하고 W&B에 로깅
    plt_path = f"latent_space_epoch_{epoch}.png"
    plt.savefig(plt_path)
    plt.close()


    # W&B에 이미지 로깅
    wandb.log({"Latent Space Contour": wandb.Image(plt_path)})
    
    

    
    
