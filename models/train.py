import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ..loss.loss_function import discriminator_loss, generator_loss
from tqdm import tqdm

def train_aesrgan(generator, discriminator, train_dataset, batch_size, num_epochs, content_weight, adversarial_weight, lr, device):
    """
    Trains an A-ESRGAN model using the given generator, discriminator, and training dataset.
    
    Args:
    - generator: The generator network.
    - discriminator: The discriminator network.
    - train_dataset: The training dataset.
    - batch_size: The batch size.
    - num_epochs: The number of epochs to train for.
    - content_weight: The weight of the content loss in the overall generator loss.
    - adversarial_weight: The weight of the adversarial loss in the overall generator loss.
    - lr: The learning rate.
    - device: The device to run the model on.
    """
    
    # Initialize optimizer for both generator and discriminator
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)
    
    # Initialize data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    for epoch in range(num_epochs):
        # Initialize progress bar
        progress_bar = tqdm(train_loader, desc='Epoch {}/{}'.format(epoch+1, num_epochs))
        
        # Loop over batches in the dataset
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(progress_bar):
            # Move data to device
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            # Train discriminator
            discriminator.zero_grad()
            sr_imgs = generator(lr_imgs)
            loss_D = discriminator_loss(discriminator, hr_imgs, lr_imgs, sr_imgs, alpha=0.001)
            loss_D.backward()
            optimizer_D.step()
            
            # Train generator
            generator.zero_grad()
            loss_G = generator_loss(discriminator, hr_imgs, lr_imgs, sr_imgs, content_weight=content_weight, adversarial_weight=adversarial_weight)
            loss_G.backward()
            optimizer_G.step()
            
            # Update progress bar
            progress_bar.set_postfix({'D_loss': loss_D.item(), 'G_loss': loss_G.item()})