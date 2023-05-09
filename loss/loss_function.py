import torch
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy_with_logits

def discriminator_loss(D, HR, LR, SR, alpha):
    """
    Computes the discriminator loss for attention-enhanced super resolution GANs.
    
    Args:
    - D: The discriminator network.
    - HR: The high-resolution image tensor.
    - LR: The low-resolution image tensor.
    - SR: The super-resolved image tensor.
    - alpha: The weight of the adversarial loss in the overall discriminator loss.
    
    Returns:
    - loss_D: The discriminator loss.
    """
    
    # Generate fake label for SR
    label_SR = torch.zeros((SR.size(0), 1), dtype=torch.float32, device=SR.device)
    
    # Generate real label for HR
    label_HR = torch.ones((HR.size(0), 1), dtype=torch.float32, device=HR.device)
    
    # Compute discriminator loss for SR
    pred_SR = D(SR)
    loss_SR = binary_cross_entropy_with_logits(pred_SR, label_SR)
    
    # Compute discriminator loss for HR
    pred_HR = D(HR)
    loss_HR = binary_cross_entropy_with_logits(pred_HR, label_HR)
    
    # Compute discriminator loss for LR
    pred_LR = D(LR)
    loss_LR = binary_cross_entropy_with_logits(pred_LR, label_SR)
    
    # Compute overall discriminator loss
    loss_D = (loss_SR + loss_LR + loss_HR) * alpha
    
    return loss_D


def generator_loss(D, HR, SR, content_weight, adversarial_weight):
    """
    Computes the generator loss for the basic SRGAN.
    
    Args:
    - D: The discriminator network.
    - HR: The high-resolution image tensor.
    - SR: The super-resolved image tensor.
    - content_weight: The weight of the content loss in the overall generator loss.
    - adversarial_weight: The weight of the adversarial loss in the overall generator loss.
    
    Returns:
    - loss_G: The generator loss.
    """
    
    # Compute content loss
    loss_content = F.mse_loss(SR, HR) * content_weight
    
    # Compute adversarial loss
    label_SR = torch.ones((SR.size(0), 1), dtype=torch.float32, device=SR.device)
    pred_SR = D(SR)
    loss_adversarial = F.binary_cross_entropy_with_logits(pred_SR, label_SR) * adversarial_weight
    
    # Compute overall generator loss
    loss_G = loss_content + loss_adversarial
    
    return loss_G
