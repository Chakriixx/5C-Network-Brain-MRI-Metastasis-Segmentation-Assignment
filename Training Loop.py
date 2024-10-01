import torch.optim as optim
from torch.utils.data import DataLoader

# Define DICE Loss (simplified version)
def dice_loss(pred, target, smooth=1e-5):
    pred = pred.contiguous()
    target = target.contiguous()
    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    
    return 1 - loss.mean()

# Training loop for both models
def train_model(model, train_loader, val_loader, epochs=25):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = dice_loss
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}')
        evaluate_model(model, val_loader)

# Example usage:
train_model(model_nested_unet, train_loader, val_loader)
