from sklearn.metrics import jaccard_score

# Evaluation with DICE score
def dice_score(preds, labels, threshold=0.5):
    preds = (preds > threshold).float()
    intersection = (preds * labels).sum()
    union = preds.sum() + labels.sum()
    return 2.0 * intersection / union

# Evaluation function
def evaluate_model(model, val_loader):
    model.eval()
    total_dice = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            dice = dice_score(outputs, masks)
            total_dice += dice.item()
    
    print(f'DICE Score: {total_dice/len(val_loader)}')

# Evaluate both models
evaluate_model(model_nested_unet, val_loader)
evaluate_model(model_attention_unet, val_loader)
