
def process_sub_batches(images, masks, sub_batch_size=8, training=True):
    sub_batch_losses = []
    
    if training:
        optimizer.zero_grad()  # Clear gradients before processing the main batch
    
    for i in range(0, len(images), sub_batch_size):
        sub_images = images[i:i + sub_batch_size]
        sub_masks = masks[i:i + sub_batch_size]
        
        with autocast():
            outputs = model(sub_images)
            loss = criterion(outputs, sub_masks)
        
        if training:
            scaler.scale(loss).backward()  # Accumulate gradients
        sub_batch_losses.append(loss.item() * sub_images.size(0))
    
    if training:
        scaler.step(optimizer)  # Update model parameters once after processing all sub-batches
        scaler.update()
    
    return sum(sub_batch_losses) / len(images)
