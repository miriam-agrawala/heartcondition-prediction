import torch
import torch.nn as nn
import tqdm

class Trainer:
  def __init__(self, network, loss_function, writer):
    self.network = network
    self.loss_function = loss_function
    self.writer = writer
    
    self.init_optimizer()
    self.init_scheduler()

  def init_optimizer(self):
    self.optim = torch.optim.Adam(self.network.parameters(), lr=0.0001)

  def init_scheduler(self):
    self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, 0.95)

  def epoch(self, dataloader, training, epoch=0):
    # We want a dedicated TQDM bar, so we can set the description after each step
    bar = tqdm.tqdm(dataloader)
    
    # Tell the network whether we are training or evaluating (to disable DropOut)
    if training:
      self.network.train()
      name="train"


    # This epoch starts
    total_loss = 0
    correct = 0
    cnt = 0

    correct_class = [0,0,0,0,0]
    total_class = [0,0,0,0,0]
    # Iterate over the whole epoch
    for batch, labels in bar:
      # If we are training, zero out the gradients in the network
      if training:
        self.optim.zero_grad()


      # Do one forward pass
      lstm = self.network(batch)

      # Reshape labels for processing
      labels = labels.view(-1)#.reshape

      # Calculcate the (BCE)-Loss
      loss = self.loss_function(lstm, labels)

      # Sum the total loss
      total_loss += loss.item()

      # Count how many correct predictions we have (for accuracy)
      for cls_idx in range(5):
        mask = (labels==cls_idx)
        if torch.any(mask):
            correct_class[cls_idx] += torch.sum(torch.argmax(lstm[mask, :]) == cls_idx)
            total_class[cls_idx] += torch.sum(mask)

      correct += torch.sum(torch.argmax(lstm, dim=1) == labels).item()

      # Count total samples processed
      cnt += batch.shape[0]

    
      bacc = 0
      for cls_idx in range(5):
        if total_class[cls_idx] != 0:
            bacc += correct_class[cls_idx] / total_class[cls_idx]

        bacc /= 5  
        # Update bar description

      bar.set_description(f"ep: {epoch:.0f} ({name}), loss: {1000.0*total_loss / cnt:.3f}, acc: {100.0*correct/cnt:.2f}%, bacc: {100.0*bacc:.2f}%")

      # If we are training, do backward pass 
      if training:
          # Calculcate backward gradiesnts
          loss.backward()

          # Step the optimizer
          self.optim.step()

          #self.validate(val_dataloader)     
          avg_loss = 1000.0 * total_loss / cnt
          avg_acc = 100.0*correct/cnt
          self.writer.add_scalar('Acc/train', avg_acc, epoch)
          self.writer.add_scalar('Loss/train', avg_loss, epoch)
          self.writer.add_graph(self.network, batch)


    return avg_loss, avg_acc

  def validate(self, val_dataloader, epoch=0):
        
        
    # Initialize correct_class and total_class
    correct_class = [0,0,0,0,0]
    total_class = [0,0,0,0,0] 

    bar = tqdm.tqdm(val_dataloader)
    # Switch to evaluation mode
    self.network.eval()
    total_loss = 0
    correct = 0
    cnt = 0
    with torch.no_grad():
        for batch, labels in bar:
            # Forward pass
            outputs = self.network(batch)
            # Reshape labels to be 1D
            labels = labels.view(-1)
            # Calculate the loss
            loss = self.loss_function(outputs, labels)
            # Sum the total loss
            total_loss += loss.item()
            # Count how many correct predictions we have (for accuracy)
            correct += torch.sum(torch.argmax(outputs, dim=1) == labels).item()
            # Count total samples processed
            cnt += batch.shape[0]

            for cls_idx in range(5):
                mask = (labels==cls_idx)
                if torch.any(mask):
                    correct_class[cls_idx] += torch.sum(torch.argmax(outputs[mask, :]) == cls_idx)
                    total_class[cls_idx] += torch.sum(mask)

    # Calculate balanced accuracy
    bacc = 0
    for cls_idx in range(5):
        if total_class[cls_idx] != 0:
            bacc += correct_class[cls_idx] / total_class[cls_idx]
    bacc /= 5

    bar.set_description(f"ep: {epoch:.0f} (val), loss: {1000.0*total_loss / cnt:.3f}, acc: {100.0*correct/cnt:.2f}%")

    # Print validation loss and accuracy
    print(f'Validation Loss: {total_loss/cnt}, Validation Accuracy: {correct/cnt}, Bacc: {bacc}')

    avg_loss = 1000.0 * total_loss / cnt
    avg_acc = 100.0*correct/cnt
    self.writer.add_scalar('Acc/val', avg_acc, epoch)
    self.writer.add_scalar('Loss/val', avg_loss, epoch)
    self.writer.add_graph(self.network, batch)
    return avg_loss, avg_acc

  def save_checkpoint(self, epoch, path):
    torch.save({
            'epoch': epoch,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            }, path)