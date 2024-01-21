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
    else:
      self.network.eval()
      name="val"

    # This epoch starts
    total_loss = 0
    correct = 0
    cnt = 0

    correct_class = [0, 0, 0, 0, 0]
    total_class = [0, 0, 0, 0, 0]

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
      #print("HERE LOSS ->", loss, "<- HERE LOSS")

      # Sum the total loss
      total_loss += loss.item()

      # Count how many correct predictions we have (for accuracy)
      preds = torch.argmax(lstm, dim=1)
      correct += torch.sum(preds == labels).item()

      # Update counters for balanced accuracy
      for cls_idx in range(5):
        correct_class[cls_idx] += torch.sum((preds == cls_idx) & (labels == cls_idx)).item()
        total_class[cls_idx] += torch.sum(labels == cls_idx).item()
    
      # Count total samples processed
      cnt += batch.shape[0]

      bacc = 0
      for cls_idx in range(5):
        if total_class[cls_idx] != 0:
          bacc += correct_class[cls_idx] / total_class[cls_idx]
      bacc /= 5

      # Calculaate average loss and accuracy    
      avg_loss = 1000.0 * total_loss / cnt
      avg_acc = 100.0*correct/cnt

      # Update bar description
      bar.set_description(f"ep: {epoch:.0f} ({name}), loss: {avg_loss:.3f}, acc: {avg_acc:.2f}%, bacc: {100.0*bacc:.2f}%")

      # If we are training, do backward pass 
      if training:
          # Calculcate backward gradiesnts
          loss.backward()

          # Step the optimizer
          self.optim.step()

          # Log the average loss and accuracy to TensorBoard
          self.writer.add_scalar('Acc/train', avg_acc, epoch)
          self.writer.add_scalar('Loss/train', avg_loss, epoch)
          self.writer.add_scalar('Bacc/train', bacc, epoch)
          # Log the network graph to TensorBoard
          #self.writer.add_graph(self.network, batch)
      else:
          # Log the average loss and accuracy for validation
          self.writer.add_scalar('Acc/val', avg_acc, epoch)
          self.writer.add_scalar('Loss/val', avg_loss, epoch)
          self.writer.add_scalar('Bacc/val', bacc, epoch)
    return avg_loss, avg_acc

  # def validate(self, val_dataloader, epoch=0):
        
  #       bar = tqdm.tqdm(val_dataloader)
  #       # Switch to evaluation mode
  #       self.network.eval()
  #       total_loss = 0
  #       correct = 0
  #       cnt = 0
  #       with torch.no_grad():
  #           for batch, labels in bar:
  #               # Forward pass
  #               outputs = self.network(batch)
  #               # Reshape labels to be 1D
  #               labels = labels.view(-1)
  #               # Calculate the loss
  #               loss = self.loss_function(outputs, labels)
  #               # Sum the total loss
  #               total_loss += loss.item()
  #               # Count how many correct predictions we have (for accuracy)
  #               correct += torch.sum(torch.argmax(outputs, dim=1) == labels).item()
  #               # Count total samples processed
  #               cnt += batch.shape[0]

  #               bar.set_description(f"ep: {epoch:.0f} (val), loss: {1000.0*total_loss / cnt:.3f}, acc: {100.0*correct/cnt:.2f}%")

  #       # Print validation loss and accuracy
  #       print(f'Validation Loss: {total_loss/cnt}, Validation Accuracy: {correct/cnt}')

  #       avg_loss = 1000.0 * total_loss / cnt
  #       avg_acc = 100.0*correct/cnt
  #       self.writer.add_scalar('Acc/val', avg_acc, epoch)
  #       self.writer.add_scalar('Loss/val', avg_loss, epoch)
  #       self.writer.add_graph(self.network, batch)
  #       return avg_loss, avg_acc

  def save_checkpoint(self, epoch, path):
    torch.save({
            'epoch': epoch,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            }, path)