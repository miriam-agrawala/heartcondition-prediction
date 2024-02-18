import torch
import tqdm

class Trainer:
    def __init__(self, network, loss_function, writer):
      # Initialize the Trainer with a network, loss function and TensorBoard writer
      self.network = network
      self.loss_function = loss_function
      self.writer = writer
      # Initialize the optimizer and scheduler
      self.init_optimizer()
      self.init_scheduler()

    def init_optimizer(self):
      # Initialize the optimizer as Adam
      self.optim = torch.optim.Adam(self.network.parameters(), lr=0.0001)

    # def init_optimizer(self):
    #   self.optim = torch.optim.AdamW(self.network.parameters(), lr=0.0001, weight_decay=0.1)

    def init_scheduler(self):
      # Initialize the LR scheduler as ExponentialLR
      self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, 0.95)

    # def init_scheduler(self):
    #   self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=30, gamma=0.1)

    def epoch(self, dataloader, training, epoch=0):
      # Start a new epoch, with a progress bar from tqdm
      bar = tqdm.tqdm(dataloader)
      
      # Set the network to training or evaluation mode
      if training:
        self.network.train()
        name="train"
      else:
        self.network.eval()
        name="val"

      # Initialize counters for loss, accuracy and balanced accuracy
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
        labels = labels.view(-1)

        # Calculcate the Loss
        loss = self.loss_function(lstm, labels)
      
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
        # Calculate balanced accuracy
        bacc = 0
        for cls_idx in range(5):
          if total_class[cls_idx] != 0:
            bacc += correct_class[cls_idx] / total_class[cls_idx]
        bacc /= 5
        # turn into percentage
        bacc *= 100
      
        # Calculaate average loss and accuracy    
        avg_loss = 1000.0 * total_loss / cnt
        avg_acc = 100.0*correct/cnt

        # Update bar description
        bar.set_description(f"ep: {epoch:.0f} ({name}), loss: {avg_loss:.3f}, acc: {avg_acc:.2f}%, bacc: {bacc:.2f}%")

        # If we are training, do backward pass 
        if training:
            # Calculcate backward gradients
            loss.backward()
            # Step the optimizer
            self.optim.step()

      if training:
            # Log the average loss and accuracy for training to TensorBoard
            self.writer.add_scalar('Acc/train', avg_acc, epoch)
            self.writer.add_scalar('Loss/train', avg_loss, epoch)
            self.writer.add_scalar('Bacc/train', bacc, epoch)

      else:
            # Log the average loss and accuracy for validation to TensorBoard
            self.writer.add_scalar('Acc/val', avg_acc, epoch)
            self.writer.add_scalar('Loss/val', avg_loss, epoch)
            self.writer.add_scalar('Bacc/val', bacc, epoch)

      # Return the average loss, accuracy and balanced accuracy
      return avg_loss, avg_acc, bacc

    def save_checkpoint(self, epoch, path):
      # Save the current state of the network and optimizer as a checkpoint
      torch.save({
              'epoch': epoch,
              'model_state_dict': self.network.state_dict(),
              'optimizer_state_dict': self.optim.state_dict(),
              }, path)