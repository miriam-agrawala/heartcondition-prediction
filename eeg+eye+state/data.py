from scipy.io import arff
import torch
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is {DEVICE}")

class EEGDataSet(torch.utils.data.Dataset):
  def __init__(self, seqlen):
    self.data = arff.loadarff(r"C:\Users\magra\OneDrive\5_Semester\Advances_in_AI\Project\heartcondition-prediction\eeg+eye+state\EEG Eye State.arff")[0]
    print(self.data)

    self.eeg = None
    self.labels = None
    self.seqlen = seqlen

    for idx in tqdm(range(len(self.data))):
      argument = [self.data[idx][cnt] for cnt in range(14)]
      #print(argument)
      #print(type(argument))
      eeg = torch.Tensor(argument).to(DEVICE).view(1,-1)
      label = torch.Tensor([0 if self.data[idx][14] == b'0' else 1]).type(torch.long).to(DEVICE)

      if self.eeg is None:
        self.eeg = eeg
      else:
        self.eeg = torch.concat((self.eeg, eeg), dim=0)

      if self.labels is None:
        self.labels = label
      else:
        self.labels = torch.concat((self.labels, label), dim=0)

    # Normalize data
    self.eeg = (self.eeg - torch.mean(self.eeg, dim=0)) / torch.std(self.eeg, dim=0)


  def __len__(self):
    return len(self.data) - self.seqlen
  
  def __getitem__(self, idx):
    return self.eeg[idx:(idx+self.seqlen)], self.labels[idx+self.seqlen]



dataset = EEGDataSet(seqlen=4)
#print(dataset[0])