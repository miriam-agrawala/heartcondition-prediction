## Introduction
The code in this repository is intended to predict various diagnoses based on AI from 10-second ECG sequences. Different combinations of LSTM and CNN architectures were used. Furthermore, different variations of optimisers, learning rate schedulers, dropout and batch normalisation were used.

The training was tested on three different datasets: a standard dataset with 10 second sequences on 12 ECG channels, a modified dataset containing only the last 2 seconds of the standard dataset and finally one dataset containing 8 seconds of the standard dataset with the start of the 8 second sequence randomised. All datasets were derived from the [PTB-XL ECG dataset on PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/).

## How to use this repository

Please follow these instructions, to use the code:

Firstly, the repository is to be cloned. Furthermore, the PTB-XL data set must be downloaded from PhysioNet.

In “import_data.py”, the path for importing the training data has to be adapted. Please insert the path that points to the folder in which the folders for the 100Hz and 500Hz recordings are located (see code example lines 16-18).

To train a network, only main.py needs to be executed. In the version in which the code is downloaded from the repository, the best tested network architecture is trained, i.e. two LSTM with three convolutional layers, dropout and AdamW with weight_decay=0.1 on the randomised data set.

## Make own Modifications

Alternatively, various changes can be made to the code in order to change the model in the desired way: 

In line 35 of “main.py”, the "net" variable can be set equal to the network architecture that you want to test. You can choose between "LSTM", "LSTM_2stacked", "LSTM_3stacked", "LSTM_3Conv" and "LSTM_5Conv". 
The dataset can also be exchanged in lines 25 and 28 of “main.py”; "ECGDatasetUpdate" (standard dataset), "ECGDataset200" (last 200 data points) and "ECGDatasetRandomStart" are available.

In addition, by varying lines 14-16 (Adam) or 18-20 (AdamW) in the “trainloop.py file”, different variations of the optimisers, their learning rates and the weight decay can be tested.

The Learning Rate Scheduler is located in the same file. In lines 22-24 the version that modifies the LR in each epoch can be tested. In lines 26-27 the version in which the step_size variable is used to specify in which epochs the LR should be changed can be tested. To actually use the LR scheduler in the code, line 61 in main.py must also be commented in.

By changing the values "hidden_size" and "num_layers" in the LSTM layer of all architectures, the network can also be made deeper or less deep (e.g. in line 19 of lstm_3conv.py).

Batchnormalisation can be commented in and out after each convolutional layer. 

Dropout can be commented in or out in the forward pass of each architecture (e.g. in line 35 of “lstm_3conv.py”). 

## Resume Training from Checkpoints

Furthermore, various checkpoints are provided, which can simply be loaded into the code as a URL using the gdown library.

Using checkpoints, the network can be further trained from the point with the best balanced accuracy on 1000 epochs (in the validation data set). 

The following table contains all tested variations for which there are checkpoints. The correct URL for the checkpoint can be selected from the list below using the Network_ID in the first column of the table. In the list below, the Network_ID is written first, followed by a ":" and the link for the URL for the checkpoint.

![alt_text](https://github.com/miriam-agrawala/heartcondition-prediction/blob/main/Table_Networkarchitectures.PNG)

1: https://drive.google.com/file/d/1f1DndXamNG1kGk3EMWdcMwNUjzm1I4Gc/view?usp=sharing

2: https://drive.google.com/file/d/1G20LqwqiAmuD7jvxEwgkHxqFIl9p5eB_/view?usp=sharing

3: https://drive.google.com/file/d/1hFZKTqr8-hYr5dOuJl7ylP9vWTIuDn3R/view?usp=sharing

4: https://drive.google.com/file/d/1on2i0BYpaoohPYMNpbTFZ0yAStfzVdD1/view?usp=sharing

5: [https://drive.google.com/file/d/1jcVWqww3sdLMDtnRmZrOnKpsZHaW8p5W/view?usp=sharing

6: https://drive.google.com/file/d/1IX5yBQsAEWbdcafAt82RyfJedDW096DJ/view?usp=sharing

7: https://drive.google.com/file/d/122tBwQEa03DFzRkWIR6h-zzADFXgczpR/view?usp=sharing

8: https://drive.google.com/file/d/1KhdiTWf-zFOCmKFL32N9d16mwUxv5KcT/view?usp=sharing

9: https://drive.google.com/file/d/1g6eIRhC2qivE2TbyQWfzA-0ueHT4cLON/view?usp=sharing

10: https://drive.google.com/file/d/1utmAXmyjanYku6r02mN8o5qSuKxci1GS/view?usp=sharing

11: https://drive.google.com/file/d/17dgIrdSdoLuuCvtrL8s2dpBSUzyttQmZ/view?usp=sharing

12: https://drive.google.com/file/d/1uMIThfnyuf_joqcqLhcaGJ8K1lkmco4L/view?usp=sharing

13: https://drive.google.com/file/d/1O5qI_K9luViq1gBEk_yDmxIvMWVV2nhp/view?usp=sharing

14: https://drive.google.com/file/d/1bJ55FkklesbSerCY69Gfuwbg-KdOLlCw/view?usp=sharing

15: https://drive.google.com/file/d/17xoyBCYhg0hDRxSeZfypDt64ahrIbUhI/view?usp=sharing

16: https://drive.google.com/file/d/19zY1y1jxKNAIE4F7z2C0tvqeU32Qn_Dk/view?usp=sharing

17: https://drive.google.com/file/d/1DyRUE7o9CzDrD04A6KQqWAGPveepMoXa/view?usp=sharing

18: https://drive.google.com/file/d/1zh8hB4ACUQvF4Y576t6R5u4XMsxOyUSh/view?usp=sharing


These links can be used in line 23 of the file "resume_training_fromcheckpoint.py" for the url variable. As soon as the network has been adapted according to the respective checkpoint, the training can be continued by starting the script.


