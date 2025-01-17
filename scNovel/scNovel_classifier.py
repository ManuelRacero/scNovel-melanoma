import torch
import torch.nn as nn

## Clase define una red neuronal profunda con 3 capas ocultas, se ajusta 
## dependiendo del número de clases.

## Tres capas ocultas: Cada capa oculta tiene una combinación de normalización por ## lotes (Batch Normalization), activación (ELU) y abandono (Dropout) para mejorar ## el rendimiento y la generalización.

class classifier(nn.Module):
    '''
    Deep Neural Network with 3 hidden layers,
    each layer has batch normalization function locate before the activation function (ELU),
    the followed by the dropout function to reduce overfiting the reference and overcome batch effect
    '''
    def __init__(self, input_size, class_num):
        '''
        :param input_size: Input layer unit number (feature number)
        :param class_num: Number of different cell types.
        '''
        if input_size == None or class_num == None:
            raise ValueError("Must declare number of features and number of classes")
        super(classifier, self).__init__()



        if class_num<16:
            self.layer1 = nn.Linear(input_size, 128)
            self.bn1 = nn.BatchNorm1d(128)
            self.elu1 = nn.ELU()
            self.dropout1 = nn.Dropout(0.5)
            self.layer2 = nn.Linear(128, 64)
            self.bn2 = nn.BatchNorm1d(64)
            self.elu2 = nn.ELU()
            self.dropout2 = nn.Dropout(0.5)
            self.layer3 = nn.Linear(64, 32)
            self.bn3 = nn.BatchNorm1d(32)
            self.elu3 = nn.ELU()
            self.dropout3 = nn.Dropout(0.1)
            self.layer4 = nn.Linear(32, class_num)
        elif class_num<64:
            self.layer1 = nn.Linear(input_size, 256)
            self.bn1 = nn.BatchNorm1d(256)
            self.elu1 = nn.ELU()
            self.dropout1 = nn.Dropout(0.5)
            self.layer2 = nn.Linear(256, 128)
            self.bn2 = nn.BatchNorm1d(128)
            self.elu2 = nn.ELU()
            self.dropout2 = nn.Dropout(0.5)
            self.layer3 = nn.Linear(128, 64)
            self.bn3 = nn.BatchNorm1d(64)
            self.elu3 = nn.ELU()
            self.dropout3 = nn.Dropout(0.1)
            self.layer4 = nn.Linear(64, class_num)
        else :
            self.layer1 = nn.Linear(input_size, 512)
            self.bn1 = nn.BatchNorm1d(512)
            self.elu1 = nn.ELU()
            self.dropout1 = nn.Dropout(0.5)
            self.layer2 = nn.Linear(512, 256)
            self.bn2 = nn.BatchNorm1d(256)
            self.elu2 = nn.ELU()
            self.dropout2 = nn.Dropout(0.5)
            self.layer3 = nn.Linear(256, 128)
            self.bn3 = nn.BatchNorm1d(128)
            self.elu3 = nn.ELU()
            self.dropout3 = nn.Dropout(0.1)
            self.layer4 = nn.Linear(128, class_num)
    def forward(self, x):
        '''
        :param x: forward calculation
        :return:
        '''

        out = self.layer1(x)
        out = self.bn1(out)
        out = self.elu1(out)
        out = self.dropout1(out)
        out = self.layer2(out)
        out = self.bn2(out)
        out = self.elu2(out)
        out = self.dropout2(out)
        out = self.layer3(out)
        out = self.bn3(out)
        out = self.elu3(out)
        out = self.dropout3(out)
        # print(torch.max(out))


        out = self.layer4(out)
        return out

