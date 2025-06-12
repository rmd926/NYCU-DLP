import torch
import torch.nn as nn
import torch.nn.functional as F

class SquareActivation(nn.Module):
    def forward(self, x):
        return x ** 2

class SCCNet(nn.Module):
    def __init__(self, num_classes = 4, input_channels = 22, Nu = 22, Nt = 1, dropout_rate = 0.5, weight_decay = 1e-4):
        super(SCCNet, self).__init__()
        self.num_classes = num_classes

        # First convolution block for spatial component analysis
        '''
        - input channel : 1
        - output channel : Nu
        - (input_channels, Nt) : conv kernel，前者為spatial filter，後者(Nt)為時間dimension上kernel大小
        - apply 0 padding : 因為Nt = 1所以沒做padding沒差
        '''
        self.conv1 = nn.Conv2d(1, Nu, (input_channels, Nt), padding=(0, 0), bias=True)
        self.batchnorm1 = nn.BatchNorm2d(Nu)#對每個feature map做normalize
        self.dropout1 = nn.Dropout(dropout_rate)

        # Second convolution block for spatio-temporal filtering
        '''
        - input channel : Nu
        - output channel : 20 square units
        - (1,12) : conv kernel，其實是Nt * 12
        - apply 0 padding : 因為空間濾波Nt = 1所以沒做padding沒差。
         後者因為kernel大小為12，合理的padding是(12-1) / 2 = 5.5 => 6
        - 在這邊使用square activation
        '''
        self.conv2 = nn.Conv2d(Nu, 20, (1, 12), padding = (0, 6), bias = True)
        self.batchnorm2 = nn.BatchNorm2d(20)
        self.square_activation = SquareActivation()
        self.dropout2 = nn.Dropout(dropout_rate)

        # Pooling block for temporal smoothing
        self.avg_pool = nn.AvgPool2d((1, 62), stride=(1, 12))

        # Compute the number of features to input to the fully connected layer
        '''
        - dummy_input(batch_size,sample channel = 1, input_channel, time_steps)
        - time_steps: 125Hz * (4-0.5)s = 437.5 => 438
        - self.num_features: 計算全連接層的features數量
        - nn.Linear(input_feature, output_feature) 前者好理解就是輸入的特徵，後者就是分成四類。
        '''
        dummy_input = torch.zeros(64,1, input_channels, 438) 
        self.num_features = self._get_fc_input_features(dummy_input)
        self.fc = nn.Linear(self.num_features, num_classes)

        self.weight_decay = weight_decay

    def _get_fc_input_features(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.elu(x)  
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.square_activation(x)
        x = self.dropout2(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1) #x = (batch_size, channels, height, width)把後三者推平成一個vector  
        return x.size(1)#輸出即為c,h,w被flatten後的東東

    def forward(self, x):
        x = self.batchnorm1(self.conv1(x))
        x = F.elu(x)  
        x = self.dropout1(x)
        x = self.batchnorm2(self.conv2(x))
        x = self.square_activation(x)
        x = self.dropout2(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
