class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        # 3.1 Initialization
        #******************************#
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv_layer_1 = nn.Conv2d(1, 6, 5)
        self.conv_layer_2 = nn.Conv2d(6, 16, 5)
        # Defining a linear layer: y = Wx + b
#         self.fc1 = nn.Linear(16*5*5, 120)
#         self.fc2 = nn.Linear(120, 84)
        self.dropout_layer_1 = nn.Dropout(p=0.5)
        self.dropout_layer_2 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(256, 32)
        self.fc2 = nn.Linear(32, 84)
        self.fc3 = nn.Linear(84, num_classes)
        #******************************#

    def forward(self, x):
        # 3.2 Define Neural Network
        #******************************#
        out = F.max_pool2d(F.relu(self.conv_layer_1(x)), (2, 2))
        out = F.max_pool2d(F.relu(self.conv_layer_2(out)), 2)
#         print('## forward ##')
#         print(out.shape)
        flat_feature_size = self.num_flat_features(out)
#         print(flat_feature_size)
        out = out.reshape(-1, flat_feature_size)
#         print('#############')
        out = self.dropout_layer_1(F.relu(self.fc1(out)))
        out = self.dropout_layer_2(F.relu(self.fc2(out)))
        out = self.fc3(out)
        #******************************#
        return out
    
    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
model = ConvNet(num_classes).to(device)

# Note: what is the difference between 'same' and 'valid' padding? 
# Take a look at the outputs to understand the difference.

print(model)
