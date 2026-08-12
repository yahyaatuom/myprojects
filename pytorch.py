import torch.nn as nn

class LinearRegressionModel(nn.Module):
    def __init__(self,in_features,out_features):
        super().__init__()
        self.linear_layer = nn.Linear(in_features,out_features)

    def forward(self,x):
        return self.linear_layer(x)

model = LinearRegressionModel(in_features=1,out_features=1)

# print("Model Architecture:")
# print(model)

import torch.optim as optim

lr=0.01
optimizer = optim.Adam(model.parameters(),lr)
loss_fn=nn.MSELoss()