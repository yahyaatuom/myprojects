import torch

relu = torch.nn.ReLU()
sample_data = torch.tensor([-2.0,-0.5,0.0,0.5,2.0])
activated_data = relu(sample_data)

print(f'Original Data:   {sample_data}')
print(f'Data after ReLU: {activated_data}')