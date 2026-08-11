import torch
# data = [[1,2,3],[4,5,6],[6,7,8],[0,1,2]]
# my_tensor = torch.tensor(data)

# print(my_tensor)

#shape = (2,3)

#ones = torch.ones(shape)
#zeros = torch.zeros(shape)
#random=torch.randn(shape)

#print(f'Random Tensor:\n {random}')

# template = torch.tensor([[1,2],[3,4]])
# rand_like = torch.randn_like(template,dtype=torch.float)

# print(f'Template Tensor:\n {template}')
# print(f'Randn_like Tensor:\n {rand_like}')

tensor = torch.randn(2,3)
print(f"shape: {tensor.shape}")
print(f"device: {tensor.device}")
print(f"dtype: {tensor.dtype}")