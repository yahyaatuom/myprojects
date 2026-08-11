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

# tensor = torch.randn(2,3)
# print(f"shape: {tensor.shape}")
# print(f"device: {tensor.device}")
# print(f"dtype: {tensor.dtype}")

x_data = torch.tensor([[1.,2.],[3.,4.]]) #data
w = torch.tensor([[1.0],[2.0]],requires_grad=True) #Parameter

# print(f'Data tensor requires_grad: {x_data.requires_grad}')
# print(f'Parameter tensor requires_grad: {w.requires_grad}')

a = torch.tensor(2.0,requires_grad=True)
b = torch.tensor(3.0,requires_grad=True)
x = torch.tensor(4.0,requires_grad=True)

y = a+b
z = x*y

print(f"grad_fn for z: {z.grad_fn}")
print(f"grad_fn for y: {y.grad_fn}")
print(f"grad_fn for a: {a.grad_fn}")