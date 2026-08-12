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

# a = torch.tensor(2.0,requires_grad=True)
# b = torch.tensor(3.0,requires_grad=True)
# x = torch.tensor(4.0,requires_grad=True)

# y = a+b
# z = x*y

# print(f"grad_fn for z: {z.grad_fn}")
# print(f"grad_fn for y: {y.grad_fn}")
# print(f"grad_fn for a: {a.grad_fn}")

# a = torch.tensor([[1,2,3],[4,5,6]])
# b = torch.tensor([[7,8],[9,10],[11,12]])

# element_wise_product = a@b
# print(element_wise_product)

# scores = torch.tensor([[10.,20.,30.],[5.,10.,15.]])
# average_per_assg = scores.mean(dim=0)
# avg_per_std=scores.mean(dim=1)
# print(f"average per assignment: {average_per_assg}")
# print(f"average per student: {avg_per_std}")

# x = torch.arange(12).reshape(3,4)
# print(x)

# scores = torch.tensor([
#     [10,0,5,20,1],
#     [1,30,2,5,0]
# ])
# print(torch.argmax(scores,dim=1))

data = torch.tensor([
    [10,11,12,13,],
    [20,21,22,23],
    [30,31,32,33]
])

indices_to_select = torch.tensor([[2],[0],[3]])
selected_values = torch.gather(data, dim=1,index=indices_to_select)
print(selected_values)