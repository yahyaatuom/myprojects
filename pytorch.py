import torch
#Linear regression y^ = XW+b

N = 10
D_in = 1
D_out = 1

X = torch.randn(N,D_in)

true_W = torch.tensor([[2.0]])
true_b = torch.tensor(1.0)
y_true = X @ true_W + true_b + torch.randn(N,D_out)*0.1

W = torch.randn(D_in,D_out,requires_grad=True)
b = torch.randn(1,requires_grad=True)

# print(f"Initial Weight W:\n {W}\n")
# print(f"Initial Bias b:\n {b}")

y_hat = X @ W + b
# print(y_hat[:3])
# print(y_true[:3])

#calculate loss
error = y_hat - y_true
squared_error = error**2
loss = squared_error.mean()

# print(f'Loss (our single scorecard number): {loss}')
loss.backward()
print(f'W.grad: {W.grad}\n')
print(f"b.grad: {b.grad}") 


lr,epochs = 0.01,100

W,b = torch.randn(1,1,requires_grad=True),torch.randn(1,requires_grad=True)

for epoch in range(epochs):
    y_hat = X@W+b
    loss = torch.mean((y_hat-y_true)**2)
    loss.backward()
    with torch.no_grad():
        W-=lr*W.grad;b -=lr*b.grad


W_new = W - lr * W.grad
b_new = b - lr * b.grad