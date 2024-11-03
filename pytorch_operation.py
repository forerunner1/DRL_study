"""
torch.autograd是pytorch中所有神经网络的核心。autograd包为张量上的所有操作提供了自动微分。
它是一个由运行定义的框架，这意味着以代码运行方式定义你的后向传播，并且每次迭代都可以不同。
有两个核心的类：torch.Tensor和Function
1.torch.Tensor
(1)所有的tensor的类型都是torch.Tensor这个class的实例。
(2)torch.Tensor 是这个包的核心类，具有属性.requires_grad，若设为True，那么它将会追踪对于该张量的所有操作。
(3)当完成计算后可以通过调用 .backward()，来自动计算所有的梯度。这个张量的所有梯度将会自动累加到.grad属性。
   如果是一个标量（即它包含一个元素数据），.backward()不需要指定任何参数，但是如果它有更多的元素，
   你需要指定一个gradient参数，该参数是一个匹配张量的形状的张量。
(5).detach()方法将其与计算历史分离，并阻止它未来的计算记录被跟踪。
   为了防止跟踪历史记录(和使用内存），可以将代码块包装在 with torch.no_grad():中和with torch.set_grad_enabled(False)中。
   不同的是，一个是局部的，另一个是全局的。全局的需要再度启用梯度跟踪的话需要使用torch.set_grad_enabled(True)。
   在评估模型时特别有用，因为模型可能具有 requires_grad = True 的可训练的参数，但是我们不需要在此过程中对他们进行梯度计算。
(6)张量的一些操作：
    torch.view()：维度转换，里面填写维度大小，注意维度大小的乘积要与原来tensor的相同。某一维度大小-1代表根据其他维度计算该维度的大小
                返回的新tensor与源tensor共享内存(其实是同一个tensor)，更改其中的一个，另外一个也会跟着改变。(view()仅是改变了观察角度)
    torch.item(x)：获取tensor里value但不获取其他属性并返回该值，原数据类型不变。此时得到的数据类型为float等，而不是原来的torch.Tensor
    torch.as_tensor(x):把数据变成tensor。
    .unsequeeze(i) :给数据再第i+1个位置增加一个维度，如x.unsequeeze(0)，则在第1个位置增加一个维度。
    .sequeeze():进行维度压缩，去除所有大小为1的维度

2.自动微分
(1)Tensor 和 Function 互相连接生成了一个无环图 (acyclic graph)，它编码了完整的计算历史。
每个张量都有一个.grad_fn属性，该属性引用了创建Tensor自身的Function(除非这个张量是用户手动创建的，即为叶子节点：grad_fn=None )。
(2)梯度
完成计算后可以通过调用 .backward()，来自动计算所有的梯度。这个张量的所有梯度将会自动累加到.grad属性。


3.权重参数更新
import torch
import torch.nn as nn

# 定义模型
model = nn.Sequential(
    nn.Linear(10, 5),  # 第一层，有权重和偏置
    nn.ReLU(),
    nn.Linear(5, 1)    # 第二层，也有权重和偏置
)

# 打印模型参数
for param in model.parameters():
    print(param)
正是这些参数：
Parameter containing:  # 第一层权重
tensor([[ 0.1234, -0.5678,  0.2345, -0.6789,  0.3456],
        [-0.1234,  0.4567, -0.8901,  0.2345, -0.6789],
        ...
        [ 0.2345, -0.3456,  0.4567, -0.5678,  0.6789]], requires_grad=True)
Parameter containing:  # 第一层偏置
tensor([0.0001, 0.0002, 0.0003, 0.0004, 0.0005], requires_grad=True)
Parameter containing:  # 第三层权重
tensor([[ 0.1234, -0.5678,  0.2345, -0.6789,  0.3456]],
       requires_grad=True)
Parameter containing:  # 第三层偏置
tensor([0.0001], requires_grad=True)

"""

import torch

x = torch.ones(2, 2, requires_grad=True)
print(x, type(x))

"""
assert用法：assert expression, message
确保代码的某个条件为真。如果条件为假，assert将会抛出一个AssertionError异常。message是可选的，表示错误信息

np.nansum()和np.nanmean()是在计算时忽略nan值的函数。
"""