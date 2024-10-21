import torch
import numpy as np


# 输入tensor


input = torch.randn(3, 2)
'''
input: tensor([[-0.2293, -1.4997],
        [-0.7896, -1.0517],
        [-1.0352,  0.9538]])
'''


# tensor求均值和方差
torch_mean = torch.mean(input, dim=0)
torch_var = torch.var(input, dim=0)
torch_var_biased = torch.var(input, dim=0, unbiased=False)
print('torch_mean:------------', torch_mean)
print('torch_var:-------------', torch_var)
print('torch_var_biased:------', torch_var_biased)
'''
torch_mean:------------ tensor([0.0904, 0.2407])
torch_var:------------- tensor([0.3105, 0.1555])
torch_var_biased:------ tensor([0.2070, 0.1037])
'''


# numpy求均值和方差
input_numpy = input.numpy()
numpy_mean = np.mean(input_numpy, axis=0)
numpy_var = np.var(input_numpy, axis=0)
print('numpy_mean:------------', numpy_mean)
print('numpy_var:-------------', numpy_var)
'''
numpy_mean:------------ [0.09037449 0.24070372]
numpy_var:------------- [0.20696904 0.10368937]
'''


# manual
man_mean = np.sum(input_numpy, axis=0)/3
print('man_mean:--------------', man_mean)
temp = input_numpy[:,0]-man_mean[0]
print('temp:------------------', temp)
man_var = np.sum(np.power(temp, 2))/3
print(man_var)  # 这个输出和torch以及numpy的第一维输出是一致的

'''
man_mean:-------------- [0.09037449 0.24070372]
temp:------------------ [ 0.5709777  -0.02869723 -0.54228044]
0.20696904261906943
'''
