import torch


print(torch.cuda.is_available())
torch.zeros(1).cuda()
