from torchvision.datasets import Omniglot
from torchvision.transforms import ToTensor
import torch
from easydict import EasyDict
from model import Proto

torch.backends.cuda.cufft_plan_cache[0].max_size = 1

data = Omniglot(root="C:\\Users\\abdul\\Projects\\Self-Supervised-Learning\\Datasets\\", background=False, download=True, transform=ToTensor())

loader  = torch.utils.data.DataLoader(data, batch_size=100, num_workers=0, pin_memory=True)

args = EasyDict(dict(num_support_tr=5, learning_rate=0.0001, lr_scheduler_step=20, lr_scheduler_gamma=0.5))

model = Proto((1,105,105), 60, args, torch.device("cuda"))
model.prototyper.cuda()

i=0
while True:
    for (x,y) in loader:
        torch.cuda.empty_cache()
        x = x.cuda()
        x_hat = model.prototyper(x)
        del x
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print(f"Total Memory: {t//(1024**2)}\nReserved Memory: {r//1024**2}\nAllocated Memory: {a//(1024**2)}\nFree memory: {(r-a)//(1024**2)}")
        print(i)
        i = i+1
