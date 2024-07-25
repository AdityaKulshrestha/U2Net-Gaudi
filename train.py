import os
import glob
import torch 
import torch.nn as nn 
from torch.autograd import Variable
from torch.nn import functional as F 
import habana_frameworks.torch.core as htcore
from model.model import U2NET 

from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

from dataloader import RescaleT
from dataloader import Rescale
from dataloader import RandomCrop
from dataloader import ToTensor
from dataloader import ToTensorLab
from dataloader import SalObjDataset


try: 
    from habana_frameworks.torch.hpex.optimizers import Adam
except ImportError:
    raise ImportError("Not supported on G2")
    
# ----------------------- 1. Define Loss Function --------------------------
bce_loss = nn.BCELoss(size_average=True)

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

    loss0 = bce_loss(d0, labels_v) 
    loss1 = bce_loss(d1, labels_v) 
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 

	# print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))
    print("l0: {:.3f}, l1: {:.3f}, l2: {:.3f}, l3: {:.3f}, l4: {:.3f}, l5: {:.3f}, l6: {:.3f}".format(
    loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(), loss4.data.item(), loss5.data.item(), loss6.data.item()
))


    return loss0, loss 


# ------------------- 2. set the directory of training dataset ---------------------------------------------
model_name = 'u2net'
device = torch.device('hpu')


data_dir = os.path.join(os.getcwd(), 'data','train_data' + os.sep)
tra_image_dir = os.path.join('DUTS-TR', 'DUTS-TR-Image' + os.sep)
tra_label_dir = os.path.join('DUTS-TR', 'DUTS-TR-Mask' + os.sep)

image_ext = '.jpg'
label_ext = '.png'


model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)

epochs = 50000 
batch_size_train = 256 
batch_size_val = 128 
train_num = 0 
val_num = 0
dtype = torch.bfloat16

tra_img_name_list = glob.glob(data_dir + tra_image_dir + "*" + image_ext)
tra_lbl_name_list = []

for img_path in tra_img_name_list: 
    img_name = img_path.split(os.sep)[-1]

    aaa = img_name.split(".")
    bbb = aaa[0: -1]
    imidx = bbb[0]

    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)


print("------")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("------")

train_num = len(tra_img_name_list)

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list, 
    lbl_name_list=tra_lbl_name_list, 
    transform=transforms.Compose([
        RescaleT(320), 
        RandomCrop(288), 
        ToTensorLab(flag=0)
    ])
)
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)

# ---------- 3. define model ----------------
net = U2NET(3, 1)
net = net.to(dtype)
net.to(device) 

# ----------- 4. define optimizer ---------------
print('----define optimizer-------')
optimizer = Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)


# ------------- 5. training process -------------
print('----start training----')
ite_num = 0 
running_loss = 0.0 
running_tar_loss = 0.0 
ite_num4val = 0 
save_frq = 2000 


for epoch in range(0, epochs): 
    net.train()


    for i, data in enumerate(salobj_dataloader):
        ite_num += 1
        ite_num4val += 1

        inputs, labels = data['image'], data['label']


        inputs = inputs.type(dtype)
        labels = labels.type(dtype)

        # wrap them in variable 
        inputs_v, labels_v = Variable(inputs.to(device), requires_grad=False), Variable(labels.to(device), requires_grad=False) 

        optimizer.zero_grad()

        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

        loss.backward()
        htcore.mark_step()

        optimizer.step() 
        htcore.mark_step()

        # print statistics 

        running_loss += loss.data.item()
        running_tar_loss += loss2.data.item()


        del d0, d1, d2, d3, d4, d5, d6, loss2, loss 



        print("[epoch: {:3d}/{:3d}, batch: {:5d}/{:5d}, ite: {:d}] train loss: {:3f}, tar: {:3f}".format(
            epoch + 1, epochs, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))



        if ite_num % save_frq == 0:

            torch.save(net.state_dict(), model_dir + f"bce_itr_{ite_num}_train_{running_loss / ite_num4val}_tar{running_tar_loss / ite_num4val}.pth")
            running_loss = 0.0 
            running_tar_loss = 0.0 
            net.train() # Resume training 
            ite_num4val = 0

