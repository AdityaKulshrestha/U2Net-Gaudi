import os
from skimage import io, transform 
import glob
import numpy as np 

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

from PIL import Image 


device = torch.device('hpu')

def normPRED(d): 
    max_value = torch.max(d) 
    min_value = torch.min(d) 
    denom = (max_value - min_value)
    dn = (d - min_value)/denom

    return dn

def save_output(image_name, pred, d_dir):

    predict = pred 
    predict = predict.squeeze() 
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]

    image = io.imread(image_name) 
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    pb_np = np.array(imo) 

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]

    for i in range(1, len(bbb)): 
        imidx = imidx + '.' + bbb[i]

    imo.save(d_dir + img_name + '.png')


def main(): 

    model_name = "u2net"
    model_chkpt = "bce_itr_34000_train_0.1629488290399313_tar0.014324243980459868.pth"
    image_dir = os.path.join(os.getcwd(), 'data','test_data', 'DUTS-TE-Image')
    prediction_dir = os.path.join(os.getcwd(), 'output_results', model_name + '_results' + os.sep)
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name ,model_chkpt)

    img_name_list = glob.glob(image_dir+os.sep+'*')

    # print(len(img_name_list))

    # ---------------- data loader ------------------
    test_salobj_dataset = SalObjDataset(
        img_name_list = img_name_list,
        lbl_name_list = [], 
        transform = transforms.Compose([Rescale(320), ToTensorLab(flag=0)]) 
        )

    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size = 1, shuffle=False, num_workers =1)

    net = U2NET(3,1)

    net.load_state_dict(torch.load(model_dir))
    net.to(device)
    net.eval()

    for i_test, data_test in enumerate(test_salobj_dataloader): 

        print("Inferencing: ", img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor) 

        inputs_test = Variable(inputs_test).to(device)

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        pred = d1[:, 0, :, :]
        pred = normPRED(pred)

        if not os.path.exists(prediction_dir): 
            os.makedirs(prediction_dir, exist_ok = True)

        save_output(img_name_list[i_test], pred, prediction_dir)

        del d1, d2, d3, d4, d5, d6, d7 

if __name__ == "__main__":
    main()
