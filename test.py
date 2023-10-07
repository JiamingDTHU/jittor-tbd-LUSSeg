import json
import os
import shutil

import numpy as np
import jittor as jt
import jittor.nn as nn
jt.flags.use_cuda = 1
from PIL import Image
import jittor.transform as transforms
from tqdm import tqdm

import src.resnet as resnet_model
from src.singlecropdataset import InferImageFolder
from src.utils import hungarian
  
def main():
    # build model
    model = resnet_model.__dict__['resnet18'](
        hidden_mlp=0, output_dim=0, nmb_prototypes=0, train_mode='finetune', num_classes=50)
    checkpoint = jt.load("./weights/pass50/pixel_finetuning/checkpoint.pth.tar")["state_dict"]
    for k in list(checkpoint.keys()):
        if k not in model.state_dict().keys():
            del checkpoint[k]
    model.load_state_dict(checkpoint)
    print("=> loaded model '{}'".format("./weights/pass50/pixel_finetuning/checkpoint.pth.tar"))
    model.eval()

    # build dataset
    data_path = os.path.join("../ImageNet-S/ImageNetS50", "test")
    validation_segmentation = os.path.join("../ImageNet-S/ImageNetS50",
                                           'validation-segmentation')
    normalize = transforms.ImageNormalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset = InferImageFolder(root=data_path,
                               transform=transforms.Compose([
                                   transforms.Resize(256),
                                   transforms.ToTensor(),
                                   normalize,
                               ]))
    dataloader = dataset.set_attrs(
        batch_size=1, 
        num_workers=16)

    # dump_path = os.path.join("./weights/pass50/pixel_finetuning", "test")
    dump_path = "./result"

    targets = []
    predictions = []
    for images, path, height, width in tqdm(dataloader):
        path = path[0]
        cate = path.split('/')[-2]
        name = path.split('/')[-1].split('.')[0]
        if not os.path.exists(os.path.join(dump_path, cate)):
            os.makedirs(os.path.join(dump_path, cate))

        with jt.no_grad():
            H = height.item()
            W = width.item()

            output = model(images)

            if H * W > 1000 * 1000 and 1000 > 0:
                output = nn.interpolate(output, (1000, int(1000 * W / H)), mode="bilinear", align_corners=False)
                output = jt.argmax(output, dim=1, keepdims=True)[0]
                prediction = nn.interpolate(output.float(), (H, W), mode="nearest").long()
            else:
                output = nn.interpolate(output, (H, W), mode="bilinear", align_corners=False)
                prediction = jt.argmax(output, dim=1, keepdims=True)[0]

            prediction = prediction.squeeze(0).squeeze(0)
            res = jt.zeros((prediction.shape[0], prediction.shape[1], 3))
            res[:, :, 0] = prediction % 256
            res[:, :, 1] = prediction // 256
            res = res.cpu().numpy()

            res = Image.fromarray(res.astype(np.uint8))
            res.save(os.path.join(dump_path, cate, name + '.png'))

            
            jt.clean_graph()
            jt.sync_all()
            jt.gc()



if __name__ == '__main__':
    main()

