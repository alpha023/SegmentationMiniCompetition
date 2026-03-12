import os
import torch
import argparse
import numpy as np
from PIL import Image
import torchvision.transforms as T

from model import LightSegNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = T.Compose([
    T.Resize((300,300)),
    T.ToTensor()
])

def run_inference(test_folder, group_number):

    model = LightSegNet().to(DEVICE)

    model.load_state_dict(torch.load("model.pth",map_location=DEVICE),strict=False)

    model.eval()

    output_folder = f"{group_number}_output"

    os.makedirs(output_folder,exist_ok=True)
    i=0
    for img_name in os.listdir(test_folder):

        img_path = os.path.join(test_folder,img_name)

        img = Image.open(img_path).convert("RGB")

        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            output = model(img_tensor)

            pred = torch.argmax(output,dim=1).squeeze().cpu().numpy()

        mask = Image.fromarray(pred.astype(np.uint8)*255)
        # color_mask = decode_segmap(pred)
        # mask = Image.fromarray(color_mask)

        name = os.path.splitext(img_name)[0]

        save_path = os.path.join(output_folder,f"{name}_mask.png")

        mask.save(save_path)
        if(i>200):
            break
        i=i+1

    print("Results saved in:",output_folder)

def voc_colormap():

    colormap = np.zeros((21,3), dtype=int)

    for i in range(21):
        r,g,b = 0,0,0
        cid = i

        for j in range(8):
            r |= ((cid >> 0) & 1) << (7-j)
            g |= ((cid >> 1) & 1) << (7-j)
            b |= ((cid >> 2) & 1) << (7-j)
            cid >>= 3

        colormap[i] = [r,g,b]

    return colormap
def decode_segmap(mask):

    colormap = voc_colormap()

    h, w = mask.shape
    color_mask = np.zeros((h,w,3), dtype=np.uint8)

    for label in range(21):
        color_mask[mask == label] = colormap[label]

    return color_mask

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--test_folder",type=str,required=True)
    parser.add_argument("--group",type=str,required=True)

    args = parser.parse_args()

    run_inference(args.test_folder,args.group)