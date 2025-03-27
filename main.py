import cv2
import numpy as np
import os
import sys
import os.path as osp

import argparse
from tqdm import tqdm
import torch.nn as nn
import torch
import torch.nn.functional as F
from natsort import natsorted
from glob import glob
from skimage import img_as_ubyte
import utils

from mambaout_unet import model_s
def main():
    print("\nMain_CXM-Vision")
    print("current_dir\t:",os.path.dirname(os.path.realpath(__file__)))  

    parser = argparse.ArgumentParser(description='Image Deraining using GVMambaIR')
    
    parser.add_argument('--input_dir', default='./datasets', type=str, help='Directory of validation images or single input file')
    parser.add_argument('--result_dir', default='./results', type=str, help='Directory for results')
    parser.add_argument('--weights', default='./weights/raindrop_net_g_440000.pth', type=str, help='Path to weights')
    parser.add_argument('--save', help='save the output images', action='store_true')
    args = parser.parse_args()

    model = model_s()
    print("Parameters\t:", sum(p.numel() for p in model.parameters()))
    checkpoint = torch.load(args.weights)
    model.load_state_dict(checkpoint['params'])
    print("Model loaded\t:",args.weights)
    model.cuda()
    model = nn.DataParallel(model)
    model.eval()
    if osp.isfile(args.input_dir):
        with torch.no_grad():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            # load image 
            img = np.float32(utils.load_img(args.input_dir))/255.
            # padded to multiple of 16
            h,w,_ = img.shape
            h_pad = 16 - h%16
            w_pad = 16 - w%16
            img = np.pad(img, ((0,h_pad),(0,w_pad),(0,0)), 'reflect')
            input_ = torch.tensor(img).unsqueeze(0).permute(0,3,1,2).cuda()

            restored = model(input_).squeeze(0).permute(1,2,0).cpu().numpy()

            # cropped
            restored = restored[:h,:w,:]
            restored = np.clip(restored,0,1)

            # Save the output image
            os.makedirs(args.result_dir, exist_ok=True)
            basename = osp.basename(args.input_dir)
            output_dir = osp.join(args.result_dir,basename)
            utils.save_img(output_dir, img_as_ubyte(restored))
            print(f"Output image saved at {output_dir}")
    else:
        # datasets = ['UAV-Rain1k','LOLv1','LOLv2','LOLv2_real']
        datasets = ['LOLv2_real']
        results = {}

        def print_results(results):
            # Print each row of results
            # num_iteration = args.weights.split('/')[-1]
            header = f"\n{'Dataset':<14}"
            
            row = f"{'PSNR|SSIM':<14}"
            for key,value in results.items():
                header += f"{key:<22}"
                row += f"{value['psnr']:<10.5f} {value['ssim']:<10.5f} "
            print(header)
            print(row)

        if args.save:
            print(f"Result dir\t: {args.result_dir}")

        for dataset in datasets:
            result_dir  = os.path.join(args.result_dir, dataset)
            os.makedirs(result_dir, exist_ok=True)
            inp_dir = os.path.join(args.input_dir, dataset, 'input')
            files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')))


            psnrs = []
            ssims = []

            print(f"\n===>Processing {dataset} dataset")
            with torch.no_grad():
                for file_ in tqdm(files):
                    torch.cuda.ipc_collect()
                    torch.cuda.empty_cache()
                    img = np.float32(utils.load_img(file_))/255.
                    gt_dir = file_.replace('input','gt')
                    gt = np.float32(utils.load_img(gt_dir))/255.

                    # padded to multiple of 16
                    h,w,_ = img.shape
                    h_pad = 16 - h%16
                    w_pad = 16 - w%16
                    img = np.pad(img, ((0,h_pad),(0,w_pad),(0,0)), 'reflect')

                    input_ = torch.tensor(img).unsqueeze(0).permute(0,3,1,2).cuda()
                    restored = model(input_).squeeze(0).permute(1,2,0).cpu().numpy()
                    # cropped
                    restored = restored[:h,:w,:]
                    # clamp 
                    restored = np.clip(restored,0,1)
                    psnr = utils.calculate_psnr(gt*255.,restored * 255.)
                    ssim = utils.calculate_ssim(gt*255.,restored * 255.)

                    psnrs.append(psnr)
                    ssims.append(ssim)
          
                    if args.save:
                        utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0]+'.png')), img_as_ubyte(restored))
                    
                

            results[dataset] = {'psnr': np.mean(psnrs), 'ssim': np.mean(ssims)}
        
        print_results(results)
        
    
 

if __name__ == "__main__":
    main()
