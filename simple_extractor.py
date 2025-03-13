#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   simple_extractor.py
@Time    :   8/30/19 8:59 PM
@Desc    :   Simple Extractor
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""

import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
print("PARSING")
import networks
print("PARSING Domfe")
from utils.transforms import transform_logits
from datasets.simple_extractor_dataset import SimpleFolderDataset
from types import SimpleNamespace
import cv2
dataset_settings = {
    'lip': {
        'input_size': [473, 473],
        'num_classes': 20,
        'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
    },
    'atr': {
        'input_size': [512, 512],
        'num_classes': 18,
        'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
                  'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
    },
    'pascal': {
        'input_size': [512, 512],
        'num_classes': 7,
        'label': ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'],
    }
}


def get_arguments():
    return SimpleNamespace(
        dataset = "atr",
        model_restore = "exp-schp-201908301523-atr.pth",
        gpu = "0",
        input_dir = "input", 
        output_dir = "output",
        logits = False
    )


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def parsing(human_image):
    args = get_arguments()
    print(args)
    # root = "input/00009_00.jpg"
    gpus = [int(i) for i in args.gpu.split(',')]
    assert len(gpus) == 1
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    num_classes = dataset_settings[args.dataset]['num_classes']
    input_size = dataset_settings[args.dataset]['input_size']
    label = dataset_settings[args.dataset]['label']
    print("Evaluating total class number {} with {}".format(num_classes, label))

    model = networks.init_model('resnet101', num_classes=num_classes, pretrained=None)

    state_dict = torch.load(args.model_restore)['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])
    # dataset = SimpleFolderDataset(root=args.input_dir, input_size=input_size, transform=transform)
    # dataloader = DataLoader(dataset)

    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)

    palette = get_palette(num_classes)
    with torch.no_grad():
        # for idx, batch in enumerate(tqdm(dataloader)):
            image, meta = SimpleFolderDataset(human_image, input_size=input_size, transform = transform)
            img_name = meta['name']
            c = meta['center']
            print("Center: ", c)
            s = meta['scale']
            print("Scale: ", s)
            w = meta['width']
            print("Width: ", w)
            h = meta['height']
            print("Height: ", h)

            print("Processing image: {}".format(img_name))
            print("IMAGE: ", image)
            print("IMAGE_SHAPE: ", image.shape)
            image = image.unsqueeze(0)
            output = model(image.cuda())
            upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
            upsample_output = upsample(output[0][-1][0].unsqueeze(0))
            upsample_output = upsample_output.squeeze()
            upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

            logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)
            parsing_result = np.argmax(logits_result, axis=2)
            # parsing_result_path = os.path.join(args.output_dir, img_name[:-4] + '.png')
            output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
            output_img.putpalette(palette)
            # output_img.save("mismatched_colour.png")


            output_img_rgb = output_img.convert('RGB')
            output_img_rgb.save("mismatched_colour_rgb.png")

            # input_ko_path = "humanparsing/mismatched_colour_rgb.png"

            # cv2.waitKey(100)
            # # Load the image
            human_image = np.array(output_img_rgb)
            human_image = cv2.cvtColor(human_image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(human_image, (768, 1024))
            if image is None:
                print(f"Error: Could not load image at {input_ko_path}")
                exit()

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

            # Define the RGB range for masking
            lower_bound = np.array([0, 0, 100], dtype=np.uint8)  # Example lower RGB bound
            upper_bound = np.array([0, 0, 160], dtype=np.uint8)  # Example upper RGB bound

            # Create a mask
            mask = cv2.inRange(image, lower_bound, upper_bound)

            # Create a blue image with the same shape as the original
            blue_color = np.full(image.shape, (254, 85, 0), dtype=np.uint8)  # Note: This is orange, not blue

            # Replace the masked region with blue
            image[mask > 0] = blue_color[mask > 0]

            # Save the result
            # cv2.imwrite("masked1.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            # print("Image processed and saved as masked1.png")
            
            # image = cv2.resize(human_image, (768, 1024))
            # if image is None:
            #     print(f"Error: Could not load image at {input_ko_path}")
            #     exit()

            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            lower_bound = np.array([90, 90, 90], dtype=np.uint8)  # Example lower RGB bound
            upper_bound = np.array([160, 160, 160], dtype=np.uint8)  # Example upper RGB bound

            # Create a mask
            mask = cv2.inRange(image, lower_bound, upper_bound)

            # Create a blue image with the same shape as the original
            blue_color = np.full(image.shape, (254, 85, 0), dtype=np.uint8)

            # Replace the masked region with blue
            image[mask > 0] = blue_color[mask > 0]


            # Define the RGB range for masking
            lower_bound = np.array([100, 90, 0], dtype=np.uint8)  # Example lower RGB bound
            upper_bound = np.array([200, 155, 35], dtype=np.uint8)  # Example upper RGB bound

            # Create a mask
            mask = cv2.inRange(image, lower_bound, upper_bound)

            # Create a blue image with the same shape as the original
            blue_color = np.full(image.shape, (0, 0, 255), dtype=np.uint8)

            # Replace the masked region with blue
            image[mask > 0] = blue_color[mask > 0]

            # Show results
            # cv2_imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            # cv2.imwrite("masked2.png",cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # import cv2
            # import numpy as np

            # Load the image
            # image = cv2.imread('masked2.png')  # Replace with your image path
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

            # Define the RGB range for masking
            lower_bound = np.array([0, 100, 0], dtype=np.uint8)  # Example lower RGB bound
            upper_bound = np.array([35, 160, 35], dtype=np.uint8)  # Example upper RGB bound

            # Create a mask
            mask = cv2.inRange(image, lower_bound, upper_bound)

            # Create a blue image with the same shape as the original
            blue_color = np.full(image.shape, (254, 0, 0), dtype=np.uint8)

            # Replace the masked region with blue
            image[mask > 0] = blue_color[mask > 0]

            # Show results
            # cv2_imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            # cv2.imwrite("masked3.png",cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # import cv2
            # import numpy as np

            # Load the image
            # image = cv2.imread('masked3.png')  # Replace with your image path
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

            # Define the RGB range for masking
            # lower_bound = np.array( [152, 100, 108], dtype=np.uint8)  # Example lower RGB bound
            # upper_bound = np.array( [222, 158, 159], dtype=np.uint8)  # Example upper RGB bound

            lower_bound = np.array( [108, 100, 152], dtype=np.uint8)  # Example lower RGB bound
            upper_bound = np.array( [159, 158, 252], dtype=np.uint8)

            # Create a mask
            mask = cv2.inRange(image, lower_bound, upper_bound)

            # Create a blue image with the same shape as the original
            blue_color = np.full(image.shape, (254, 254, 0), dtype=np.uint8)

            # Replace the masked region with blue
            image[mask > 0] = blue_color[mask > 0]

            # Show results
            # cv2_imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            # cv2.imwrite("masked4.png",cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            # cv2.imwrite("masked4.png",image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # import cv2
            # import numpy as np

            # Load the image
            # image = cv2.imread('masked4.png')  # Replace with your image path
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

            # Define the RGB range for masking
            lower_bound = np.array( [34, 108, 108], dtype=np.uint8)  # Example lower RGB bound
            upper_bound = np.array( [94, 158, 158], dtype=np.uint8)  # Example upper RGB bound

            # Create a mask
            mask = cv2.inRange(image, lower_bound, upper_bound)

            # Create a blue image with the same shape as the original
            blue_color = np.full(image.shape, (51, 169, 220), dtype=np.uint8)

            # Replace the masked region with blue
            image[mask > 0] = blue_color[mask > 0]

            # Show results
            # cv2_imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            # cv2.imwrite("masked5.png",cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # import cv2
            # import numpy as np

            # Load the image
            # image = cv2.imread('masked5.png')  # Replace with your image path
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

            # Define the RGB range for masking
            lower_bound = np.array( [108, 0, 108], dtype=np.uint8)  # Example lower RGB bound
            upper_bound = np.array( [158, 30, 158], dtype=np.uint8)  # Example upper RGB bound

            # Create a mask
            mask = cv2.inRange(image, lower_bound, upper_bound)

            # Create a blue image with the same shape as the original
            blue_color = np.full(image.shape, (0, 85, 85), dtype=np.uint8)

            # Replace the masked region with blue
            image[mask > 0] = blue_color[mask > 0]

            # Show results
            # cv2_imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            # cv2.imwrite("masked6.png",cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # import cv2
            # import numpy as np

            # Load the image
            # image = cv2.imread('masked6.png')  # Replace with your image path
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

            # Define the RGB range for masking
            lower_bound = np.array( [152, 0, 108], dtype=np.uint8)  # Example lower RGB bound
            upper_bound = np.array( [222, 0, 158], dtype=np.uint8)  # Example upper RGB bound

            # Create a mask
            mask = cv2.inRange(image, lower_bound, upper_bound)

            # Create a blue image with the same shape as the original
            blue_color = np.full(image.shape, (169, 254, 85), dtype=np.uint8)

            # Replace the masked region with blue
            image[mask > 0] = blue_color[mask > 0]

            # Show results
            # cv2_imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            # cv2.imwrite("masked7.png",cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # import cv2
            # import numpy as np

            # Load the image
            # image = cv2.imread('masked7.png')  # Replace with your image path
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

            # Define the RGB range for masking
            lower_bound = np.array( [34, 0, 108], dtype=np.uint8)  # Example lower RGB bound
            upper_bound = np.array( [94, 30, 168], dtype=np.uint8)  # Example upper RGB bound

            # Create a mask
            mask = cv2.inRange(image, lower_bound, upper_bound)

            # Create a blue image with the same shape as the original
            blue_color = np.full(image.shape, (85, 254, 169), dtype=np.uint8)

            # Replace the masked region with blue
            image[mask > 0] = blue_color[mask > 0]

            # Show results
            # cv2_imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            # cv2.imwrite("masked8.png",cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


            #palette yespachi

            # from google.colab.patches import cv2_imshow

            # Load the image

            # image_path = "masked8.png"
            # image = cv2.imread(image_path)  # Ensure image is in BGR mode

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_array = np.array(image)

            target_color =np.array([255, 0, 0])

            tolerance = 50

            # Compute lower and upper bounds for each channel
            lower_bound = target_color - tolerance
            upper_bound = target_color + tolerance

            # Create a boolean mask with the tolerance applied to each channel
            mask = np.all((image_array >= lower_bound) & (image_array <= upper_bound), axis=-1)


            output_array = np.where(mask, 13, 0)

            image_array = np.array(image)

            target_color =np.array([0, 51, 85])

            tolerance = 50

            # Compute lower and upper bounds for each channel
            lower_bound = target_color - tolerance
            upper_bound = target_color + tolerance

            # Create a boolean mask with the tolerance applied to each channel
            mask = np.all((image_array >= lower_bound) & (image_array <= upper_bound), axis=-1)


            output_array[mask] = 10



            image_array = np.array(image)

            target_color =np.array([0, 85, 254])

            tolerance = 50

            # Compute lower and upper bounds for each channel
            lower_bound = target_color - tolerance
            upper_bound = target_color + tolerance

            # Create a boolean mask with the tolerance applied to each channel
            mask = np.all((image_array >= lower_bound) & (image_array <= upper_bound), axis=-1)


            output_array[mask] = 5


            image_array = np.array(image)

            target_color =np.array([0, 0, 254])

            tolerance = 50

            # Compute lower and upper bounds for each channel
            lower_bound = target_color - tolerance
            upper_bound = target_color + tolerance

            # Create a boolean mask with the tolerance applied to each channel
            mask = np.all((image_array >= lower_bound) & (image_array <= upper_bound), axis=-1)

            output_array[mask] = 2


            image_array = np.array(image)

            target_color =np.array([254, 254, 0])

            tolerance = 50

            # Compute lower and upper bounds for each channel
            lower_bound = target_color - tolerance
            upper_bound = target_color + tolerance

            # Create a boolean mask with the tolerance applied to each channel
            mask = np.all((image_array >= lower_bound) & (image_array <= upper_bound), axis=-1)


            output_array[mask] = 15


            image_array = np.array(image)

            target_color =np.array([220, 169, 51])

            tolerance = 50

            # Compute lower and upper bounds for each channel
            lower_bound = target_color - tolerance
            upper_bound = target_color + tolerance

            # Create a boolean mask with the tolerance applied to each channel
            mask = np.all((image_array >= lower_bound) & (image_array <= upper_bound), axis=-1)


            output_array[mask] = 14

            # output_array[mask] = 14
            image_array = np.array(image)

            target_color =np.array([85, 85, 0])

            tolerance = 50

            # Compute lower and upper bounds for each channel
            lower_bound = target_color - tolerance
            upper_bound = target_color + tolerance

            # Create a boolean mask with the tolerance applied to each channel
            mask = np.all((image_array >= lower_bound) & (image_array <= upper_bound), axis=-1)


            output_array[mask] = 9


            image_array = np.array(image)

            target_color =np.array([85, 254, 169])

            tolerance = 50

            # Compute lower and upper bounds for each channel
            lower_bound = target_color - tolerance
            upper_bound = target_color + tolerance

            # Create a boolean mask with the tolerance applied to each channel
            mask = np.all((image_array >= lower_bound) & (image_array <= upper_bound), axis=-1)


            output_array[mask] = 17


            image_array = np.array(image)

            target_color =np.array([169, 254, 85])

            tolerance = 50

            # Compute lower and upper bounds for each channel
            lower_bound = target_color - tolerance
            upper_bound = target_color + tolerance

            # Create a boolean mask with the tolerance applied to each channel
            mask = np.all((image_array >= lower_bound) & (image_array <= upper_bound), axis=-1)

            output_array[mask] = 16

            image_array = np.array(image)

            target_color =np.array([47, 154, 47])

            tolerance = 50

            # Compute lower and upper bounds for each channel
            lower_bound = target_color - tolerance
            upper_bound = target_color + tolerance

            # Create a boolean mask with the tolerance applied to each channel
            mask = np.all((image_array >= lower_bound) & (image_array <= upper_bound), axis=-1)

            output_array[mask] = 9

            height, width = output_array.shape

            # Create a new 'P' mode image with the desired dimensions.
            img = Image.new('P', (width, height))
            img.putdata(output_array.flatten())
            my_palette = [0, 0, 0, 128, 0, 0, 254, 0, 0, 0, 85, 0, 169, 0, 51, 254, 85, 0, 0, 0, 85, 0, 119, 220, 85, 85, 0, 0, 85, 85, 85, 51, 0, 52, 86, 128, 0, 128, 0, 0, 0, 254, 51, 169, 220, 0, 254, 254, 85, 254, 169, 169, 254, 85, 254, 254, 0, 254, 169, 0, 20, 20, 20, 21, 21, 21, 22, 22, 22, 23, 23, 23, 24, 24, 24, 25, 25, 25, 26, 26, 26, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31, 31, 32, 32, 32, 33, 33, 33, 34, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 40, 40, 40, 41, 41, 41, 42, 42, 42, 43, 43, 43, 44, 44, 44, 45, 45, 45, 46, 46, 46, 47, 47, 47, 48, 48, 48, 49, 49, 49, 50, 50, 50, 51, 51, 51, 52, 52, 52, 53, 53, 53, 54, 54, 54, 55, 55, 55, 56, 56, 56, 57, 57, 57, 58, 58, 58, 59, 59, 59, 60, 60, 60, 61, 61, 61, 62, 62, 62, 63, 63, 63, 64, 64, 64, 65, 65, 65, 66, 66, 66, 67, 67, 67, 68, 68, 68, 69, 69, 69, 70, 70, 70, 71, 71, 71, 72, 72, 72, 73, 73, 73, 74, 74, 74, 75, 75, 75, 76, 76, 76, 77, 77, 77, 78, 78, 78, 79, 79, 79, 80, 80, 80, 81, 81, 81, 82, 82, 82, 83, 83, 83, 84, 84, 84, 85, 85, 85, 86, 86, 86, 87, 87, 87, 88, 88, 88, 89, 89, 89, 90, 90, 90, 91, 91, 91, 92, 92, 92, 93, 93, 93, 94, 94, 94, 95, 95, 95, 96, 96, 96, 97, 97, 97, 98, 98, 98, 99, 99, 99, 100, 100, 100, 101, 101, 101, 102, 102, 102, 103, 103, 103, 104, 104, 104, 105, 105, 105, 106, 106, 106, 107, 107, 107, 108, 108, 108, 109, 109, 109, 110, 110, 110, 111, 111, 111, 112, 112, 112, 113, 113, 113, 114, 114, 114, 115, 115, 115, 116, 116, 116, 117, 117, 117, 118, 118, 118, 119, 119, 119, 120, 120, 120, 121, 121, 121, 122, 122, 122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 127, 127, 127, 128, 128, 128, 129, 129, 129, 130, 130, 130, 131, 131, 131, 132, 132, 132, 133, 133, 133, 134, 134, 134, 135, 135, 135, 136, 136, 136, 137, 137, 137, 138, 138, 138, 139, 139, 139, 140, 140, 140, 141, 141, 141, 142, 142, 142, 143, 143, 143, 144, 144, 144, 145, 145, 145, 146, 146, 146, 147, 147, 147, 148, 148, 148, 149, 149, 149, 150, 150, 150, 151, 151, 151, 152, 152, 152, 153, 153, 153, 154, 154, 154, 155, 155, 155, 156, 156, 156, 157, 157, 157, 158, 158, 158, 159, 159, 159, 160, 160, 160, 161, 161, 161, 162, 162, 162, 163, 163, 163, 164, 164, 164, 165, 165, 165, 166, 166, 166, 167, 167, 167, 168, 168, 168, 169, 169, 169, 170, 170, 170, 171, 171, 171, 172, 172, 172, 173, 173, 173, 174, 174, 174, 175, 175, 175, 176, 176, 176, 177, 177, 177, 178, 178, 178, 179, 179, 179, 180, 180, 180, 181, 181, 181, 182, 182, 182, 183, 183, 183, 184, 184, 184, 185, 185, 185, 186, 186, 186, 187, 187, 187, 188, 188, 188, 189, 189, 189, 190, 190, 190, 191, 191, 191, 192, 192, 192, 193, 193, 193, 194, 194, 194, 195, 195, 195, 196, 196, 196, 197, 197, 197, 198, 198, 198, 199, 199, 199, 200, 200, 200, 201, 201, 201, 202, 202, 202, 203, 203, 203, 204, 204, 204, 205, 205, 205, 206, 206, 206, 207, 207, 207, 208, 208, 208, 209, 209, 209, 210, 210, 210, 211, 211, 211, 212, 212, 212, 213, 213, 213, 214, 214, 214, 215, 215, 215, 216, 216, 216, 217, 217, 217, 218, 218, 218, 219, 219, 219, 220, 220, 220, 221, 221, 221, 222, 222, 222, 223, 223, 223, 224, 224, 224, 225, 225, 225, 226, 226, 226, 227, 227, 227, 228, 228, 228, 229, 229, 229, 230, 230, 230, 231, 231, 231, 232, 232, 232, 233, 233, 233, 234, 234, 234, 235, 235, 235, 236, 236, 236, 237, 237, 237, 238, 238, 238, 239, 239, 239, 240, 240, 240, 241, 241, 241, 242, 242, 242, 243, 243, 243, 244, 244, 244, 245, 245, 245, 246, 246, 246, 247, 247, 247, 248, 248, 248, 249, 249, 249, 250, 250, 250, 251, 251, 251, 252, 252, 252, 253, 253, 253, 254, 254, 254, 255, 255, 255]
            img.putpalette(my_palette)
            # unique_elements = np.unique(np.array(img))
            # print(unique_elements)
            img.save("outputsavedarray_palette.png", optimize=False)

    return img




