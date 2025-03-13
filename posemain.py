import os
import cv2
import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
# import matplotlib.pyplot as plt
from PIL import Image
import json
import torch
import torch.nn as nn
# from google.colab.patches import cv2_imshow

# BASE_PATH = "model/Final_epoch3.pth"  # Change this if models are in a different location
BASE_PATH = "C:/Users/ASUS/OneDrive/Desktop/anup/Self-Correction-Human-Parsing-master/model/Final_epoch3.pth"  # Change this if models are in a different location
# CLOTH_MASK_MODEL_PATH = os.path.join(BASE_PATH, "cloth_mask_model.h5")
# BODY_SEGMENTATION_MODEL_PATH = os.path.join(BASE_PATH, "u2net_modelnewnew2.h5")
POSE_ESTIMATION_MODEL_PATH = os.path.join(BASE_PATH, "Final_epoch3.pth")

# assert os.path.exists(CLOTH_MASK_MODEL_PATH), "Cloth Mask model not found!"
# assert os.path.exists(BODY_SEGMENTATION_MODEL_PATH), "Body Segmentation model not found!"
assert os.path.exists(POSE_ESTIMATION_MODEL_PATH), "Pose Estimation model not found!"

class DoubleConv(nn.Module):
  """(convolution => [BN] => ReLU) * 2"""

  def __init__(self, in_channels, out_channels, mid_channels=None):
    super().__init__()
    if not mid_channels:
        mid_channels = out_channels
    self.double_conv = nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )

  def forward(self, x):
    return self.double_conv(x)


class Down(nn.Module):
  """Downscaling with maxpool then double conv"""

  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.maxpool_conv = nn.Sequential(
        nn.MaxPool2d(2),
        DoubleConv(in_channels, out_channels)
        )

  def forward(self, x):
    return self.maxpool_conv(x)


class Up(nn.Module):
  """Upscaling then double conv"""

  def __init__(self, in_channels, out_channels, bilinear=True):
    super().__init__()

    # if bilinear, use the normal convolutions to reduce the number of channels
    if bilinear:
      self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
      self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
    else:
      self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
      self.conv = DoubleConv(in_channels, out_channels)


  def forward(self, x1, x2):
    x1 = self.up(x1)
    # input is CHW
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])
    # if you have padding issues, see
    # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
    # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
    x = torch.cat([x2, x1], dim=1)
    return self.conv(x)


class OutConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(OutConv, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

  def forward(self, x):
    return self.conv(x)

# copy-pasted and modified from unet_model.py

class UNet(nn.Module):
  def __init__(self, n_channels, n_landmarks, bilinear=True):
    super(UNet, self).__init__()
    self.n_channels = n_channels
    self.n_landmarks = n_landmarks
    self.bilinear = bilinear
    self.nchannels_inc = 8

    # define the layers

    # number of channels in the first layer
    nchannels_inc = self.nchannels_inc
    # increase the number of channels by a factor of 2 each layer
    nchannels_down1 = nchannels_inc*2
    nchannels_down2 = nchannels_down1*2
    nchannels_down3 = nchannels_down2*2
    # decrease the number of channels by a factor of 2 each layer
    nchannels_up1 = nchannels_down3//2
    nchannels_up2 = nchannels_up1//2
    nchannels_up3 = nchannels_up2//2

    if bilinear:
      factor = 2
    else:
      factor = 1

    self.layer_inc = DoubleConv(n_channels, nchannels_inc)

    self.layer_down1 = Down(nchannels_inc, nchannels_down1)
    self.layer_down2 = Down(nchannels_down1, nchannels_down2)
    self.layer_down3 = Down(nchannels_down2, nchannels_down3//factor)

    self.layer_up1 = Up(nchannels_down3, nchannels_up1//factor, bilinear)
    self.layer_up2 = Up(nchannels_up1, nchannels_up2//factor, bilinear)
    self.layer_up3 = Up(nchannels_up2, nchannels_up3//factor, bilinear)

    self.layer_outc = OutConv(nchannels_up3//factor, self.n_landmarks)

  def forward(self, x, verbose=False):
    x1 = self.layer_inc(x)
    if verbose: print(f'inc: shape = {x1.shape}')
    x2 = self.layer_down1(x1)
    if verbose:print(f'inc: shape = {x2.shape}')
    x3 = self.layer_down2(x2)
    if verbose: print(f'inc: shape = {x3.shape}')
    x4 = self.layer_down3(x3)
    if verbose: print(f'inc: shape = {x4.shape}')
    x = self.layer_up1(x4, x3)
    if verbose: print(f'inc: shape = {x.shape}')
    x = self.layer_up2(x, x2)
    if verbose: print(f'inc: shape = {x.shape}')
    x = self.layer_up3(x, x1)
    if verbose: print(f'inc: shape = {x.shape}')
    logits = self.layer_outc(x)
    if verbose: print(f'outc: shape = {logits.shape}')

    return logits

  def output(self, x, verbose=False):
    return torch.sigmoid(self.forward(x, verbose=verbose))

  def __str__(self):
    s = ''
    s += 'inc: '+str(self.layer_inc)+'\n'
    s += 'down1: '+str(self.layer_down1)+'\n'
    s += 'down2: '+str(self.layer_down2)+'\n'
    s += 'down3: '+str(self.layer_down3)+'\n'
    s += 'up1: '+str(self.layer_up1)+'\n'
    s += 'up2: '+str(self.layer_up2)+'\n'
    s += 'up3: '+str(self.layer_up3)+'\n'
    s += 'outc: '+str(self.layer_outc)+'\n'
    return s

  def __repr__(self):
    return str(self)


def heatmap2landmarks(hms):
  idx = np.argmax(hms.reshape(hms.shape[:-2] + (hms.shape[-2]*hms.shape[-1], )),
                  axis=-1)
  locs = np.zeros(hms.shape[:-2] + (2, ))
  locs[...,1],locs[...,0] = np.unravel_index(idx,hms.shape[-2:])
  return locs


model = UNet(n_channels=3, n_landmarks=18)
model.load_state_dict(torch.load("model/Final_epoch3.pth", weights_only=True))
model.eval()
def get_pose_key_and_json(image_path):
    def preprocess_pose(image_path):
        """Preprocess for pose estimation."""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (128, 128))
        image = np.transpose(image, (2, 0, 1))  # Convert to (C, H, W)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device) / 255.0
        return image



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def detect_pose_keypoints(image_path):
        """Detects pose keypoints on the input image using the pose model."""
        image = preprocess_pose(image_path)  # Preprocess image for pose model

        # Ensure the image is on the same device as the model
        image = image.to(device)

        model.to(device)

        with torch.no_grad():  # Disable gradients during inference
            pose_keypoints = model(image)  # Get keypoints from the pose model

        pose_keypoints = heatmap2landmarks(pose_keypoints.detach().cpu().numpy())  # Convert heatmap to landmarks
        return pose_keypoints

    # Example usage
    human_image_path = image_path
    pose_keypoints = detect_pose_keypoints(human_image_path)
    # print(pose_keypoints)


    point_middle = (pose_keypoints[0][8]+pose_keypoints[0][11])/2
    new_element = point_middle.reshape(1, 2)
    updated_coords = np.insert(pose_keypoints, 8, new_element, axis=1)
    pose_keypoints = updated_coords

    image = cv2.imread(human_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w, _ = image.shape
    input_size=(128, 128)
    scale_x = orig_w / input_size[0]
    scale_y = orig_h / input_size[1]
    for i in range(len(pose_keypoints[0])):
        x, y = pose_keypoints[0][i]
        pose_keypoints[0][i] = (x * scale_x, y * scale_y)


    def draw_skeleton_on_image(image_path, keypoints, skeleton):
        """
        Draws the keypoints and skeleton connections directly on the input image.

        Args:
            image_path (str): Path to the image.
            keypoints (ndarray): Detected keypoints of shape (1, N, 2).
            skeleton (list of tuple): Connections between keypoints.
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert for proper display
        # image = cv2.resize(image, (256, 256))

        height, width = image.shape[:2]
        visualization = np.zeros((height, width, 3), dtype=np.uint8)

        colors = [
            (255, 105, 180), (128, 0, 128), (75, 0, 130), (255, 192, 203),  # Face (Red)
            (155, 28, 49), (255, 255, 255), (255, 255, 255),  # Upper torso (Blue)
            (0, 255, 0), (0, 255, 0),  # Left hand (Green)
            (0, 255, 0), (0, 255, 0),  # Right hand (Yellow)
            (255, 182, 193),  # Middle part (Cyan)
            (0, 0, 200),(0, 255, 0)  # Bottom part (Magenta)
        ]

        for idx, (i, j) in enumerate(skeleton_connections):
            if i < len(keypoints[0]) and j < len(keypoints[0]):
                x1, y1 = int(keypoints[0][i][0]), int(keypoints[0][i][1])
                x2, y2 = int(keypoints[0][j][0]), int(keypoints[0][j][1])
                cv2.line(visualization, (x1, y1), (x2, y2), colors[idx], 8)


        for i, keypoint in enumerate(keypoints[0]):  # Access keypoints for the person
            x, y = int(keypoint[0]), int(keypoint[1])  # Unpack x, y from keypoint
            cv2.circle(visualization, (x, y), 10, (0, 0, 255), -1)  # Draw keypoint in red
            # cv2.putText(visualization, str(i), (int(x), int(y)),  # Label keypoint, convert x and y to integers
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 10, cv2.LINE_AA)


        visualization = Image.fromarray(visualization)
        return visualization
        # cv2.imwrite('/content/pose_image.jpg',visualization)
        # print(visualization.mode())


    def filter_skeleton_connections(keypoints, skeleton_connections):
        """ Removes edges if a keypoint is (0,0). """
        valid_edges = []
        for i, j in skeleton_connections:
            if not np.array_equal(keypoints[0, i], [0, 0]) and not np.array_equal(keypoints[0, j], [0, 0]):
                valid_edges.append((i, j))
        return valid_edges


    skeleton_connections = [
        # (0,1),(0,2),(2,4),(1,3), #Face
        # (0,5),(5,6),(5,7), #Upper torso
        # (7,9),(9,12), #Left hand
        # (6,8),(8,10), #Right hand
        # (5,11), #Middle part
        # (11,13),(11,14) #Bot part

        (0,15),(0,16),(16,18),(15,17),#FACE
        (0,1),(1,2),(1,5),#Upper ,torso
        (5,6),(6,7),#Left arm
        (2,3),(3,4),#Right arm
        (1,8),#middle ma
        (8,9),(8,12)#Both part


        # (5, 6), (5, 7), (7, 9),  # Left arm
        # (6, 8), (8, 10),  # Right arm
        # (5, 11), (6, 12),  # Torso
        # (11, 13), (13, 15),  # Left leg
        # (12, 14), (14, 16),  # Right leg
        # (5, 1), (6, 2), (1, 3), (2, 4), (3, 5), (4, 6)  # Head connections
    ]
    filtered_edges = filter_skeleton_connections(pose_keypoints, skeleton_connections)
    output_image = draw_skeleton_on_image(human_image_path, pose_keypoints, filtered_edges)

    arr = np.array(pose_keypoints)
    flat_list = arr.flatten().tolist()

    pose_keypoints_2d = []

    default_confidence = 1

    x = flat_list[0]
    y = flat_list[1]
    pose_keypoints_2d.extend([x, y, default_confidence])

    x = flat_list[2]
    y = flat_list[3]
    pose_keypoints_2d.extend([x, y, default_confidence])

    x = flat_list[4]
    y = flat_list[5]
    pose_keypoints_2d.extend([x, y, default_confidence])

    x = flat_list[6]
    y = flat_list[7]
    pose_keypoints_2d.extend([x, y, default_confidence])

    x = flat_list[8]
    y = flat_list[9]
    pose_keypoints_2d.extend([x, y, default_confidence])

    x = flat_list[10]
    y = flat_list[11]
    pose_keypoints_2d.extend([x, y, default_confidence])

    x = flat_list[12]
    y = flat_list[13]
    pose_keypoints_2d.extend([x, y, default_confidence])

    x = flat_list[14]
    y = flat_list[15]
    pose_keypoints_2d.extend([x, y, default_confidence])

    x = flat_list[16]
    y = flat_list[17]
    pose_keypoints_2d.extend([x, y, default_confidence])

    x = flat_list[18]
    y = flat_list[19]
    pose_keypoints_2d.extend([x, y, default_confidence])

    x = 0
    y = 0
    pose_keypoints_2d.extend([x, y, 0])

    x = 0
    y = 0
    pose_keypoints_2d.extend([x, y, 0])

    x = flat_list[24]
    y = flat_list[25]
    pose_keypoints_2d.extend([x, y, default_confidence])

    x = 0
    y = 0
    pose_keypoints_2d.extend([x, y, 0])

    x = 0
    y = 0
    pose_keypoints_2d.extend([x, y, 0])

    x = flat_list[30]
    y = flat_list[31]
    pose_keypoints_2d.extend([x, y, default_confidence])

    x = flat_list[32]
    y = flat_list[33]
    pose_keypoints_2d.extend([x, y, default_confidence])

    x = flat_list[34]
    y = flat_list[35]
    pose_keypoints_2d.extend([x, y, default_confidence])

    x = flat_list[36]
    y = flat_list[37]
    pose_keypoints_2d.extend([x, y, default_confidence])

    x = 0
    y = 0
    pose_keypoints_2d.extend([x, y, 0])

    x = 0
    y = 0
    pose_keypoints_2d.extend([x, y, 0])

    x = 0
    y = 0
    pose_keypoints_2d.extend([x, y, 0])

    x = 0
    y = 0
    pose_keypoints_2d.extend([x, y, 0])

    x = 0
    y = 0
    pose_keypoints_2d.extend([x, y, 0])

    x = 0
    y = 0
    pose_keypoints_2d.extend([x, y, 0])


    import json
    data = {
        "version": 1.3,
        "people": [
            {
                "person_id": [-1],
                "pose_keypoints_2d": pose_keypoints_2d,
                "face_keypoints_2d": [],
                "hand_left_keypoints_2d": [],
                "hand_right_keypoints_2d": [],
                "pose_keypoints_3d": [],
                "face_keypoints_3d": [],
                "hand_left_keypoints_3d": [],
                "hand_right_keypoints_3d": []
            }
        ]
    }

    # Write the JSON data to a file
    # json_data = json.dumps(data, indent=2)
    return output_image,data

