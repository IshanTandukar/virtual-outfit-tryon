# image_path_human_image = "00024_00.jpg"
# image_path_human_image = "inputs\image.png"
# image_path_human_image = "best.jpg"
import torch
import torchvision
import numpy as np
import cv2
from torchvision import transforms
import json
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load a pretrained Keypoint R-CNN model
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_pose_key_and_json(image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        def preprocess_image(image):
            """Preprocess image before keypoint detection."""
            # image = cv2.imread(image_path)
            # print(image)
            image = cv2.resize(image, (768, 1024))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            transform = transforms.Compose([transforms.ToTensor()])
            image_tensor = transform(image).unsqueeze(0).to(device)
            return image_tensor

        def detect_pose_keypoints(image_tensor, threshold=0.5):
            with torch.no_grad():
                output = model(image_tensor)

            if len(output[0]['keypoints']) == 0:
                return np.zeros((1, 17, 2))  # No detection case

            keypoints = output[0]['keypoints'][0, :, :2].cpu().numpy()  # Extract (x, y) keypoints
            return np.expand_dims(keypoints, axis=0)  # Shape (1, N, 2)

        def draw_skeleton_on_image(image, keypoints, skeleton):
            """
            Draws the keypoints and skeleton connections directly on the input image.

            Args:
                image_path (str): Path to the image.
                keypoints (ndarray): Detected keypoints of shape (1, N, 2).
                skeleton (list of tuple): Connections between keypoints.
            """
            # image = cv2.imread(image_path)
            image = cv2.resize(image, (768, 1024))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert for proper display

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
                            # cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 10, cv2.LINE_AA)

            return visualization


        def filter_skeleton_connections(keypoints, skeleton_connections):
            """ Removes edges if a keypoint is (0,0). """
            valid_edges = []
            for i, j in skeleton_connections:
                if not np.array_equal(keypoints[0, i], [0, 0]) and not np.array_equal(keypoints[0, j], [0, 0]):
                    valid_edges.append((i, j))
            return valid_edges

        # COCO 17-keypoint skeleton structure
        skeleton_connections = [
            (0,1),(0,2),(2,4),(1,3), #Face
            (0,5),(5,6),(5,7), #Upper torso
            (7,9),(9,12), #Left hand
            (6,8),(8,10), #Right hand
            (5,11), #Middle part
            (11,13),(11,14) #Bot part


            # (5, 6), (5, 7), (7, 9),  # Left arm
            # (6, 8), (8, 10),  # Right arm
            # (5, 11), (6, 12),  # Torso
            # (11, 13), (13, 15),  # Left leg
            # (12, 14), (14, 16),  # Right leg
            # (5, 1), (6, 2), (1, 3), (2, 4), (3, 5), (4, 6)  # Head connections
        ]

        # Load and process the image
        # human_image_path = image_path_human_image
        preprocessed_image = preprocess_image(image)
        pose_keypoints = detect_pose_keypoints(preprocessed_image)




        point_middle = (pose_keypoints[0][12]+pose_keypoints[0][11])/2
        new_element = point_middle.reshape(1, 2)
        updated_coords = np.insert(pose_keypoints, 10, new_element, axis=1)
        pose_keypoints = updated_coords

        point_middle_top = (pose_keypoints[0][5]+pose_keypoints[0][6])/2
        new_element_top = point_middle_top.reshape(1, 2)
        updated_coords_top = np.insert(pose_keypoints, 5, new_element_top, axis=1)
        pose_keypoints = updated_coords_top

        pose_keypoints_new = pose_keypoints[:, :15, :]

        pose_keypoints = pose_keypoints_new

        filtered_edges = filter_skeleton_connections(pose_keypoints, skeleton_connections)
        output_image = draw_skeleton_on_image(image, pose_keypoints, filtered_edges)

        arr = np.array(pose_keypoints)
        flat_list = arr.flatten().tolist()
        pose_keypoints_2d = []

        default_confidence = 1

        x = flat_list[0]
        y = flat_list[1]
        pose_keypoints_2d.extend([x, y, default_confidence])

        x = flat_list[10]
        y = flat_list[11]
        pose_keypoints_2d.extend([x, y, default_confidence])

        x = flat_list[14]
        y = flat_list[15]
        pose_keypoints_2d.extend([x, y, default_confidence])

        x = flat_list[18]
        y = flat_list[19]
        pose_keypoints_2d.extend([x, y, default_confidence])

        x = flat_list[24]
        y = flat_list[25]
        pose_keypoints_2d.extend([x, y, default_confidence])

        x = flat_list[12]
        y = flat_list[13]
        pose_keypoints_2d.extend([x, y, default_confidence])

        x = flat_list[16]
        y = flat_list[17]
        pose_keypoints_2d.extend([x, y, default_confidence])

        x = flat_list[20]
        y = flat_list[21]
        pose_keypoints_2d.extend([x, y, default_confidence])

        x = flat_list[22]
        y = flat_list[23]
        pose_keypoints_2d.extend([x, y, default_confidence])

        x = flat_list[28]
        y = flat_list[29]
        pose_keypoints_2d.extend([x, y, default_confidence])

        x = 0
        y = 0
        pose_keypoints_2d.extend([x, y, 0])

        x = 0
        y = 0
        pose_keypoints_2d.extend([x, y, 0])

        x = flat_list[26]
        y = flat_list[27]
        pose_keypoints_2d.extend([x, y, default_confidence])

        x = 0
        y = 0
        pose_keypoints_2d.extend([x, y, 0])

        x = 0
        y = 0
        pose_keypoints_2d.extend([x, y, 0])

        x = flat_list[4]
        y = flat_list[5]
        pose_keypoints_2d.extend([x, y, default_confidence])

        x = flat_list[2]
        y = flat_list[3]
        pose_keypoints_2d.extend([x, y, default_confidence])

        x = flat_list[8]
        y = flat_list[9]
        pose_keypoints_2d.extend([x, y, default_confidence])

        x = flat_list[6]
        y = flat_list[7]
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
        # output_image = np.array(output_image)
        # output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        output_image = Image.fromarray(output_image)

        # Write the JSON data to a file
        # json_data = json.dumps(data, indent=2)

        return output_image,data

