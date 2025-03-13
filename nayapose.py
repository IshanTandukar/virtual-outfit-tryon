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


        def blend_transparent(background, overlay, alpha=0.5):
            """Blends overlay on background with given transparency"""
            return cv2.addWeighted(overlay, alpha, background, 1 - alpha, 0)

        def get_perpendicular_offset(point1, point2, thickness):
            """Returns perpendicular vector scaled to given thickness"""
            dx = point2[0] - point1[0]
            dy = point2[1] - point1[1]
            length = np.hypot(dx, dy)
            if length == 0:
                return (0, 0)
            offset_x = -dy * (thickness / length) // 2
            offset_y = dx * (thickness / length) // 2
            return int(offset_x), int(offset_y)

        def draw_tapered_line(canvas, prev_point, keypoint, color, thickness_start=5, thickness_middle=12):
            """Draws a tapered line from prev_point to keypoint."""
            overlay = canvas.copy()  # Create overlay for transparency

            # Compute the middle point
            mid_x = (prev_point[0] + keypoint[0]) // 2
            mid_y = (prev_point[1] + keypoint[1]) // 2
            mid_point = (mid_x, mid_y)

            # Get offsets for width control
            offset_start = get_perpendicular_offset(prev_point, mid_point, thickness_start)
            offset_mid = get_perpendicular_offset(prev_point, mid_point, thickness_middle)
            offset_end = get_perpendicular_offset(mid_point, keypoint, thickness_start)

            # Define polygon points for tapered effect
            points = np.array([
                (prev_point[0] + offset_start[0], prev_point[1] + offset_start[1]),
                (mid_point[0] + offset_mid[0], mid_point[1] + offset_mid[1]),
                (keypoint[0] + offset_end[0], keypoint[1] + offset_end[1]),
                (keypoint[0] - offset_end[0], keypoint[1] - offset_end[1]),
                (mid_point[0] - offset_mid[0], mid_point[1] - offset_mid[1]),
                (prev_point[0] - offset_start[0], prev_point[1] - offset_start[1])
            ], dtype=np.int32)

            # Draw the tapered shape
            cv2.fillPoly(canvas, [points], color)

            # Draw transparent circles for keypoints
            cv2.circle(overlay, prev_point, 7, color, -1)  # Start keypoint
            cv2.circle(overlay, keypoint, 7, color, -1)  # End keypoint

            # Blend the overlay with transparency
            canvas[:] = blend_transparent(canvas, overlay, alpha=0.5)



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
        copy_keypoints = pose_keypoints
        converted_keypoints = [(int(x), int(y)) for x, y in copy_keypoints[0]]

        print("IMAGE SHAPEDJ KDJDJFDJ",image.shape)
        
        canvas_size = (1024,768,3)
        canvas = np.zeros(canvas_size, dtype=np.uint8)

        # colors = [(119, 0, 156),(165, 0, 102),(169, 0, 155),(76, 0, 157),(163, 0, 47),(76, 168, 0),(160, 36, 0),(155, 106, 0),(145, 165, 0),(0, 169, 0),(0, 169, 0),(162, 0, 0),(6, 93, 156),(0, 168, 43)]
        colors = [(156, 0, 119),(102, 0, 165),(155, 0, 169),(157, 0, 76),(47, 0, 163),(0, 168, 76),(0, 36, 160),(0, 106, 155),(0, 165, 145),(0, 169, 0),(0, 169, 0),(0, 0, 162),(156, 93, 6),(43, 168, 0)]

        for i, (start, end) in enumerate(skeleton_connections):
            if start < len(converted_keypoints) and end < len(converted_keypoints):
                color = colors[i % len(colors)]  # Cycle through colors
                draw_tapered_line(canvas, converted_keypoints[start], converted_keypoints[end], color)
                # mid_x = (converted_keypoints[start][0] + converted_keypoints[end][0]) // 2
                # mid_y = (converted_keypoints[start][1] + converted_keypoints[end][1]) // 2

                # # Draw index number at midpoint
                # cv2.putText(canvas, str(i+1), (mid_x, mid_y),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the result
        imageko = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        img_np = np.array(imageko)


        hsv = cv2.cvtColor(img_np, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0], dtype=np.uint8)
        upper_black = np.array([50, 50, 50], dtype=np.uint8)  # Adjust this threshold if needed
        mask = cv2.inRange(hsv, lower_black, upper_black)

        # Increase brightness for non-black pixels
        increase_value = 50  # Adjust as needed
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + increase_value, 0, 255)

        # Apply the mask back to keep background black
        hsv[:, :, 2][mask > 0] = 0  # Set background brightness to zero (black)

        # Convert back to BGR
        bright_imgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        cv2.imwrite("brightened_image_path_black.png", bright_imgb)
        # cv2_imshow(bright_imgb)
        output_image = Image.fromarray(cv2.cvtColor(bright_imgb, cv2.COLOR_BGR2RGB))

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
        # output_image = Image.fromarray(output_image)

        # Write the JSON data to a file
        json_data = json.dumps(data, indent=2)
        with open("data.json", "w") as file:
            file.write(json_data)
        output_image.save("output_image.png")

        return output_image,data
