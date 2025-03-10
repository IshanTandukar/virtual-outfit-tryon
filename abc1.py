import cv2
import numpy as np

input_ko_path = "output\\00009_00.png"  # Use \\ or / for Windows paths

# Load the image
image = cv2.imread(input_ko_path)
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
cv2.imwrite("masked1.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
# print("Image processed and saved as masked1.png")

# Optional: Display (uncomment if running in an environment with a display)
# cv2.imshow("Result", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np

# Load the image
image = cv2.imread('masked1.png')  # Replace with your image path
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

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
cv2.imwrite("masked2.png",cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np

# Load the image
image = cv2.imread('masked2.png')  # Replace with your image path
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

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
cv2.imwrite("masked3.png",cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np

# Load the image
image = cv2.imread('masked3.png')  # Replace with your image path
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

# Define the RGB range for masking
lower_bound = np.array( [152, 100, 108], dtype=np.uint8)  # Example lower RGB bound
upper_bound = np.array( [222, 158, 159], dtype=np.uint8)  # Example upper RGB bound

# Create a mask
mask = cv2.inRange(image, lower_bound, upper_bound)

# Create a blue image with the same shape as the original
blue_color = np.full(image.shape, (0, 254, 254), dtype=np.uint8)

# Replace the masked region with blue
image[mask > 0] = blue_color[mask > 0]

# Show results
# cv2_imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
cv2.imwrite("masked4.png",cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np

# Load the image
image = cv2.imread('masked4.png')  # Replace with your image path
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
cv2.imwrite("masked5.png",cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np

# Load the image
image = cv2.imread('masked5.png')  # Replace with your image path
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

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
cv2.imwrite("masked6.png",cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np

# Load the image
image = cv2.imread('masked6.png')  # Replace with your image path
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

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
cv2.imwrite("masked7.png",cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np

# Load the image
image = cv2.imread('masked7.png')  # Replace with your image path
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

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
cv2.imwrite("masked8.png",cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)
# cv2.destroyAllWindows()


