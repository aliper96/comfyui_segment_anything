import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the original image
image_path = "car.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib

# Load the detection results
with open("api_response.json", "r") as f:
    results = json.load(f)

# Create a copy of the image for drawing
image_with_contour = image.copy()

# Draw each detected object
for obj in results["objects"]:
    # Extract the bounding box
    bbox = obj["bbox"]
    x1, y1, x2, y2 = map(int, bbox)

    # Draw the bounding box
    cv2.rectangle(image_with_contour, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Add label and confidence
    label = f"{obj['label']}: {obj['confidence']:.2f}"
    cv2.putText(image_with_contour, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw contours
    for contour in obj["contours"]:
        # Reshape the contour data
        contour_array = np.array(contour).reshape(-1, 2)

        # Draw the contour
        cv2.polylines(image_with_contour, [contour_array], True, (255, 0, 0), 2)

        # Optionally, you can also fill the contour with a semi-transparent color
        # Create a mask for the contour
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [contour_array], 255)

        # Apply a semi-transparent overlay
        overlay = image_with_contour.copy()
        overlay[mask == 255] = [255, 0, 0]  # Red color for the segmentation
        cv2.addWeighted(overlay, 0.3, image_with_contour, 0.7, 0, image_with_contour)

# Display the images side by side
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(image_with_contour)
axes[1].set_title("Image with Detection")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("detection_result.png", dpi=300)
plt.show()

print("Results saved to 'detection_result.png'")