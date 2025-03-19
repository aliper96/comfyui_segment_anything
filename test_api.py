import requests
import os
import json

url = "http://localhost:8000/detect_objects"
image_path = "car.jpg"  # Update this with your actual image path

# Print file details to verify it exists
print(f"Testing with image: {image_path}")
print(f"File exists: {os.path.exists(image_path)}")
print(f"File size: {os.path.getsize(image_path)} bytes")

try:
    with open(image_path, "rb") as f:
        image_data = f.read()
        print(f"Successfully read {len(image_data)} bytes from file")

    files = {"image": (os.path.basename(image_path), open(image_path, "rb"), "image/jpeg")}
    data = {
        "prompt": "car",  # Try with a more specific prompt that matches your image
        "use_hq": "false",
        "confidence_threshold": "0.35",
        "box_threshold": "0.3"
    }

    print(f"Sending request with data: {data}")
    response = requests.post(url, files=files, data=data)
    print(f"Status code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Success: {result['success']}")
        print(f"Detected {len(result['objects'])} objects")

        # Print details of first object if any were detected
        if result['objects']:
            first_obj = result['objects'][0]
            print(f"First object: {first_obj['label']} with confidence {first_obj['confidence']}")
            print(f"Bounding box: {first_obj['bbox']}")

        # Save the full response to a file for inspection
        with open("api_response.json", "w") as f:
            json.dump(result, f, indent=2)
            print(f"Full response saved to api_response.json")
    else:
        print(f"Error response: {response.text}")

except Exception as e:
    print(f"Error: {e}")
finally:
    # Make sure to close any open file handles
    try:
        files["image"][1].close()
    except:
        pass