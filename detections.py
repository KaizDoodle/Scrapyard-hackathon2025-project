import cv2
import numpy as np
import time
import requests
import threading

# Load YOLO file
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []

# Replace with camera URL
url = 'http://10.0.3.252/cam-hi.jpg'

# Load class names from coco file
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Global variables for traffic light status and number of detections
current_light = "Green Light"
num_detections = 0

# Traffic light control function
def traffic_light_control():
    global current_light, num_detections
    while True:
        if num_detections > 2:
            green_time = 15
            red_time = 5
        else:
            green_time = 10
            red_time = 10

        current_light = "Green Light"
        print(current_light)
        time.sleep(green_time)
        current_light = "Yellow Light"
        print(current_light)
        time.sleep(2)
        current_light = "Red Light"
        print(current_light)
        time.sleep(red_time)

# Start the traffic light control in a separate thread
traffic_thread = threading.Thread(target=traffic_light_control)
traffic_thread.daemon = True
traffic_thread.start()

while True:
    try:
        # Load an image from the camera server
        img_resp = requests.get(url, stream=True)
        img_resp.raise_for_status()  # Raise an error for bad status codes
        img_resp.raw.decode_content = True
        imgnp = np.asarray(bytearray(img_resp.raw.read()), dtype=np.uint8)
        frame = cv2.imdecode(imgnp, -1)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image: {e}")
        time.sleep(1)
        continue

    # Run YOLO on the frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Initialize counter for the object (e.g., person)
    num_person = 0

    # Process YOLO output
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.9:  # Filter detections by confidence threshold
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])

                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                # Draw box and label on the frame
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                label = f"{classes[class_id]}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Increment counter if the detected object is a person
                if classes[class_id] == "bottle":
                    num_person += 1

    # Update the global number of detections
    num_detections = num_person

    # Print the current traffic light status
    print(f"Current traffic light: {current_light}")

    # Print the number of persons detected in the current frame
    if num_person > 0:
        print(f"Number of persons detected: {num_person}")

    # Display the frame with detections
    cv2.imshow("DISPLAYING : FRAME | Detections", frame)
    key = cv2.waitKey(5)  # Continue after 5 msec unless a keypress is registered, in which case break out of the loop
    if key == ord('q'):
        print("out")
        break

cv2.destroyAllWindows()