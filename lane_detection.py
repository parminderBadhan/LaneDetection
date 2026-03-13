import cv2
import numpy as np
import time

def canny(image):
    grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurImg = cv2.GaussianBlur(grayImg, (5, 5), 0)
    cannyImg = cv2.Canny(blurImg, 50, 150)
    return cannyImg

def limit(image):
    height = image.shape[0]
    width = image.shape[1]
    top_width = int(width / 3)
    bottom_width = int(width * 4 / 5)
    trapezoid_height = int(height / 4)
    position = height - trapezoid_height

    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.array([[
        [int((width - top_width) / 2), position],
        [int((width + top_width) / 2), position],
        [int((width + bottom_width) / 2), position + trapezoid_height],
        [int((width - bottom_width) / 2), position + trapezoid_height],
    ]], dtype=np.int32)

    points = points.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [points], color=255)
    trapezoid_cutout = cv2.bitwise_and(image, image, mask=mask)

    return trapezoid_cutout

def slope_filter(lines, width, treshold=0.4):
    filtered_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            if y1 > y2:
                y1, y2 = y2, y1
                x1, x2 = x2, x1

            slope = (y2 - y1) / (x2 - x1)
            if x1 < width / 2:
                if slope < -treshold:
                    filtered_lines.append([x1, y1, x2, y2])
            else:
                if slope > treshold:
                    filtered_lines.append([x1, y1, x2, y2])

    return filtered_lines

def closest(lines, width):
    largest = [-1e9, 0, 0, 0]
    smallest = [1e9, 0, 0, 0]
    for line in lines:
        if line[0] < width / 2 and line[0] > largest[0]:
            largest = line
        if line[0] > width / 2 and line[0] < smallest[0]:
            smallest = line

    res = [None, None]
    if largest[0] != -1e9:
        res[0] = largest
    if smallest[0] != 1e9:
        res[1] = smallest

    return res

def detect_objects(image):
    try:
        # Load YOLO pre-trained model and configuration
        net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
        classes = []
        with open("Data/coco.names", "r") as f:
            classes = [line.strip() for line in f]

        # Load image
        height, width, _ = image.shape

        # Preprocess the image for the YOLO model
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (256, 256), swapRB=True, crop=False)
        net.setInput(blob)

        # Get output layer names
        layer_names = net.getUnconnectedOutLayersNames()

        # Run forward pass to get predictions
        outs = net.forward(layer_names)

        # Process the results to draw bounding boxes
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    # Bounding box coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Draw bounding box
                    color = (0, 255, 0)  # BGR format
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

                    # Add class label and confidence
                    label = f"{classes[class_id]}: {confidence:.2f}"
                    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return image
    except Exception as e:
        print(f"Error during object detection: {e}")
        return image

# Start Video Processing
test = 'input_video2.mp4'  # Specify your video file here
cap = cv2.VideoCapture(test)
_, shp = cap.read()

prevLines = [[0, 0, 0, 0], [0, 0, 0, 0]]

# Initialize variables for FPS calculation
prev_time = time.time()

while cap.isOpened():
    _, frame = cap.read()
    width = frame.shape[1]
    height = frame.shape[0]

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Draw FPS on the frame (move it a little to the left)
    cv2.putText(frame, f"FPS: {fps:.2f}", (frame.shape[1] - 150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    cannyImg = canny(frame)
    croppedImg = limit(cannyImg)
    lines = cv2.HoughLinesP(croppedImg, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=40)

    frame = detect_objects(frame)

    if lines is not None:
        lines = slope_filter(lines, width)
        lines = closest(lines, width)
        for i in range(2):
            if lines[i] == None:
                lines[i] = prevLines[i]
            else:
                for j in range(4):
                    lines[i][j] = int(lines[i][j] * 0.1 + prevLines[i][j] * 0.9)

            x1, y1, x2, y2 = lines[i]
            x21, y21 = int(x1 + (x1 - x2) / 2), int(y1 + (y1 - y2) / 2)

            cv2.line(frame, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=5)
            prevLines[i] = lines[i]

    # If two lines are detected, fill the area between them
    if lines[0] is not None and lines[1] is not None:
        x1, y1, x2, y2 = lines[0]
        x3, y3, x4, y4 = lines[1]
        
        # Midpoint between the two lines
        x21, y21 = int(x1 + (x1 - x2) / 2), int(y1 + (y1 - y2) / 2)
        x41, y41 = int(x3 + (x3 - x4) / 2), int(y3 + (y3 - y4) / 2)

        # Create a polygon to fill between the two lines
        pts = np.array([[x1, y1], [x2, y2], [x4, y4], [x3, y3]], np.int32)
        pts = pts.reshape((-1, 1, 2))

        # Fill the polygon with a light red color
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 255))  # Light red
        opacity = 0.4  # Adjust opacity
        frame = cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0)

    # Show the frame with FPS counter
    cv2.imshow(f"Lane Detection: {test} {shp.shape[:-1]} - press 'c to close window", frame)

    if cv2.waitKey(25) == ord('c'):
        cap.release()
        cv2.destroyAllWindows()