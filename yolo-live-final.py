# Importing Libraries

import numpy as np
import cv2
import time

"""
Reading stream video from camera
"""

camera = cv2.VideoCapture(0)

h, w = None, None

"""
Loading YOLO V3 Network
"""

# Loading COCO class labels from file and opening it
with open('yolo-coco-data/coco.names') as f:
    labels = [line.strip() for line in f]

# Loading trained YOLO v3 Object Detector
network = cv2.dnn.readNetFromDarknet('yolo-coco-data/yolov3.cfg',
                                     'yolo-coco-data/yolov3.weights')

layers_names_all = network.getLayerNames()
layers_names_output = \
    [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]

# Setting minimum probability and threshold values
probability_minimum = 0.5
threshold = 0.3

# Generating colours for representing every detected objects
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

"""
Reading frames in the loop
"""

while True:
    _, frame = camera.read() # capturing frame-by-frame from camera
    if w is None or h is None:
        h, w = frame.shape[:2] # getting spatial dimensions of the frame

    # Getting the blob from current frame
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)

    # Implementing forward pass with our blob
    network.setInput(blob)  # setting blob as input to the network
    start = time.time()
    output_from_network = network.forward(layers_names_output)
    end = time.time()
    print('Current frame took {:.5f} seconds'.format(end - start))

    # Preparing lists for detecting bounding boxes
    bounding_boxes = []
    confidences = []
    class_numbers = []

    # Going through all output layers and detections
    for result in output_from_network:
        for detected_objects in result:
            scores = detected_objects[5:]  # getting 80 classes' probabilities for current output layer
            class_current = np.argmax(scores)   # getting index of the class with max value of probability
            confidence_current = scores[class_current]

            # Eliminating weak predictions with minimum probability
            if confidence_current > probability_minimum:
                box_current = detected_objects[0:4] * np.array([w, h, w, h])  # scaling bounding box
                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                # Adding results into prepared lists
                bounding_boxes.append([x_min, y_min,
                                       int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)

    # Implementing Non-maximum suppression filtering
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)

    # Checking if there is at least one detected object
    if len(results) > 0:
        for i in results.flatten():
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

            colour_box_current = colours[class_numbers[i]].tolist()

            cv2.rectangle(frame, (x_min, y_min), (x_min + box_width, y_min + box_height),
                          colour_box_current, 2)
            text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                   confidences[i])

            cv2.putText(frame, text_box_current, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        colour_box_current, 2)

    # Showing results obtained from camera in real time
    cv2.namedWindow('YOLO v3 Real Time Detections', cv2.WINDOW_NORMAL)
    cv2.imshow('YOLO v3 Real Time Detections', frame)

    # Breaking the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing camera and destroying all opened windows
camera.release()
cv2.destroyAllWindows()