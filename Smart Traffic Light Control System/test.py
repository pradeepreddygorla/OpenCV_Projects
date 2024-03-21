import cv2
import imutils
import numpy as np
import urllib.request
import time

# port = serial.Serial("com1",9600,timeout=0.5)
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
cap = cv2.VideoCapture("4.mp4")
cap1 = cv2.VideoCapture("2.mp4")
# url="http://192.168.0.108:8080/shot.jpg"
bicycle1=0
car1=0
bus1=0
truck1=0
bicycle2=0
car2=0
bus2=0
truck2=0
vehicles1=0
vehicles2=0


def run(img,lane):
    global bicycle1
    global car1
    global bus1
    global truck1
    global bicycle2
    global car2
    global bus2
    global truck2
    global vehicles1
    global vehicles2
    img = imutils.resize(img, width=600)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.7:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if(lane==0):
                if(classes[class_ids[i]] == "bicycle"):
                    bicycle1+=1
                if(classes[class_ids[i]] == "car"):
                    car1+=1
                if(classes[class_ids[i]] == "bus"):
                    bus1+=1
                if(classes[class_ids[i]] == "truck"):
                    truck1+=1
                vehicles1= bicycle1+car1+bus1+truck1
            if(lane==1):
                if(classes[class_ids[i]] == "bicycle"):
                    bicycle2+=1
                if(classes[class_ids[i]] == "car"):
                    car2+=1
                if(classes[class_ids[i]] == "bus"):
                    bus2+=1
                if(classes[class_ids[i]] == "truck"):
                    truck2+=1
                vehicles2= bicycle2+car2+bus2+truck2
##                print("Number of vehicles: ",vehicles)
            if(lane==1):
                print("Number of vehicles west lane: ",vehicles1)
                print("Number of vehicle in east lane: ",vehicles2)
                if(vehicles1>=3 or vehicles2>=3):
                    if(vehicles1>vehicles2):
                        print("opening west lane")
                        # port.write('A'.encode())
                        #time.sleep(10)
                    if(vehicles1<vehicles2):
                        print("opening east lane")
                        # port.write('B'.encode())
                        #time.sleep(10)
                
            
            vehicles = ["bicycle","car","bus","truck"]
            for j in vehicles:
                if(label==j):
                    color = colors[i]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
                    cv2.putText(img, label, (x, y + 30), font, 3, color, 1)
    return(img)




while True:

    # imgPath=urllib.request.urlopen(url)
    # imgNp=np.array(bytearray(imgPath.read()),dtype=np.uint8)    
    # img=cv2.imdecode(imgNp,-1)
    _,img=cap.read()
    _,img1=cap1.read()
    for i in range(2):
        if(i == 0):
            img = run(img,i)
            cv2.imshow("West Lane", img)
        else:
            img1 = run(img1,i)
            cv2.imshow("East Lane", img1)
    bicycle1=0
    car1=0
    bus1=0
    truck1=0
    bicycle2=0
    car2=0
    bus2=0
    truck2=0
    vehicles1=0
    vehicles2=0
    
    # show the output frame


    # if the `q` key was pressed, break from the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
            break
cv2.destroyAllWindows()
