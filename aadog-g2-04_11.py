#!/usr/bin/env python3

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import matplotlib.pyplot as plt
import os

import logging

logging.getLogger().setLevel(logging.INFO)

# check if the deviations in the series of coordinates, needed to calculate the direction of movement, are below the assumed threshold
# reject a random error that occurs in collecting of depth coords X,Y,Z. The object is a list of data: [obj_id, detect_time, (xc, yc, X, Y),(....),(....)]
def check_deviation_of_depth_coords(obj, xc, yc, X, Y, Z):   # this function should be called if len(obj)>4
    #compute an average deviation of object coords for three last position
    dx = (abs(obj[2][0] - obj[3][0]) + abs(obj[3][0] - obj[4][0]))//2 * 2
    if dx == 0: dx = 5
    dy = (abs(obj[2][1] - obj[3][1]) + abs(obj[3][1] - obj[4][1]))//2 * 2
    if dy == 0: dy = 5
    dX = (abs(obj[2][2] - obj[3][2]) + abs(obj[3][2] - obj[4][2]))//2 * 2
    if dX < 10: dX = 50
    dY = (abs(obj[2][3] - obj[3][3]) + abs(obj[3][3] - obj[4][3]))//2 * 2
    if dY < 10: dY = 50
    dZ = (abs(obj[2][4] - obj[3][4]) + abs(obj[3][4] - obj[4][4]))//2 * 2
    if dZ < 10: dZ = 50

    # if xc, yc is within the mean deviation and X or Y is large ignore this
    if abs(obj[-1][0] - xc) <= dx and abs(obj[-1][1] - yc) <= dy and abs(obj[-1][2] - X) > dX*3:
        X = obj[-1][2]
    if abs(obj[-1][0] - xc) <= dx and abs(obj[-1][1] - yc) <= dy and abs(obj[-1][3] - Y) > dY*3:
        Y = obj[-1][3]
    if abs(obj[-1][0] - xc) <= dx and abs(obj[-1][1] - yc) <= dy and abs(obj[-1][4] - Z) > dZ*3:
        Z = obj[-1][4]

    return X,Y,Z


'''
Spatial detection network demo.
    Performs inference on RGB camera and retrieves spatial location coordinates: x,y,z relative to the center of depth map.
'''

## MobilenetSSD label texts
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# person-vehicle-bike-detection-crossroad-1016 label texts
#labelMap = ["bike", "vehicle", "person"]
#pedestrian-and-vehicle-detector-adas-0001

syncNN = True

# Get argument first
nnBlobPath = str((Path(__file__).parent / Path('models/mobilenet.blob')).resolve().absolute())
#nnBlobPath = str((Path(__file__).parent / Path('models/person-vehicle-bike-detection-crossroad-1016.blob')).resolve().absolute())

#nnBlobPath = str((Path(__file__).parent / Path('models/pedestrian-and-vehicle-detector-adas-0001.blob')).resolve().absolute())
#[14442C1051F310D100] [76.417] [SpatialDetectionNetwork(1)] [error] Input tensor 'data' (0) exceeds available data range. Data size (774144B), tensor offset (0), size (1548288B) - skipping inference

if len(sys.argv) > 1:
    nnBlobPath = sys.argv[1]

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
colorCam = pipeline.createColorCamera()
spatialDetectionNetwork = pipeline.createMobileNetSpatialDetectionNetwork()
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()

xoutRgb = pipeline.createXLinkOut()
xoutNN = pipeline.createXLinkOut()
xoutBoundingBoxDepthMapping = pipeline.createXLinkOut()
xoutDepth = pipeline.createXLinkOut()

xoutRgb.setStreamName("rgb")
xoutNN.setStreamName("detections")
xoutBoundingBoxDepthMapping.setStreamName("boundingBoxDepthMapping")
xoutDepth.setStreamName("depth")

#colorCam.setPreviewSize(672, 384) # preview output resized to fit the pedestrian-and-vehicle-detector-adas-0001.blob
#colorCam.setPreviewSize(512, 512) # preview output resized to fit the person-vehicle-bike-detection-crossroad-1016.blob
colorCam.setPreviewSize(300, 300)
colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
colorCam.setInterleaved(False)
colorCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# setting node configs
stereo.setOutputDepth(True)
stereo.setConfidenceThreshold(240) #255#600  set the confidence for disparity; set it to higher will cause less "holes"(values 0) but disparities/depths won't be as accurate

spatialDetectionNetwork.setBlobPath(nnBlobPath)
spatialDetectionNetwork.setConfidenceThreshold(0.5)
spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
spatialDetectionNetwork.setDepthLowerThreshold(100)
spatialDetectionNetwork.setDepthUpperThreshold(5000)

# Create outputs

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

colorCam.preview.link(spatialDetectionNetwork.input)
if(syncNN):
    spatialDetectionNetwork.passthrough.link(xoutRgb.input)
else:
    colorCam.preview.link(xoutRgb.input)

spatialDetectionNetwork.out.link(xoutNN.input)
spatialDetectionNetwork.boundingBoxMapping.link(xoutBoundingBoxDepthMapping.input)

stereo.depth.link(spatialDetectionNetwork.inputDepth)
spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)


# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
    # Start pipeline
    device.startPipeline()

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
    xoutBoundingBoxDepthMapping = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=4, blocking=False)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    frame = None
    detections = []

    startTime = time.monotonic()
    counter = 0
    fps = 0
    color = (255, 255, 255)

    # create lists to collect persons and vehicles last position (X,Y coordinates)
    cars, persons = [], [] 
    car_id = 0
    person_id = 0
    count = 0   # frame number

    #create lists for scatterplot data
    plotf, plotx, ploty, plotX, plotY = [], [], [], [], [] 

    while True:
        inPreview = previewQueue.get()
        inNN = detectionNNQueue.get()
        depth = depthQueue.get()

        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        frame = inPreview.getCvFrame()
        depthFrame = depth.getFrame()

        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
        detections = inNN.detections
        # create bb of a roi of depth data
        #if len(detections) != 0:
        #    boundingBoxMapping = xoutBoundingBoxDepthMapping.get()
        #    roiDatas = boundingBoxMapping.getConfigData()

        #    for roiData in roiDatas:
        #        roi = roiData.roi
        #        roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
        #        topLeft = roi.topLeft()
        #        bottomRight = roi.bottomRight()
        #        xmin = int(topLeft.x)
        #        ymin = int(topLeft.y)
        #        xmax = int(bottomRight.x)
        #        ymax = int(bottomRight.y)

        #        cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

        # prepare data for tracking:
        count += 1
        i = 0   # bb number in a frame
        detections_list = []

        # if the frame is available, draw bounding boxes on it and show the frame
        height = frame.shape[0]
        width  = frame.shape[1]
        for detection in detections:
            # denormalize bounding box
            x1 = int(detection.xmin * width)
            x2 = int(detection.xmax * width)
            y1 = int(detection.ymin * height)
            y2 = int(detection.ymax * height)
            #gets the center point of a bounding box
            xc, yc = (x2+x1)//2, (y2+y1)//2


            X = int(detection.spatialCoordinates.x)
            Y = int(detection.spatialCoordinates.y)
            Z = int(detection.spatialCoordinates.z)

            try:
                label = labelMap[detection.label]
            except:
                label = detection.label

            # sets object's id 
            if label != "car" or label != "person":
                obj_id = "other"
            else:
                obj_id = label

#---start tracking---------------- should be here def of tracking objects; collecting data for further computations of movement direction and collision point
            # updates person_id and localization in the frame
            if persons and (label == "person"):  # if list of cars is not empty and it's a car
                j = 0  #index of person in persons
                not_found = True
                # find if an object exist in the list, try until is not find
                while not_found:   
                    p = persons[j]  #predecessor data
                    if len(p) > 4:
                        X, Y, Z = check_deviation_of_depth_coords(p, xc, yc, X, Y, Z)
                    if (abs(p[-1][0]-xc) < 50) and (abs(p[-1][1]-yc) < 50) and (abs(p[-1][2]-X) < 500) and (abs(p[-1][3]-Y) < 500):
                        p_time = time.monotonic()
                        p[1] = p_time
                        # if it is not a "hole" value (depth measurement error), add new coordinates of an object
                        if X != 0 or Y != 0:
                            p.append((xc, yc, X, Y, Z))    # 
                        if len(p) > 5:  # leave only the last three positions of the person needed to calculate the direction of movement
                            del p[2]
                        not_found = False
                        continue
                    elif j < len(persons)-1:   # try to take next object from the list
                        j += 1
                    else:            # append a new object
                        # if it is not a "hole" value (depth measurement error), add a new object
                        if X != 0 or Y != 0:
                            p_time = time.monotonic()
                            person_id += 1
                            persons.append([person_id, p_time, (xc, yc, X, Y, Z)])   # append coordinates to the list as a new object position 
                            not_found = False
            elif label == "person":     # append the first object
                p_time = time.monotonic()
                persons.append([person_id, p_time, (xc, yc, X, Y, Z)])
            # check the list of objects to see if there's an object that has come out of a frame for more than 2sec
            l = [] # list of persons which has come out of a frame and should to be deleted from persons tracking list
            for c in range(len(persons)):
                if current_time - persons[c][1] > 2:
                    l.append(c)
            for e in l: del persons[e]

            # updates car_id and time and its last position in the frame
            if cars and (label == "car"):  # if list of cars is not empty and it's a car
                j = 0  #index of car in cars
                not_found = True
                # find if an object exist in the list, try until is not find
                while not_found:   
                    p = cars[j]  #predecessor data
                    if len(p) > 4:
                        X, Y, Z = check_deviation_of_depth_coords(p, xc, yc, X, Y, Z)
                    if (abs(p[-1][0]-xc) < 50) and (abs(p[-1][1]-yc) < 50) and (abs(p[-1][2]-X) < 500) and (abs(p[-1][3]-Y) < 500):
                        p_time = time.monotonic()
                        p[1] = p_time
                        # if it is not a "hole" value (depth measurement error), add new coordinates of an object
                        if X != 0 or Y != 0:
                            p.append((xc, yc, X, Y, Z))    # 
                        if len(p) > 5:  # leave only the last three positions of the car needed to calculate the direction of movement
                            del p[2]
                        not_found = False
                        continue
                    elif j < len(cars)-1:   # try to take next object from the list
                        j += 1
                    else:            # append a new object
                        # if it is not a "hole" value (depth measurement error), add a new object
                        if X != 0 or Y != 0:
                            p_time = time.monotonic()
                            car_id += 1
                            cars.append([car_id, p_time, (xc, yc, X, Y, Z)])   # append coordinates to the list as a new object position 
                            not_found = False
            elif label == "car":     # append the first object
                p_time = time.monotonic()
                cars.append([car_id, p_time, (xc, yc, X, Y, Z)])


            # updates the tracker  
            # check the list of objects to see if there's an object that has come out of a frame for more than 2sec
            l = [] # list of cars which has come out of a frame and should to be deleted from cars tracking list
            for c in range(len(cars)):
                if current_time - cars[c][1] > 2:
                    l.append(c)
            for e in l: del cars[e]
                            

            #fill out the checking list(for testing purpose)
            detections_list.append((i, label, xc, yc, X, Y, Z))

            i += 1

            ##collecting data for a scatterplot
            plotf.append(count)
            plotx.append(xc)
            ploty.append(yc)
            plotX.append(X)
            plotY.append(Y)
            # create a file with the data
            file = open('aadog_stats.txt', 'a')
            if X == 0 and Y == 0:
                file.write('>>>>>>>>>>>>>>>>>>>>>'+'\n')
                file.write('f:'+str(count)+', '+'xc: '+str(xc)+' yc: '+str(yc)+', '+'X: '+str(X)+' Y: '+str(Y)+' Z: '+str(Z)+'\n')
                #file.write('--------------------------------------------------'+'\n')
            else:
                file.write('f:'+str(count)+', '+'xc: '+str(xc)+' yc: '+str(yc)+', '+'X: '+str(X)+' Y: '+str(Y)+' Z: '+str(Z)+'\n')
            file.close()

#---end tracking-------------------
            
            # show data in the frame
            cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            #cv2.putText(frame, f"(X,Y,Z): {int(detection.spatialCoordinates.x)}, {int(detection.spatialCoordinates.y)}, {int(detection.spatialCoordinates.z)}mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

            cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)
            cv2.circle(frame, (xc, yc), 5, (0,0,255), -1)

            # COMPUTE AN EXTRAPOLATION LINE 
            pcp = []  # list of presumed cars positions
            for car in cars:
                if len(car) > 3:   # if there are in car at least two tuples with coords
                    # get the first and the last spacial position of a car
                    (X0,Y0,Z0), (X2,Y2,Z2) = car[2][2:], car[-1][2:]  
                    cp0 = np.array([X0,Y0,Z0])
                    # an equation of a line in 3D space: (x-x1)/l == (y-y1)/m == (z-z1)/n
                    # with direction vector for two points on a straight line: v = <l, m, n> = <x2-x1, y2-y1, z2-z1>
                    v = np.array([X2 - X0, Y2 - Y0, Z2 - Z0])
                    cp = cp0 + v  #next car possible position(point on the line)
                    # Limits calculations for X and Y to 100m==100,000mm in the field of view of the camera i.e. from -50m to + 50m, for Z to 200m
                    lim = 50000
                    # compute points of an extrapolation line for a current frame
                    while (cp[0]>=-lim and cp[0]<=lim and cp[1]>=-lim and cp[1]<=lim and cp[2]<=lim*4 ):
                        cp = cp + (10 * v) 
                    pcp.append((cp0,cp,v))   # append direction vector and first and last point on the line of hypothetical car movement
            ppp = []  # list of presumed persons positions
            for person in persons:
                if len(person) > 3:   # if there are in car at least two tuples with coords
                    # get the first and the last spacial position of a person
                    (X0,Y0,Z0), (X2,Y2,Z2) = person[2][2:], person[-1][2:]  
                    pp0 = np.array([X0,Y0,Z0])  # point zero == first from the last three person positions
                    # an equation of a line in 3D space: (x-x1)/l == (y-y1)/m == (z-z1)/n
                    # with direction vector v = <l, m, n> = <x2-x1, y2-y1, z2-z1>
                    v = np.array([X2 - X0, Y2 - Y0, Z2 - Z0])
                    pp = pp0 + v  #next car possible position(point on the imaginary line) as a sum of two vectors
                    # Limits of calculations for X and Y to 100m==100,000mm in the field of view of the camera i.e. from -50m to + 50m, for Z to 200m
                    lim = 50000
                    # compute points of an extrapolation line for a current frame
                    while (pp[0]>=-lim and pp[0]<=lim and pp[1]>=-lim and pp[1]<=lim and pp[2]<=lim*4 ):
                        pp = pp + (10 * v) 
                    ppp.append((pp0,pp,v))   # appends first and last point on the line of hypothetical car movement, also direction vector
            
            # COMPUTE AN INTERSECTION POINT IN GIVEN FRAME. 
            # calculations for each pair of car and person
            # from given two straight lines (lc for car, lp for person)
            # for lc: point [x,y,z] = cp0 + t1*v1,   # for lp: [x,y,z] = pp0 + t2*v2
            # if these two lines actually intersect at a point
            # from lc obtain:
            #x = cp0[0] + t1*v1[0]  #y = cp0[1] + t1*v1[1]  #z = cp0[2] + t1*v1[2] 
            ## from lp obtain:
            #x = pp0[0] + t2*v2[0]  #y = pp0[1] + t2*v2[1]  #z = pp0[2] + t2*v2[2] 
            ## equate
            #cp0[0] + t1*v1[0] = pp0[0] + t2*v2[0]
            # so: t1*v1[0] - t2*v2[0] = pp0[0] - cp0[0]
            #cp0[1] + t1*v1[1] = pp0[1] + t2*v2[1]
            # so: t1*v1[1] - t2*v2[1] = pp0[1] - cp0[1]
            #cp0[2] + t1*v1[2] = pp0[2] + t2*v2[2]
            # so: t1*v1[2] - t2*v2[2] = pp0[2] - cp0[2]
            if len(pcp) != 0 and len(ppp) != 0:   # if there is a car and a person detected
                for c in pcp:
                    for p in ppp:
                        x1, a1, x2, a2, y1, b1, y2, b2, z1, c1, z2, c2 = c[0][0], c[2][0], p[0][0], p[2][0], c[0][1], c[2][1], p[0][1], p[2][1], c[0][2], c[2][2], p[0][2], p[2][2]
                        t1 = (a2*(y2-y1) - b2*(x2-x1)) / (a2*b1-a1*b2)
                        t2 = (a1*(y2-y1) - b1*(x2-x1)) / (a2*b1-a1*b2)
                        if (t1 * c1) - (t2 * c2) == (z2 - z1):  # if these lines do intersect get intersection point as a np.array
                            intersection_point = c[0] + t1 * c[2]
                        # if these lines are skew :


# if the set of coefficients of the direction vector in two lines L1 and L2 are proportional these lines are parallel to each other
            #print('pcp: ', pcp)
            #a1 = pcp[0][0]


        #logging.info('detections_list: (i, label, x1, y1, X, Y)')
        print('\nF', count, current_time, detections_list)
        print('\nCars: F', count, cars)
        print('\nPersons: F', count, persons)


        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
        #cv2.imshow("depth", depthFrameColor)
        cv2.imshow("rgb", frame)

        #create scatterplot of y and Y data with gridlines
        plt.scatter(plotf, ploty, s=1)
        plt.scatter(plotf, plotY, s=1)
        plt.minorticks_on()
        plt.grid(which='minor')
        plt.grid(which='major')
        plt.xlabel("Frames")
        plt.ylabel("The value of the yc_bb(blue) and Y_depth(orange)")
        plt.title("Discontinuities in the occurrence of the Y, depth coordinates while bb is detected in the frame")
        if 500 < count < 502: plt.show()
        #plt.close()


        if cv2.waitKey(1) == ord('q'):
            break


