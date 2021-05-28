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

# check if deviations in the series of coordinates needed to calculate the direction of movement, are below an assumed threshold
# reject a random error that occurs in the collection of depth coords X,Y,Z. The object is the list of data: [obj_id, detect_time, (xc, yc, X, Y),(....),(....)]
def check_deviation_of_depth_coords(obj, xc, yc, X, Y, Z):   # this function should be called if len(obj)>6
    #compute an average deviation of object coords for three last positions and multiply by 2
    dx = (abs(obj[4][0] - obj[5][0]) + abs(obj[5][0] - obj[6][0]))//2 * 2
    if dx <= 5: dx = 5
    dy = (abs(obj[4][1] - obj[5][1]) + abs(obj[5][1] - obj[6][1]))//2 * 2
    if dy <= 5: dy = 5
    dX = (abs(obj[4][2] - obj[5][2]) + abs(obj[5][2] - obj[6][2]))//2 * 2
    if dX <= 10: dX = 50
    dY = (abs(obj[4][3] - obj[5][3]) + abs(obj[5][3] - obj[6][3]))//2 * 2
    if dY <= 10: dY = 50
    dZ = (abs(obj[4][4] - obj[5][4]) + abs(obj[5][4] - obj[6][4]))//2 * 2
    if dZ <= 10: dZ = 50

    # if xc, yc is within the mean deviation and value of X or Y is large ignore it
    if abs(obj[-1][0] - xc) <= dx and abs(obj[-1][1] - yc) <= dy and abs(obj[-1][2] - X) > dX*2:
        X = obj[-1][2]
    if abs(obj[-1][0] - xc) <= dx and abs(obj[-1][1] - yc) <= dy and abs(obj[-1][3] - Y) > dY*2:
        Y = obj[-1][3]
    if abs(obj[-1][0] - xc) <= dx and abs(obj[-1][1] - yc) <= dy and abs(obj[-1][4] - Z) > dZ*2:
        Z = obj[-1][4]

    return X,Y,Z


# updating objects data; 
# updates object_id, detection time, (two points of an object spatial location on a line of movement and the line's direction vector), 
# [line x plane intersection point], (cordinates of bounding box and depth coords) of the last three positions
def update_object_data(objects, obj_id, obj_label, xc, yc, X, Y, Z):
    if objects and (label == obj_label):  # e.g.: if list of cars is not empty and it's a "car"
        j = 0  #index of object in objects
        not_found = True
        # find if an object exist in the list, try until is not find
        while not_found:   
            obj = objects[j]  #predecessor data
            if len(obj) > 6:
                X, Y, Z = check_deviation_of_depth_coords(obj, xc, yc, X, Y, Z)
            if (abs(obj[-1][0]-xc) < 50) and (abs(obj[-1][1]-yc) < 50) and (abs(obj[-1][2]-X) < 500) and (abs(obj[-1][3]-Y) < 500) and (abs(obj[-1][4]-Z) < 1000):
                p_time = time.monotonic()
                # if it is not a "hole" value (depth measurement error), add new coordinates of an object
                if X != 0 or Y != 0:
                    obj[1] = p_time
                    obj.append((xc, yc, X, Y, Z))    # 
                if len(obj) > 7:  # leave only the last three positions of the object, needed to calculate the direction of movement
                    del obj[4]
                not_found = False
                continue
            elif j < len(objects)-1:   # try to take next object from the list
                j += 1
            else:            # append a new object
                # if it is not a "hole" value (depth measurement error), add a new object
                if X != 0 or Y != 0:
                    p_time = time.monotonic()
                    obj_id += 1
                    objects.append([obj_id, p_time, ([0,0,0],[0,0,0],[0,0,0]), [], (xc, yc, X, Y, Z)])   # append coordinates to the list as a new object position 
                    not_found = False
    elif label == obj_label:     # append the first object
        p_time = time.monotonic()
        # append obj id, last possition detection time, extrapolation line parameters(p0,pn,v), [intersection point coords and colliding obj id-s], spatial position
        persons.append([obj_id, p_time, ([0,0,0],[0,0,0],[0,0,0]), [], (xc, yc, X, Y, Z)])
            
    # check the list of objects to see if there's an object that has come out of a frame for more than 2sec
    l = [] # list of indices of objects which has come out of a frame and should to be deleted from objects tracking list
    for i in range(len(objects)):
        if current_time - objects[i][1] > 2:
            l.append(i)
    objects = [obj for idx,obj in enumerate(objects) if idx not in l]

    return objects



# COMPUTE AN EXTRAPOLATION LINE
# extrapolation of the path of the object movement 
def compute_extrapolation_line(objects):             
    # add the first and the last point of presumed object position and direction vector of a trajectory straight line
    for obj in objects:
        if len(obj) > 6:   # if there are in car at least two tuples with coords
            # get the first and the last spacial position of a car
            (X0,Y0,Z0), (X2,Y2,Z2) = obj[4][2:], obj[-1][2:]  
            cp0 = np.array([X0,Y0,Z0])
            # an equation of a line in 3D space: (x-x1)/l == (y-y1)/m == (z-z1)/n
            # with direction vector for two points on a straight line: v = <l, m, n> = <x2-x1, y2-y1, z2-z1>
            v = np.array([X2 - X0, Y2 - Y0, Z2 - Z0])
            cp = cp0 + v  #next car possible position(point on the line)
            # ??! Limits calculations for the depth X and Y to 50m==50,000mm in the field of view of the camera i.e. from -25m to + 25m, for Z to 50m
            lim = 25000 
            # compute points of an extrapolation line for a current frame
            while (cp[0]>=-lim and cp[0]<=lim and cp[1]>=-lim and cp[1]<=lim and cp[2]<=lim*2 ):
                cp = cp + (1000 * v) 
            obj[2] = (cp0, cp, v)   # append direction vector v and first and last position of the object on the line of its hypothetical movement

    return objects


# COMPUTE AN INTERSECTION POINTS
def add_pedestrian_car_collision_points(cars, persons):
    # calculate for each pair of car and person
    if len(cars) != 0 and len(persons) != 0:   # if there is a car and a person detected
        for c in cars:
            if len(c) > 6:

                print('car id: ', c[0])

                for p in persons:
                    if len(p) > 6:

                        print('person id: ', p[0])

                        ## if the set of coefficients of the direction vectors of two lines that is routs of a car and a person, are proportional these lines are parallel to each other
                        #and their direction vectors Cross Product equals 0
                        if not any(np.cross(c[2][2], p[2][2])):   # so if lines are not parallel
                            # get a point on the car line and its direction vector coefficients 
                            x_1, a_1, y_1, b_1, z_1, c_1 = c[2][0][0], c[2][2][0], c[2][0][1], c[2][2][1], c[2][0][2], c[2][2][2]

                            print('y_1: {},  z_1: {}'.format(c[2][0][1], c[2][0][2]))
                            print('a_1: {}, b_1: {},  c_1: {}'.format(c[2][2][0], c[2][2][1], c[2][2][2]))

                            # car line equation: x=a_1*t+x_1, y=b_1*t+y_1, z=c_1*t+z_1
                            # a normal vector n to a person's plane it is a np.cross product of two vectors (pp0->pp1, pp0->pp2) 
                            # where pp0, pp1, pp2 are three points in the plane, and pp2 differs from pp0 only by the value of the y coordinate increased by 10
                            # vector pp0->pp1 == direction vector == p[2][2]
                            # vector pp0->pp2 == [p[2][0][0] - p[2][0][0], p[2][0][1] + 10 - p[2][0][1], p[2][0][2] - p[2][0][2]] == [0, 10, 0]

                            # Get a normal vector of a pedestrian's plane
                            n = np.cross(p[2][2], np.array([0, 10, 0]))
                            nx, ny, nz = n[0], n[1], n[2]

                            print('nx: {}, ny: {}, nz: {}'.format(nx, ny, nz))

                            # get a point on the plane
                            x0, y0, z0 = p[2][0][0], p[2][0][1], p[2][0][2] #coords of pp0 
                            # an equation of the plane: nx*(x-x0) + ny*(y-y0) + nz*(z-z0) == 0, so:
                            d = -(nx*x0 + ny*y0 + nz*z0)

                            # as nx*x + ny*y + nz*z + d = 0
                            # so nx*(a_1*t + x_1) + n_y*(b_1*t + y_1) + nz*(c_1*t + z_1) + d == 0

                            if (nx*a_1 + ny*b_1 + nz*c_1) != 0:
                                t = -(d + nx*x_1 + ny*y_1 + nz*z_1) / (nx*a_1 + ny*b_1 + nz*c_1)

                                print('t: ', t)

                                xi = a_1*t + x_1
                                yi = b_1*t + y_1
                                zi = c_1*t + z_1
                                c[3].append((xi, yi, zi, p[0]))  # insert intersection coords and an id of a person the car can collide with
                                p[3].append((xi, yi, zi, c[0]))  # insert intersection coords and an id of a car the person can collide with

                            print('END OF THE LOOP OF SEARCHING FOR THE CROSSING POINT OF A CAR WITH A PEDESTRIAN PLANE ')

    return persons, cars




    # Draw bounding box, central bb point; show confidence, object label and object spatial data in the frame
def show_data_in_frame(frame, fps, label, x1, y1, x2, y2, xc, yc, X, Y, Z, color):
    cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
    if label == "person" or label == "car":
        cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"X: {X} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"Y: {Y} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.putText(frame, f"Z: {Z} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)
        cv2.circle(frame, (xc, yc), 5, (0,0,255), -1)
    return





# Spatial detection network demo.
# Performs inference on RGB camera and retrieves spatial location coordinates: x,y,z relative to the center of depth map.

# MobilenetSSD label texts
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
spatialDetectionNetwork.setDepthUpperThreshold(15000)

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

        # prepare data for tracking:
        count += 1  # frame number
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

            # updates objects id and localization
            persons = update_object_data(persons, person_id, "person", xc, yc, X, Y, Z)
            cars = update_object_data(cars, car_id, "car", xc, yc, X, Y, Z)

            # calculate the projected path and direction of movement of the object
            persons = compute_extrapolation_line(persons)
            cars = compute_extrapolation_line(cars)

            # calculate possible collision points
            persons, cars = add_pedestrian_car_collision_points(cars, persons)

            show_data_in_frame(frame, fps, label, x1, y1, x2, y2, xc, yc, X, Y, Z, color)

            #fill out the checking list(for testing purpose)
            detections_list.append((i, label, xc, yc, X, Y, Z))

            i += 1

            ## COLLECT DATA IN A FILE AND DRAW A PLOT
            ## collecting data for a scatterplot
            #plotf.append(count)
            #plotx.append(xc)
            #ploty.append(yc)
            #plotX.append(X)
            #plotY.append(Y)

            ## create a file with the data
            #file = open('aadog_stats.txt', 'a')
            #if X == 0 and Y == 0:
            #    file.write('>>>>>>>>>>>>>>>>>>>>>'+'\n')
            #    file.write('f:'+str(count)+', '+'xc: '+str(xc)+' yc: '+str(yc)+', '+'X: '+str(X)+' Y: '+str(Y)+' Z: '+str(Z)+'\n')
            #    #file.write('--------------------------------------------------'+'\n')
            #else:
            #    file.write('f:'+str(count)+', '+'xc: '+str(xc)+' yc: '+str(yc)+', '+'X: '+str(X)+' Y: '+str(Y)+' Z: '+str(Z)+'\n')
            #file.close()

            

        print('\nF', count, current_time, detections_list)
        print('\nCars: F', count, cars)
        print('\nPersons: F', count, persons)

        
        cv2.imshow("rgb", frame)

        ##create scatterplot of y and Y data with gridlines
        #plt.scatter(plotf, ploty, s=1)
        #plt.scatter(plotf, plotY, s=1)
        #plt.minorticks_on()
        #plt.grid(which='minor')
        #plt.grid(which='major')
        #plt.xlabel("Frames")
        #plt.ylabel("The value of the yc_bb(blue) and Y_depth(orange)")
        #plt.title("Discontinuities in the occurrence of the Y, depth coordinates while bb is detected in the frame")
        #if 500 < count < 502: plt.show()
        ##plt.close()


        if cv2.waitKey(1) == ord('q'):
            break


