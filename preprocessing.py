import numpy as np
import time
import cv2
import imutils
from imutils import face_utils
import dlib
import argparse
import os
import json

color = np.random.randint(0, 255, (100, 3))

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


protoFile = "/Users/shreya/Downloads/HandPose/hand/pose_deploy.prototxt"
weightsFile = "/Users/shreya/Downloads/HandPose/hand/pose_iter_102000.caffemodel"
nPoints = 22
POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]

threshold = 0.2


# Loop over video while tracking
def makeVideo(cap,n):


    hasFrame, frame = cap.read()

    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray,
                                 mask=None,
                                 maxCorners=300,
                                 qualityLevel=0.3,
                                 minDistance=7,
                                 blockSize=7)

    mask = np.zeros(frame.shape, frame.dtype)

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    aspect_ratio = frameWidth / frameHeight

    inHeight = 368
    inWidth = int(((aspect_ratio * inHeight) * 8) // 8)

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    k = 0

    count = 0
    fourc = cv2.VideoWriter_fourcc('M','J','P','G')
    vid_writer = cv2.VideoWriter(n,fourc, 15, (frameWidth, frameHeight))
    print("w1 ", frameWidth, "h", frameHeight)

    while True:
        ret, frame = cap.read()
        t = time.time()
        count = count+1
        if(ret==False):
            break
        if(count % 3 == 0):


            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            (height, width, depth) = frame.shape
            rMask = np.zeros((height, width, 1), np.uint8)

            # detect faces in the grayscale frame
            rects = detector(gray, 0)

            # loop over the face detections
            for rect in rects:
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                (x, y, w, h) = face_utils.rect_to_bb(rect)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # you can remove this chunk of code later
                (height, width, depth) = frame.shape
                gImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # rMask = np.zeros((height, width, 1), np.uint8)
                rMask[y:y + h, x:x + w] = 255
                # frame = cv2.bitwise_and(frame, frame, mask=rMask)

                # loop over the (x, y)-coordinates for the facial landmarks
                # and draw them on the image
                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)



            inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                            (0, 0, 0), swapRB=False, crop=False)

            net.setInput(inpBlob)

            output = net.forward()

            #print("forward = {}".format(time.time() - t))

            # Empty list to store the detected keypoints
            points = []
            frameCopy = np.copy(frame)

            for i in range(nPoints):
                # confidence map of corresponding body's part.
                probMap = output[0, i, :, :]
                probMap = cv2.resize(probMap, (frameWidth, frameHeight))

                # Find global maxima of the probMap.
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

                if prob > threshold:
                    cv2.circle(frameCopy, (int(point[0]), int(point[1])), 6, (0, 255, 255), thickness=-1,
                               lineType=cv2.FILLED)
                    cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, .8,
                                (0, 0, 255), 2, lineType=cv2.LINE_AA)

                    # Add the point to the list if the probability is greater than the threshold
                    points.append((int(point[0]), int(point[1])))
                else:
                    points.append(None)

            # Draw Skeleton
            frameWidth = frame.shape[1]
            frameHeight = frame.shape[0]

            leftx = frameWidth
            rightx = 0
            topy = frameHeight
            bottomy = 0



            for pair in POSE_PAIRS:
                partA = pair[0]
                partB = pair[1]

                if points[partA] and points[partB]:
                    cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2, lineType=cv2.LINE_AA)
                    cv2.circle(frame, points[partA], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                    cv2.circle(frame, points[partB], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

                    if (points[partA][0] < leftx):
                        leftx = points[partA][0]
                    if (points[partB][0] < leftx):
                        leftx = points[partB][0]
                    if (points[partA][0] > rightx):
                        rightx = points[partA][0]
                    if (points[partB][0] > rightx):
                        rightx = points[partA][0]
                    if (points[partA][1] < topy):
                        topy = points[partA][1]
                    if (points[partB][1] < topy):
                        topy = points[partB][1]
                    if (points[partA][1] > bottomy):
                        bottomy = points[partA][1]
                    if (points[partB][1] > bottomy):
                        bottomy = points[partA][1]

            (height, width, depth) = frame.shape
            # gImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # rMask = np.zeros((height, width, 1), np.uint8)

            rMask[topy:bottomy, leftx:rightx] = 255
            #frame = cv2.bitwise_and(frame, frame, mask=rMask)

            #print("Time Taken for frame = {}".format(time.time() - t))
            #"leftx ", leftx, "  rightx ", rightx, " topy ", topy, " bottomy ", bottomy)
            # cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
            # cv2.putText(frame, "Hand Pose using OpenCV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 0), 2, lineType=cv2.LINE_AA)
            #cv2.imshow("First", frame)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None,
                                                   winSize=(15, 15),
                                                   maxLevel=2,
                                                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
            frame = cv2.add(frame, mask)

            frame = cv2.bitwise_and(frame, frame, mask=rMask)

            print("w ", frame.shape[1], "h", frame.shape[0])
            #frame = cv2.resize(frame, (400, 400), interpolation=cv2.INTER_AREA)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            vid_writer.write(frame)
            cv2.imshow('Output-Skeleton', frame)
            # cv2.imwrite("video_output/{:03d}.jpg".format(k), frame)


            key = cv2.waitKey(1)
            if key == 27:
                break

            print("total = {}".format(time.time() - t))

            #vid_writer.write(frame)


    vid_writer.release()
    cap.release()
    cv2.destroyAllWindows()


file_path = '/Users/shreya/Downloads/WLASL-master/start_kit/WLASL_v0.3.json'
with open(file_path) as ipf:
    content = json.load(ipf)

vidNames = os.listdir("/Users/shreya/Downloads/WLASL-master/start_kit/raw_videos")

for ent in content:
    num = 0
    gloss = ent['gloss']
    for inst in ent['instances']:
        id = inst['video_id']
        url = inst['url']
        source = inst['source']
        signer = inst['video_id']
        if(num < 4 and gloss != 'drink' and gloss != 'book'):
            if(url.find('youtube')>=0 and vidNames.__contains__(url[url.find("=")+1 : len(url)] + ".mp4") and source != 'northtexas' and source!='asl5200'):
                cap = cv2.VideoCapture("/Users/shreya/Downloads/WLASL-master/start_kit/raw_videos/" + url[url.find("=")+1 : len(url)] + ".mp4")
                #makeVideo(cap, gloss + ".mp4")
                makeVideo(cap, gloss + "-" + str(signer) + '.avi')
                print(gloss + "-" + str(signer) + '.avi')
                num = num+1
            elif(url.find('youtu.be')>=0 and vidNames.__contains__(url[url.find("youtu.be")+9 : len(url)] + ".mp4")):
                cap = cv2.VideoCapture("/Users/shreya/Downloads/WLASL-master/start_kit/raw_videos/" + url[url.find("youtu.be")+9 : len(url)] + ".mp4")
                #makeVideo(cap, gloss + ".mp4")
                makeVideo(cap, gloss + "-" + str(signer) + '.avi')
                print(gloss + "-" + str(signer) + '.avi')
                num = num+1
            else:
                if(url.find('swf')>=0):
                    num = num
                elif(vidNames.__contains__(id + ".mp4")):
                    num = num + 1
                    cap = cv2.VideoCapture("/Users/shreya/Downloads/WLASL-master/start_kit/raw_videos/" + id + ".mp4")
                    #makeVideo(cap, gloss + ".mp4")
                    makeVideo(cap, gloss + "-" + str(signer) + '.avi')
                    print(gloss + "-" + str(signer) + '.avi')















