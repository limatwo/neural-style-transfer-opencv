import argparse
import time
import cv2
import numpy as np


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", help="neural style transfer model")
ap.add_argument("-i", "--image", help="input image to apply neural style transfer to")
ap.add_argument("-v", "--video", help="input video to apply neural style transfer to")
args = vars(ap.parse_args())

# read the video
cap = cv2.VideoCapture(args["video"])

# load the neural style transfer model from disk
print("[INFO] loading style transfer model...")
net = cv2.dnn.readNetFromTorch(args["model"])

# work on video
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fps = int((cap.get(cv2.CAP_PROP_FPS)))
success,image = cap.read()
count = 1
success = True
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('theScreamTransfer.mp4',fourcc, fps, (1920,1080))

while (count < length):
    success, image = cap.read()
    # cv2.imwrite("frame%d.jpg" % count, image)  # save frame as JPEG file
    if cv2.waitKey(10) == 27:
        break

    print(str(count) + '/' + str(length))
    count += 1
    (h, w) = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    out.write(gray)

    # construct a blob from the image, set the input, and then perform a
    # forward pass of the network
    blob = cv2.dnn.blobFromImage(image, 1.0, (w, h),
                                 (103.939, 116.779, 123.680), swapRB=False, crop=False)
    net.setInput(blob)
    # start = time.time()
    output = net.forward()
    # end = time.time()
    # reshape the output tensor, add back in the mean subtraction, and
    # then swap the channel ordering
    output = output.reshape((3, output.shape[2], output.shape[3]))
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.680
    # output /= 255.0
    output = output.transpose(1, 2, 0)
    output = np.uint8(output)
    # output= cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    out.write(output)



out.release()

# python neural_style_transfer_vedieo_offline.py --model models/instance_norm/the_scream.t7 --video videos/Video\ Of\ Flower\ Blooming.mp4
