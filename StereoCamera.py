import cv2
import requests
from io import BytesIO
import numpy as np
import cv2
import matplotlib.pyplot as plt
from fontTools.misc.psOperators import ps_real


def sobel(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply the Sobel filter
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Convert the result back to uint8
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(sobely)

    # Combine the horizontal and vertical edges
    sobel_combined = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

    return sobel_combined

def disparityArray(imageL, imageR, kernelSize):
    # Ensure the data is in the format of a numpy array
    imageL = np.array(imageL)
    imageR = np.array(imageR)
    print(imageL.shape)
    print(imageR.shape)
    assert(imageL.shape[0] == imageR.shape[0] and imageL.shape[1] == imageR.shape[1])

    # Calculate the H x W of the resulting image
    disparityArrayHeight = 1 + (imageL.shape[0]-kernelSize)    # Stride = 1; Padding = 0
    disparityArrayLength = int((1 + (imageL.shape[1] - kernelSize)) * 0.90)   # Stride = 1; Padding = 0

    disparityArray = np.zeros((disparityArrayHeight, disparityArrayLength))
    for row in range(disparityArrayHeight):
        print(row, " out of ", disparityArrayHeight)
        disparityArray[row] = disparityRow(imageL[row:row+kernelSize], imageR[row:row+kernelSize], kernelSize,
                                           disparityArrayLength)
    return disparityArray

def disparityRow(imageLRow, imageRRow, kernelSize, disparityRowLength):
    disparityRow = np.zeros((1, disparityRowLength))
    for col in range(disparityRowLength):
        peakCorrelation = 0
        peakCorrelationDist = 0
        imageLKernel = imageLRow[:, col:col + kernelSize]

        # Ignore an area with low information density
        peakUnique = np.max(imageLKernel - np.average(imageLKernel))
        if peakUnique < 100 * np.max(imageLRow):
            peakCorrelationDist = 0
            peakCorrelation = kernelSize ** 2 * np.max(imageLRow)

        n = 2
        maxImageWidthToCorrelatePercent = 0.15
        for dist in range(int((disparityRowLength - col)/n)):
            if dist > maxImageWidthToCorrelatePercent * disparityRowLength:
                break
            dist = dist*n
            correlation = crossCorrelate(imageLKernel, imageRRow[:, dist+col:dist+col+kernelSize])
            correlation = correlation * (imageLRow.shape[1] - dist ** 2) / (imageLRow.shape[1])
            if correlation > peakCorrelation:
                peakCorrelation = correlation
                peakCorrelationDist = dist

        disparityRow[0, col] = peakCorrelationDist

        if False:
            f, arr = plt.subplots(4, 1)
            imageLRowMask = np.array(imageLRow)
            imageLRowMask[:, col] = np.max(imageLRow)
            imageLRowMask[:, col + kernelSize] = np.max(imageLRow)

            imageRRowMask = np.array(imageRRow)
            imageRRowMask[:, peakCorrelationDist + col] = np.max(imageRRow)
            imageRRowMask[:, peakCorrelationDist + col + kernelSize] = np.max(imageRRow)

            arr[0].imshow(imageLRowMask)
            arr[1].imshow(imageRRowMask)
            arr[2].imshow(imageLRow[:, col:col + kernelSize])
            arr[3].imshow(imageRRow[:, peakCorrelationDist + col:peakCorrelationDist + col + kernelSize])
            plt.show()


    return disparityRow

def crossCorrelate(image, kernel, maxValue=1):
    assert(image.shape[0] == kernel.shape[0] and image.shape[1] == kernel.shape[1])
    score = 0
    if len(image.shape) == 2:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                score = score + int(image[i, j]) * int(kernel[i, j])
    if len(image.shape) == 3:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(image.shape[2]):
                    score = score + int(image[i, j, k]) * int(kernel[i, j, k])
    return score / maxValue

def preprocessImage(image, size):
    image = cv2.medianBlur(image, 5)
    #
    image = sobel(image)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
    return np.array(image)

ipAddress1 = '192.168.0.62'
ipAddress2 = '192.168.0.61'

streamUrl1 = 'http://' + ipAddress1 + ':81/stream'
streamUrl2 = 'http://' + ipAddress2 + ':81/stream'

while True:
    # OpenCV video capture
    capture1 = cv2.VideoCapture(streamUrl1)
    print("Passed 1")
    capture2 = cv2.VideoCapture(streamUrl2)
    print("Passed 2")

    while True:
        ret1, frame1 = capture1.read()
        ret2, frame2 = capture2.read()

        if not ret1 and ret2:
            print("Breaking")
            # If frame reading fails, break the loop
            break

        frame1 = preprocessImage(frame1, 200)
        frame2 = preprocessImage(frame2, 200)


        offset = 5
        if offset > 0:
            frame2 = frame2[:-offset, :]
            frame1 = frame1[offset:, :]
        elif offset < 0:
            offset = -offset
            frame1 = frame1[:-offset, :]
            frame2 = frame2[offset:, :]

        # Display the frame
        frame = np.concatenate((frame1, frame2), axis=1)
        cv2.imshow('Video Stream 1', frame)
        cv2.waitKey()

        # Create the disparity array
        disparityArr = (disparityArray(frame1, frame2, 20))
        plt.imshow(disparityArr)
        plt.show()



        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close the window
    capture1.release()
    capture2.release()
    cv2.destroyAllWindows()

