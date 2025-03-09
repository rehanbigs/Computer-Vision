import cv2
import numpy as np
import matplotlib.pyplot as plt

def convolve(image, kernel):
    imgh, imgwidth = image.shape
    kernelh, kernelw = kernel.shape
    
    padh, padw = kernelh // 2, kernelw // 2
    paddedimage = np.pad(image, ((padh, padh), (padw, padw)), mode='constant', constant_values=0)
    
    output = np.zeros_like(image)
    
    for i in range(imgh):
        for j in range(imgwidth):
            region = paddedimage[i:i+kernelh, j:j+kernelw]
            output[i, j] = np.sum(region * kernel)
    
    return output
image = cv2.imread('/home/rehanfarooq/cv/boy.jpg', cv2.IMREAD_GRAYSCALE)
xfilter = np.array([[-1, 1]])
yfilter = np.array([[-1], [1]])



xderivative = convolve(image, xfilter)
yderivative = convolve(image, yfilter)
gradientmagnitude = np.sqrt(xderivative**2 + yderivative**2)

edgeorientation = np.arctan2(yderivative, xderivative)
edgeorientationdegrees = np.degrees(edgeorientation)

roundedangles = np.zeros_like(edgeorientationdegrees)
roundedangles[(edgeorientationdegrees >= -22.5) & (edgeorientationdegrees <= 22.5)] = 0

roundedangles[(edgeorientationdegrees > 22.5) & (edgeorientationdegrees <= 67.5)] = 45
roundedangles[(edgeorientationdegrees > 67.5) | (edgeorientationdegrees <= -67.5)] = 90

roundedangles[(edgeorientationdegrees < -22.5) & (edgeorientationdegrees >= -67.5)] = 135

def nonmaxsuppression(gradientmagnitude, roundedangles):
    imgh, imgwidth = gradientmagnitude.shape
    output = np.zeros_like(gradientmagnitude)

    for i in range(1, imgh-1):
        for j in range(1, imgwidth-1):
            direction = roundedangles[i, j]

            if direction == 0:
                neighbors = [gradientmagnitude[i, j-1], gradientmagnitude[i, j+1]]
            elif direction == 45:
                neighbors = [gradientmagnitude[i-1, j+1], gradientmagnitude[i+1, j-1]]
            elif direction == 90:
                neighbors = [gradientmagnitude[i-1, j], gradientmagnitude[i+1, j]]
            elif direction == 135:
                neighbors = [gradientmagnitude[i-1, j-1], gradientmagnitude[i+1, j+1]]
            
            if gradientmagnitude[i, j] >= max(neighbors):
                output[i, j] = gradientmagnitude[i, j]
            else:
                output[i, j] = 0

    return output

nmsresult = nonmaxsuppression(gradientmagnitude, roundedangles)



def hysteresisthresholding(nmsresult, lowthresh, highthresh):
    imgh, imgwidth = nmsresult.shape
    strongedges = (nmsresult >= highthresh).astype(np.uint8)
    weakedges = ((nmsresult >= lowthresh) & (nmsresult < highthresh)).astype(np.uint8)
    
    outputedges = np.zeros_like(nmsresult, dtype=np.uint8)

    outputedges[strongedges == 1] = 255

    for i in range(1, imgh-1):
        for j in range(1, imgwidth-1):
            if weakedges[i, j] == 1:
                if np.any(strongedges[i-1:i+2, j-1:j+2] == 1):
                    outputedges[i, j] = 255

    return outputedges

gradientmin = nmsresult.min()
gradientmax = nmsresult.max()
lowthresh = 0.1 * gradientmax
highthresh = 0.3 * gradientmax

finaledges = hysteresisthresholding(nmsresult, lowthresh, highthresh)
plt.figure(figsize=(12, 6))
plt.subplot(131), plt.imshow(image, cmap='gray'), plt.title('original image')
plt.subplot(132), plt.imshow(nmsresult, cmap='gray'), plt.title('after non-maximum suppression')
plt.subplot(133), plt.imshow(finaledges, cmap='gray'), plt.title('final edges after hysteresis thresholding)')
plt.show()

