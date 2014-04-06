
import numpy as np
import cv2

cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b

def warpTriangle(imgIn, srcVertices, trgVertices, triangles, imageSize):
    imgOut = np.zeros(imageSize)
    posuvX = []
    posuvY = []
    r = []
    cdef double c12, c20, c01
    cdef int x, y, w, h
    cdef int imageWidth = imageSize[1]
    cdef int imageHeight = imageSize[0]
    cdef int minBoundX, minBoundY, maxBoundX, maxBoundY
    cdef double lambdat1, lambdat2, lambdat3
    cdef double lambdaArray1, lambdaArray2, lambdaArray3
    cdef double g12X, g12Y, g20X, g20Y, g01X, g01Y
    cdef double p1X, p1Y, p2X, p2Y, p3X, p3Y
    cdef double f12, f20, f01
    cdef double s1X, s1Y, s2X, s2Y, s3X, s3Y
    cdef int length = 0
    for tr in triangles:
        (s1X, s1Y), (s2X, s2Y), (s3X, s3Y) = srcVertices[tr]
        (p1X, p1Y), (p2X, p2Y), (p3X, p3Y) = trgVertices[tr]
        trianglePoints = np.array([[(p1X, p1Y)], [(p2X, p2Y)], [(p3X, p3Y)]],
                                  dtype=np.float32)
        x, y, w, h = cv2.boundingRect(trianglePoints)
        minBoundX = int_max(x, 0)
        minBoundY = int_max(y, 0)
        maxBoundX = int_min(x + w, imageWidth - 1)
        maxBoundY = int_min(y + h, imageHeight - 1)

        f12 = (p2Y - p3Y) * p1X + (p3X - p2X) * p1Y + p2X * p3Y - p3X * p2Y
        f20 = (p3Y - p1Y) * p2X + (p1X - p3X) * p2Y + p3X * p1Y - p1X * p3Y
        f01 = (p1Y - p2Y) * p3X + (p2X - p1X) * p3Y + p1X * p2Y - p2X * p1Y

        g12X = (p2Y - p3Y) / f12
        g12Y = (p3X - p2X) / f12
        g20X = (p3Y - p1Y) / f20
        g20Y = (p1X - p3X) / f20
        g01X = (p1Y - p2Y) / f01
        g01Y = (p2X - p1X) / f01

        c12 = (p2X * p3Y - p3X * p2Y) / f12
        c20 = (p3X * p1Y - p1X * p3Y) / f20
        c01 = (p1X * p2Y - p2X * p1Y) / f01

        lambdat1 = g12X * minBoundX + g12Y * minBoundY + c12
        lambdat2 = g20X * minBoundX + g20Y * minBoundY + c20
        lambdat3 = g01X * minBoundX + g01Y * minBoundY + c01
        for j in xrange(minBoundY, maxBoundY + 1):
            lambdaArray1 = lambdat1
            lambdaArray2 = lambdat2
            lambdaArray3 = lambdat3
            for i in xrange(minBoundX, maxBoundX + 1):
                if (0 <= lambdaArray1 <= 1 and 0 <= lambdaArray2 <= 1 and
                    0 <= lambdaArray3 <= 1):
                    posuvX.append(lambdaArray1 * s1X + lambdaArray2 *
                                  s2X + lambdaArray3 * s3X)
                    posuvY.append(lambdaArray1 * s1Y + lambdaArray2 *
                                  s2Y + lambdaArray3 * s3Y)
                    r.append((j, i))
                    length += 1
                lambdaArray1 += g12X
                lambdaArray2 += g20X
                lambdaArray3 += g01X
            lambdat1 += g12Y
            lambdat2 += g20Y
            lambdat3 += g01Y
    if length != 0:
        imgTr = np.empty(len(posuvX), dtype='double')
        cv2.remap(imgIn.astype('double'), np.array(posuvX).astype('float32'),
                  np.array(posuvY).astype('float32'), cv2.INTER_LINEAR, imgTr,
                  borderMode=cv2.BORDER_REPLICATE)
        #imgTr /= 256
    for i in xrange(length):
        imgOut[r[i]] = imgTr[i]
    return imgOut
