import sys
sys.path.append("../piecewisewarp")
import PiecewiseWarpCython as pw
import cv2
import numpy as np
import matplotlib.delaunay as triang

def delaunay(vector):
	tri = triang.delaunay(vector[:, 0], vector[:, 1])[2]
	return tri

def loadMarkup(markupFile):
		f = open(markupFile)
		strings = f.read().split("\n")
		coords = [tuple(string.split(" ")[1:]) for string in strings if len(string) > 0]
		result = [(float(x), float(y)) for (x, y) in coords]
		return np.array(result)

if __name__ == "__main__":
	imgIn = cv2.imread("107_0764.bmp")
	imgIn = cv2.cvtColor(imgIn, cv2.COLOR_BGR2GRAY)
	srcVertices = loadMarkup("107_0764.bmp.mat.dat")
	dstVertices = loadMarkup("107_0779.bmp.mat.dat")
	triangles = delaunay(srcVertices)
	imgOut = pw.warpTriangle(imgIn, srcVertices, dstVertices, triangles, imgIn.shape)
	cv2.imwrite("result.bmp", imgOut)
