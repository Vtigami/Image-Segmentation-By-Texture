import glob
import cv2 as cv
import numpy as np

def showImg(img):
	cv.imshow("Imagem", img)
	cv.waitKey(0) 
	cv.destroyAllWindows()

def getFilters(size):
	array=[]

	for theta in np.arange(0, np.radians(180), np.radians(45)):
		kernel = cv.getGaborKernel((size,size),4.0,theta,8,0,0)
		array.append(kernel)
	kernel = cv.getGaborKernel((size,size),4.0,0,8,1,0,ktype=cv.CV_32F) 
	array.append(kernel)
		

	return array


def filterImg(img, kernel,stride):
	height,width = img.shape
	kernel_size, _= kernel.shape[:2]

	output_height = (height - kernel_size) // stride + 1
	output_width = (width - kernel_size) // stride + 1
	output = np.zeros((output_height, output_width), dtype=img.dtype)

	for h in range(0, height - kernel_size +1, stride):
		for w in range(0, width - kernel_size+1, stride):
			window = img[h:h+kernel_size,w:w+kernel_size]
			output[h // stride, w // stride] = np.sum(window + kernel)/(kernel_size*kernel_size)
	return output

def addBorder(img,size):
	return  cv.copyMakeBorder(img,top=size, bottom=size, left=size, right=size, borderType=cv.BORDER_CONSTANT, value=[0, 0, 0]
)

def getFeatureCoords(img,kernel_size,stride):
	height,width = img.shape

	output_height = (height - kernel_size) // stride + 1
	output_width = (width - kernel_size) // stride + 1
	output = np.zeros((output_height*output_width,2))

	for i in range(0, output_height - kernel_size , stride):
		for j in range(0, output_width - kernel_size, stride):
			output[i*output_width+j,0] = int(i/4)
			output[i*output_width+j,1] = int(j/4)
	return output

def getRandomColorsArray(n):
	colors=[]

	for i in range(0,n):
		colors.append(np.random.choice(range(256),size=3))
	return colors

#===========================Main===========================

kernels_sizes=[9,18,27]
num_clusters=7
stride=1
useSize = False

for infile in glob.glob("imagens/*.jpg"):

	imgOr=cv.imread(infile)

	_,fileName=infile.split("/")
	print("Segmentando imagem "+fileName+"...")

	imgResults=[]

	for kernel_size in kernels_sizes:

		img = imgOr

		#Gray Scale e Sharp Filter
		img=cv.cvtColor(imgOr,cv.COLOR_BGR2GRAY)
		kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
		img=cv.filter2D(img, -1, kernel)

		#obtem os filtros de gabor
		kernels = getFilters(kernel_size)

		kernel_size,_=kernels[0].shape

		#adiciona bordas na imagem
		img=addBorder(img,int(kernel_size/2))

		#caso for utilizar as coordenadas da imagem como features
		if useSize==True:
			features=getFeatureCoords(img,kernel_size,stride) 
		else:
			features=[]

		#aplica os filtros
		for kernel in kernels:
			afterImg = filterImg(img, kernel,stride)

			#ajusta o formato da imagem para o kmeans
			afterImg = afterImg.flatten()
			afterImg=afterImg.reshape((len(afterImg),1))
			if len(features)==0:
				features=afterImg
			else:
				features=np.hstack((features,afterImg))

		

		criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1.0)

		features = np.float32(features)

		#realiza o kmeans para realizar o agrupamento
		_,labels,centers= cv.kmeans (features,num_clusters,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)


		#obtem o cluster de cada pixel e cores aleatórias para cada cluster
		h,w,_=imgOr.shape
		labels = labels.reshape(h,w);
		colors=getRandomColorsArray(num_clusters)
		

		#preenche a imagem com os clusters
		result=np.zeros((h, w,3), np.uint8)
		for i in range(0,h):
			for j in range(0,w):
				result[i,j] = colors[labels[i][j]]

		result = 		cv.resize(result,None,fx=stride,fy=stride,interpolation = cv.INTER_CUBIC)
		imgResults.append(result)

	print("    -Imagens segmentadas:\n")
	cv.imshow("Imagem original",imgOr)
	for i in range(0,3):
		cv.imshow("Kernel size: "+ str(kernels_sizes[i]),imgResults[i])
	cv.waitKey(0)
	cv.destroyAllWindows()



