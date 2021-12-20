import tensorflow.compat.v1 as tf
import cv2
import numpy as np
from PIL import Image
import torch

square_size = 912

def calculaHW(h,w):
	fh = True
	fw = True
	conth = 1
	contw = 1
	while fh == True:
		if(h/conth <= square_size):
			fh = False  
		else:
			conth += 1
	while fw == True:
		if(w/contw <= square_size):
			fw = False  
		else:
			contw += 1
	return(conth,contw)

def cropImagem(img,conth,contw):
	boxes3 = []
	scores3 = []
	boxes4 = []
	scores4 = []
	cont = 0
	Y = img.shape[0]
	X = img.shape[1]
	for i in range(0,conth):
		if (square_size * (i+1) < Y):
			cropH1 = i*square_size
			cropH2 = (i+1)*square_size
		elif (conth == 1):
			cropH1 = 0
			cropH2 = Y
		else:
			ajusteH = (i+1)*square_size - Y
			cropH1 = i*square_size - ajusteH
			cropH2 = (i+1)*square_size - ajusteH

		for j in range(0,contw):
			cont += 1
			if (square_size * (j+1) < X):
				cropW1 = j*square_size
				cropW2 = (j+1)*square_size
			elif (contw == 1):
				cropW1 = 0
				cropW2 = X
			else:
				ajusteW = (j+1)*square_size - X
				cropW1 = j*square_size - ajusteW
				cropW2 = (j+1)*square_size - ajusteW

			crop_img = img[cropH1:cropH2, cropW1:cropW2]
			cv2.imwrite("out/"+path+"_"+str(cont)+".jpg", crop_img)

#lendo a imagem
path = "parcela1"
ext = ".tif"
img = cv2.imread(path+ext)
if square_size > img.shape[0]:
	square_size = img.shape[0]
divh, divw = calculaHW(img.shape[0],img.shape[1])


cropImagem(img,divh,divw)
