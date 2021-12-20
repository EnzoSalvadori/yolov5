import tensorflow.compat.v1 as tf
import cv2
import numpy as np
from PIL import Image
import torch

IOU_T = 0.25
THRES_SCORE = 0.5
square_size = 960

def wrap_frozen_graph(gd, inputs, outputs):
				x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped import
				return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
							   tf.nest.map_structure(x.graph.as_graph_element, outputs))

def convertImgTensor(img):
	img = cv2.resize(img, (960 , 960))
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = torch.from_numpy(img).to()
	img = img.float()  # uint8 to fp16/32
	img /= 255.0  # 0 - 255 to 0.0 - 1.0
	if len(img.shape) == 3:
		img = img[None]  # expand for batch dim
	return img

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
			crop_img = convertImgTensor(crop_img)
			detect(crop_img,cropH1,cropW1,boxes3,scores3)

			if (cropH2+(square_size/2)) < Y:
				crop_img = img[int(cropH1+(square_size/2)):int(cropH2+(square_size/2)), cropW1:cropW2]
				crop_img = convertImgTensor(crop_img)
				detect(crop_img,int(cropH1+(square_size/2)),cropW1,boxes3,scores3)

			if (cropW2+(square_size/2)) < X:
				crop_img = img[cropH1:cropH2, int(cropW1+(square_size/2)):int(cropW2+(square_size/2))]
				crop_img = convertImgTensor(crop_img)
				detect(crop_img,cropH1,int(cropW1+(square_size/2)),boxes3,scores3)

			print(str(((i)*(contw)+(j+1))*100/(conth*contw)) + "%")
		#print(str((i+1)*(contw)*100/(conth*contw)) + "%")

	selected_indices = tf.image.non_max_suppression(
		boxes3, scores3, (len(boxes3)), iou_threshold=IOU_T
	)
	Ml = 0
	Ma = 0
	for i in selected_indices:
		largura = boxes3[i][2] - boxes3[i][0]
		altura = boxes3[i][3] - boxes3[i][1]
		if altura < 200 and largura < 200:
			boxes4.append(boxes3[i])
			scores4.append(scores3[i])
	desenha(img,boxes4,scores4)    


def detect(img,H,W,boxes3,scores3):

	pred = frozen_func(x=tf.constant(img)).numpy()

	pred[..., 0] *= img.shape[1]  # x
	pred[..., 1] *= img.shape[2]  # y
	pred[..., 2] *= img.shape[1]  # w
	pred[..., 3] *= img.shape[2]  # h

	boxes = np.squeeze(pred[..., :4])
	scores  = np.squeeze(pred[..., 4:5]) 

	prct = square_size/960

	for i in boxes:
		cx = int(prct*i[0])
		cy = int(prct*i[1]) 
		w = int(prct*i[2]/2)
		h = int(prct*i[3]/2)
		x1 = cx - w
		y1 = cy - h
		x2 = cx + w
		y2 = cy + h
		i[0] = x1 
		i[1] = y1 
		i[2] = x2 
		i[3] = y2 

	selected_indices = tf.image.non_max_suppression(
			boxes, scores, 500, iou_threshold=IOU_T)

	for i in selected_indices:
		if scores[i] > THRES_SCORE:
			boxes[i][0] = boxes[i][0] + W
			boxes[i][2] = boxes[i][2] + W
			boxes[i][1] = boxes[i][1] + H
			boxes[i][3] = boxes[i][3] + H
			boxes3.append(boxes[i])
			scores3.append(scores[i])

def desenha(image,boxes,scores):
	draw = image.copy()
	draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
	cont = 0
	# visualize detections
	for i in range(len(boxes)):
	  # scores are sorted so we can break
		if scores[i] > THRES_SCORE:
			cont +=1
			#cv2.rectangle(draw,(int(boxes[i][0]),int(boxes[i][1])),(int(boxes[i][2]),int(boxes[i][3])),(0,0,255),2)
			meioX = int((boxes[i][2] - boxes[i][0])/2 + boxes[i][0])
			meioY = int((boxes[i][3] - boxes[i][1])/2 + boxes[i][1])
			#cv2.circle(draw,(meioX,meioY), 3, (255,0,0), -1)
			#cv2.circle(image,(meioX,meioY), 3, (0,0,255), -1)
			cv2.rectangle(image,(int(boxes[i][0]),int(boxes[i][1])),(int(boxes[i][2]),int(boxes[i][3])),(0,0,255),2)
	#plt.figure(figsize=(10, 10))
	#plt.axis('off')
	#plt.imshow(draw)
	#plt.show()
	print(cont)
	cv2.imwrite("DJI_0655_2.JPG",image)

#LOAD MODEL
PATH_TO_CKPT = "plantas_2019.pb"
graph_def = tf.Graph().as_graph_def()
graph_def.ParseFromString(open(PATH_TO_CKPT, 'rb').read())
frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")

img = cv2.imread("DJI_0655.JPG")
if square_size > img.shape[0]:
	square_size = img.shape[0]
divh, divw = calculaHW(img.shape[0],img.shape[1])

cropImagem(img,divh,divw)
