import tensorflow.compat.v1 as tf
import cv2
import numpy as np
from PIL import Image
import torch

IOU_T = 0.25
THRES_SCORE = 0.5
square_size = 960

boxes3 = []
scores3 = []

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


def detect(img,H,W,path):

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

	desenha(img,path)

def desenha(image,img_path):

	f = open(img_path + ".xml", "w")

	f.write("<annotation>\n")
	f.write("	<folder>Train</folder>\n")
	f.write("	<filename>"+img_path+">JPG</filename>\n")
	f.write("	<path>C:/Users/Enzo/Desktop/labelImg-master/TCC/Train/"+img_path+".JPG</path>\n")
	f.write("	<source>\n")
	f.write("		<database>Unknown</database>\n")
	f.write("	</source>\n")
	f.write("	<size>\n")
	f.write("		<width>"+str(image.shape[1])+"</width>\n")
	f.write("		<height>"+str(image.shape[0])+"</height>\n")
	f.write("		<depth>"+str(image.shape[2])+"</depth>\n")
	f.write("	</size>\n")
	f.write(" <segmented>0</segmented>\n")

	for b in boxes3:
		b = b.astype(int)
		f.write(" <object>\n")
		f.write(" 	<name>Planta</name>\n")
		f.write(" 	<pose>Unspecified</pose>\n")
		f.write(" 	<truncated>0</truncated>\n")
		f.write(" 	<difficult>0</difficult>\n")
		f.write(" 	<bndbox>\n")
		f.write(" 		<xmin>"+str(b[0])+"</xmin>\n")
		f.write(" 		<ymin>"+str(b[1])+"</ymin>\n")
		f.write(" 		<xmax>"+str(b[2])+"</xmax>\n")
		f.write(" 		<ymax>"+str(b[3])+"</ymax>\n")
		f.write(" 	</bndbox>\n")
		f.write("</object>\n")

	f.write("</annotation>")
	f.close()


#LOAD MODEL
PATH_TO_CKPT = "../plantas2.pb"
graph_def = tf.Graph().as_graph_def()
graph_def.ParseFromString(open(PATH_TO_CKPT, 'rb').read())
frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")

path = "parcela28_6"
#lendo a imagem
img = cv2.imread(path+".jpg")
if square_size > img.shape[0]:
	square_size = img.shape[0]

img = convertImgTensor(img)

detect(img,0,0,path)





	