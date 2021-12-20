import cv2
import os
import numpy as np
import pandas as pd
import math
from scipy.spatial import distance

distPlantas = 30
grau = 10
tamFalha = 60
angulação = 90
pLinha = 4
angMenor = [4,12]

class Ponto():
	def __init__(self,X,Y,comp,liga,idx):
		self.X = X
		self.Y = Y
		self.comp = comp
		self.liga = liga
		self.idx = idx

class Distancia():
	def __init__(self,dist,idx):
		self.dist = dist
		self.idx = idx

def get_atr_dist(self):
	return self.dist

def get_atr_Y(self):
	return self.Y

def mudaComponente(comp,newComp):
	global pontos
	for p in pontos:
		if p.comp == comp:
			p.comp = newComp

def visita(a,b):
	global componente
	if a.comp != 0:
		if b.comp != 0:
			mudaComponente(b.comp,a.comp)
		else:
			b.comp = a.comp
	else:
		if b.comp != 0:
			a.comp = b.comp
		else:
			a.comp = componente
			b.comp = componente
			componente += 1
	a.liga.append(b.idx)
	b.liga.append(a.idx)

def pLinhas(rango):

	img = cv2.imread("parcelaT_2/parcela"+str(rango)+".tif")
	fundo = np.zeros((img.shape[0], img.shape[1],3),dtype = "uint8")
	txt = pd.read_csv("parcelaT_2/pontos"+str(rango)+".txt")
	txt = txt.sort_values(by='meioX')
	arquivo = open('Contagem.csv','a')
	
	global pontos
	global componente
	global variaG
	pontos = []
	linhas = []
	Vaux = []
	cont = 0
	componente = 1

	if rango in angMenor:
		variaG = 10
	else:
		variaG = grau

	print(rango)
	print(variaG)

	for i in txt.index:
		meioX = txt["meioX"][i]
		meioY = txt["meioY"][i]
		fundo[meioY][meioX] = (0,0,255)
		ponto = Ponto(meioX,meioY,0,[],cont)
		pontos.append(ponto)
		cont += 1

	print("iniciou")

	for p1 in pontos:
		distancias = []
		a = (p1.X, p1.Y)

		for p2 in pontos:
			b = (p2.X, p2.Y)
			#calculando as distancias entre a e todos os outros pontos
			dist = distance.euclidean(a,b)
			distancia = Distancia(dist,p2.idx)
			distancias.append(distancia)

		distancias.sort(key=get_atr_dist)

		for j in range(1,len(pontos)):
			jIdx = distancias[j].idx
			if distancias[j].dist < distPlantas and pontos[jIdx].idx not in p1.liga:
				if p1.comp != 0 or pontos[jIdx].comp != 0:
					if pontos[jIdx].comp == p1.comp:
						pass
					else:
						#marcar os pontos visitados
						visita(p1,pontos[jIdx])
						break
				else:
					#marcar os pontos visitados
					visita(p1,pontos[jIdx])
					break
			else:
				radianos = math.atan2(p1.Y-pontos[jIdx].Y, p1.X-pontos[jIdx].X)
				graus = math.degrees(radianos)
				if graus >= angulação-variaG and graus <= angulação+variaG and pontos[jIdx].idx not in p1.liga:
					if p1.comp != 0 or pontos[jIdx].comp != 0:
						if pontos[jIdx].comp == p1.comp:
							pass
						else:
							#marcar os pontos visitados
							visita(p1,pontos[jIdx])
							break
					else:
						#marcar os pontos visitados
						visita(p1,pontos[jIdx])
						break

	print("separando linhas")

	#separando linhas uma por uma
	for p in pontos:
		if p.comp in Vaux or p.comp == 0:
			pass
		else:
			Vaux.append(p.comp)
			aux = p.comp
			linha = []
			for p2 in pontos:
				if p2.comp == aux:
					linha.append(p2)
			linhas.append(linha)

	print("desenhando e contando")

	c_linhas = 0
	#desenhando resultado 
	arquivo.write("Rango "+str(rango))
	arquivo.write("\n")	
	for l in range (1,65):
		arquivo.write("linha "+str(l)+";")
	arquivo.write("\n")	

	for i in range (len(linhas)):
		linhas[i].sort(key=get_atr_Y)
		if len(linhas[i]) > pLinha:
			c_linhas += 1
			if c_linhas > 64:
				break
			#print("Linha"+str(c_linhas)+": "+str(len(linhas[i])))
			arquivo.write(str(len(linhas[i]))+";")
			for j in range (len(linhas[i]) - 1):
				a = (linhas[i][j].X, linhas[i][j].Y)
				b = (linhas[i][j+1].X, linhas[i][j+1].Y)
				dist = distance.euclidean(a,b)
				#if dist > tamFalha:
				cv2.line(img, a,b,(0,0,200) , 5)
				cv2.circle(img ,a, 3, (0,0,0), -1)
				cv2.circle(img ,b, 3, (0,0,0), -1)
				if j == 1:
					pass
					cv2.putText(img,str(c_linhas),(a),cv2.FONT_HERSHEY_SIMPLEX,1,(200,0,0),2)
		else:
			pass
	arquivo.write("\n")	
	arquivo.write("\n")	
	cv2.imwrite("parcelaT_2/linhas"+str(rango)+".png",img)
	#cv2.imwrite("linhas"+str(rango)+".png",img)

for i in range(1,30):
	pLinhas(i)

