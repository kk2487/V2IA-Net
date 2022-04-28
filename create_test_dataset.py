import os
import sys
import cv2 
import numpy as np
from tqdm import tqdm
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import PyQt5.QtGui as QtGui
from PyQt5.QtCore import *
from pathlib import Path  

# It can read action name from .txt file
def read_classes(file_path):

	fp = open(file_path, "r")
	classes = fp.readline()
	classes = classes. split(",")
	fp.close()

	return classes

# Choose file by GUI
class Qt(QWidget):
	def mv_Chooser(self):    
		opt = QFileDialog.Options()
		opt |= QFileDialog.DontUseNativeDialog
		fileUrl = QFileDialog.getOpenFileName(self,"Input Video", "D:/new_night/distract/crop/","Mp4 (*.mp4)", options=opt)
	
		return fileUrl[0]

# Real action name
classes = read_classes('classes.txt')

# Image size for saving
imgSize = (920,720)

if __name__ == '__main__':

	# input video
	qt_env = QApplication(sys.argv)
	process = Qt()
	fileUrl = process.mv_Chooser()
	print(fileUrl)

	# create folder
	if(not os.path.exists('./test_dataset')):
		os.makedirs('./test_dataset/')

	file_name = Path(fileUrl).stem
	mode = file_name.split('_',3)
	# print(mode)
	if(mode[0] == "night"):		
		save_dir = mode[0]+'_'+mode[1]+'_'+mode[2]
	else:
		save_dir = mode[0]+'_'+mode[1]

	path = "./test_dataset/"+save_dir+"/"
	os.makedirs(path)

	for i in range(len(classes)):
			os.makedirs(path+str(classes[i]))

	
	print("path : ", path)
	
	# open video with opencv
	cap = cv2.VideoCapture(fileUrl)
	movie_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

	
	ret, frame = cap.read()
	i = 1
	series = []

	in_rate = 0 
	while(ret):

		if(in_rate%2 == 0):
			draw = frame.copy()

			frame = cv2.resize(frame, imgSize)
			frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

			# choose action type, 0~5 is action type, 6 is ignore 
			for c in range(len(classes)):
				cv2.putText(draw,str(c)+":"+classes[c],(20,20+c*40),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.2,(255),1)
			cv2.putText(draw,str(len(classes))+":"+"X",(20,20+len(classes)*40),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.2,(255),1)

			draw = cv2.resize(draw, (int(2*draw.shape[1]/3), int(2*draw.shape[0]/3)))
			cv2.imshow("src", draw)

			
			key = cv2.waitKey(0)
			#print(key)

			
			#print('All content numbers is',len(all_content))
			#print(key)
			if(key < 54 and key>47):
				save_img_path = path+str(classes[key-48])+"/"
				all_content=os.listdir(save_img_path)

			if(key < 54 and key>47):
				
				cv2.imwrite(save_img_path + str(len(all_content)+1) + ".jpg", frame)

			if(key==27):
				cap.release()
				cv2.destroyAllWindows()
				break
			
			
			for i in range(1):
				ret, frame = cap.read()


		in_rate = in_rate + 1 

		ret, frame = cap.read()

	



		
		