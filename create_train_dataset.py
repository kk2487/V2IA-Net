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
		fileUrl = QFileDialog.getOpenFileName(self,"Input Video", "D:/new_daytime/distract/crop/","Mp4 (*.mp4)", options=opt)
	
		return fileUrl[0]

# Real action name
classes = read_classes('classes.txt')

# How many image to save a once
num_frame = 2

# Image size for saving
imgSize = (920,720)

if __name__ == '__main__':

	# create folder
	if(not os.path.exists('./rgbir_dataset')):
		os.makedirs('./rgbir_dataset/'+'A')
		os.makedirs('./rgbir_dataset/'+'B')
		for i in range(len(classes)):
			os.makedirs('./rgbir_dataset/A/'+str(classes[i]))
			os.makedirs('./rgbir_dataset/B/'+str(classes[i]))
	
	save_A_path = './rgbir_dataset/'+'A'
	save_B_path = './rgbir_dataset/'+'B'

	# input video
	qt_env = QApplication(sys.argv)
	process = Qt()
	fileUrl = process.mv_Chooser()
	print(fileUrl)

	# split file name 
	file_name = Path(fileUrl).stem
	mode = file_name.split('_',3)
	#print(mode)

	# night type video 
	if(mode[0] == "night"):
		save_path = save_B_path
		if(mode[1] == "talk" or mode[1] == "text"):
			save_dir = str(mode[1]) + "_" + str(mode[2])
		else:
			save_dir = mode[1]
	# daytime type video
	else:
		save_path = save_A_path
		if(mode[0] == "talk" or mode[0] == "text"):
			save_dir = mode[0] + "_" +  mode[1]
		else:
			save_dir = mode[0]

	path = save_path + "/" + save_dir + "/"
	print("save_path : ", save_path)
	print("save_dir : ", save_dir)
	
	# open video with opencv
	cap = cv2.VideoCapture(fileUrl)
	movie_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

	
	ret, frame = cap.read()
	i = 1

	# save tmp
	series = [] 
	in_rate = 0 
	while(ret):

		if(in_rate%2 == 0):
			draw = frame.copy()

			frame = cv2.resize(frame, imgSize)
			frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
			series.append(frame)

			# key enter for saving, key 0 for ignore
			cv2.putText(draw,"Save : Key Enter",(20,30),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.6,(0,0,255),1)
			cv2.putText(draw,"Pass : Key 0",(20,70),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.6,(0,0,255),1)
			draw = cv2.resize(draw, (int(2*draw.shape[1]/3), int(2*draw.shape[0]/3)))
			cv2.imshow("src", draw)

			if(len(series)%num_frame == 0):
				key = cv2.waitKey(0)
				#print(key)

				all_content=os.listdir(path)
				#print('All content numbers is',len(all_content))

				if(key == 13):
					for n in range(num_frame):
						cv2.imwrite(path + str(len(all_content)+1+n) + ".jpg", series[n])

				series = []

				if(key==27):
					cap.release()
					cv2.destroyAllWindows()
					break
			cv2.waitKey(50)
			
			# interval image	
			for i in range(4):
				ret, frame = cap.read()


		in_rate = in_rate + 1 

		ret, frame = cap.read()

	



		
		