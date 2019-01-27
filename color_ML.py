import cv2
import os 

import webcolors as wb

import pandas as pd 
import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
from gensim.models import doc2vec

dir_path = os.path.dirname(os.path.realpath(__file__))

# ===========================================================================================

def takePics():
	# create my VideoCapture object
	cam = cv2.VideoCapture(0)

	# create the window
	cv2.namedWindow('Ben Webcam')

	# keeps track of how many images have been taken
	imgcounter = 0

	# keeps the paths of all the images taken
	images = []

	while True:
		# retvalue is a boolean (if frame read properly, it returns true)
		# frame is the current frame 
		retvalue, frame = cam.read()

		# Display the window
		cv2.imshow('Ben Webcam', frame)

		# if frame not read properly -> break out of the loop
		if not retvalue:
			break

		# wait for a key event
		keyPressed = cv2.waitKey(1)

		# SPACE key pressed
		if keyPressed%256 == 32:
			imgname = "picture_{}.jpg".format(imgcounter)
			cv2.imwrite(imgname, frame)
			print("{} saved!".format(imgname))
			images.append(str(dir_path) + '/' + imgname)
			imgcounter += 1

		# ESC key pressed
		elif keyPressed%256 == 27:
			print("Closing webcam")
			break

	cam.release()
	cv2.destroyAllWindows()

# ===========================================================================================

def colorML():
	#read image
	img = cv2.imread('picture_0.jpg')

	#convert from BGR to RGB
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	#get rgb values from image to 1D array
	r, g, b = cv2.split(img)
	r = r.flatten()
	g = g.flatten()
	b = b.flatten()

	# colors = []
	# for i in range(0, len(r)):
	# 	colors.append(rgb_to_name((r[i], g[i], b[i])))

	r_avg = sum(r)/len(r)
	g_avg = sum(g)/len(g)
	b_avg = sum(b)/len(b)

# ===========================================================================================

	# Load data from path 
	df = pd.read_csv('/Users/benhwang/Desktop/Color dataset - Sheet1.csv', low_memory=False)

	subset = df[['Red (8 bit)', 'Green (8 bit)', 'Blue (8 bit)']]
	tuples = [tuple(x) for x in subset.values]

	X = np.array(tuples)

# ===========================================================================================

	# labels (self created)
	labels = []

	red = 1
	orange = 2
	yellow = 3
	green = 4
	blue = 5
	gray = 6
	white = 7
	black = 8
	brown = 9
	pink = 10
	purple = 11
	beige = 12
	
	for i in range(0, len(df)):
		label = 0

		if (df.iloc[i, 5] == 'red'):
			label = red

		elif (df.iloc[i, 5] == 'orange'):
			label = orange

		elif (df.iloc[i, 5] == 'yellow'):
			label = yellow

		elif (df.iloc[i, 5] == 'green'):
			label = green

		elif (df.iloc[i, 5] == 'blue'):
			label = blue

		elif (df.iloc[i, 5] == 'gray'):
			label = gray

		elif (df.iloc[i, 5] == 'white'):
			label = white

		elif (df.iloc[i, 5] == 'black'):
			label = black

		elif (df.iloc[i, 5] == 'brown'):
			label = brown

		elif (df.iloc[i, 5] == 'pink'):
			label = pink

		elif (df.iloc[i, 5] == 'purple'):
			label = purple

		elif (df.iloc[i, 5] == 'beige'):
			label = beige

		labels.append(label)

	y = np.array(labels)

# ===========================================================================================

	# test_size determines percentage of data to be used for testing
	X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)

	clf = neighbors.KNeighborsClassifier()
	clf.fit(X_train, y_train)

	accuracy = clf.score(X_test, y_test)
	print("Accuracy: " + str(accuracy)) # around 90-91%

# ===========================================================================================
	
	tuple_rgb = (r_avg, g_avg, b_avg)
	example = np.array(tuple_rgb)
	example = example.reshape(1, -1)
	prediction = clf.predict(example)

	label_str = ""

	red = 1
	orange = 2
	yellow = 3
	green = 4
	blue = 5
	gray = 6
	white = 7
	black = 8
	brown = 9
	pink = 10
	purple = 11
	beige = 12

	if prediction[0] == 1:
		label_str = 'Red'

	elif prediction[0] == 2:
		label_str = "Orange"

	elif prediction[0] == 3:
		label_str = "Yellow"

	elif prediction[0] == 4:
		label_str = "Green"

	elif prediction[0] == 5:
		label_str = "Blue"

	elif prediction[0] == 6:
		label_str = "Gray"

	elif prediction[0] == 7:
		label_str = "White"

	elif prediction[0] == 8:
		label_str = "Black"

	elif prediction[0] == 9:
		label_str = "Brown"

	elif prediction[0] == 10:
		label_str = "Pink"

	elif prediction[0] == 11:
		label_str = "Purple"

	elif prediction[0] == 12:
		label_str = "Beige"

	print("Your shirt is likely " + label_str + ".")

# ===========================================================================================

takePics()
colorML()

# ===========================================================================================
