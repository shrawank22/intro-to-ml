# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# filenames: a list of strings containing filenames of images

# OUTPUT CONVENTION
# The method must return a list of strings. Make sure that the length of the list is the same as
# the number of filenames that were given. The evaluation code may give unexpected results if
# this convention is not followed.

import pickle

import cv2
import numpy as np


### Background remover code is taken from freedomvc article 'https://www.freedomvc.com/index.php/2022/01/17/basic-background-remover-with-opencv/'
def bgremover(myimage):
    # BG Remover 3
    myimage_hsv = cv2.cvtColor(myimage, cv2.COLOR_BGR2HSV)
     
    #Take S and remove any value that is less than half
    s = myimage_hsv[:,:,1]
    s = np.where(s < 127, 0, 1) # Any value below 127 will be excluded
 
    # We increase the brightness of the image and then mod by 255
    v = (myimage_hsv[:,:,2] + 127) % 255
    v = np.where(v > 127, 1, 0)  # Any value above 127 will be part of our mask
 
    # Combine our two masks based on S and V into a single "Foreground"
    foreground = np.where(s+v > 0, 1, 0).astype(np.uint8)  #Casting back into 8bit integer
    foreground=cv2.bitwise_and(myimage,myimage,mask=foreground) # Apply our foreground map to original image
    finalimage = 0+foreground # Combine foreground and background
    return finalimage

def decaptcha(filenames):
	kernel = np.ones((5, 5), np.uint8) 
	classifier = pickle.load(open('model.pkl', 'rb'))
	labels = []
	for file in filenames:
		img = cv2.imread(file)
		res_img = bgremover(img)
		img_erosion = cv2.erode(res_img, kernel, iterations=2)
		img_dilation = cv2.dilate(img_erosion, kernel, iterations=2)
		img = cv2.cvtColor(img_dilation,cv2.COLOR_BGR2GRAY)
		x = 0
		tmp_labels = ''
		label_dic = {0: 'ALPHA', 1: 'BETA', 2: 'CHI', 3: 'DELTA', 4: 'EPSILON', 5: 'ETA', 6: 'GAMMA', 7: 'IOTA', 8: 'KAPPA', 9: 'LAMDA', 10: 'MU', 11: 'NU', 12: 'OMEGA', 13: 'OMICRON', 14: 'PHI', 15: 'PI', 16: 'PSI', 17: 'RHO', 18: 'SIGMA', 19: 'TAU', 20: 'THETA', 21: 'UPSILON', 22: 'XI', 23: 'ZETA'}
		c = 0
		for iw in range(3):
				cimg = img[0:150, x:x+166]
				cimg = np.array(cimg)
				cimg = cimg.reshape(-1,150,166,1)
				pred = classifier.predict(cimg)
				tmp_labels += label_dic[np.argmax(pred)]
				if (c < 3):
					tmp_labels += ','
				c += 1
				x += 166
		labels.append(tmp_labels)
	return labels
# The use of a model file is just for sake of illustration
#with open( "model.txt", "r" ) as file:
#labels = file.read().splitlines()
#maxx = max index from pred
