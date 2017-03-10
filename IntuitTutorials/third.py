import sys, cv2 as cv

cap = cv.VideoCapture(0)
cascade = cv.CascadeClassifier("lbpcascades/lbpcascade_frontalface.xml")

while True:
	ok, img = cap.read()

	if not ok:
		break
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	sf = min(640./img.shape[1], 480./img.shape[0])
	gray = cv.resize(gray, (0,0), None, sf, sf)
	rects = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors = 4, minSize=(40,40), flags=2)
	
	gray = cv.GaussianBlur(gray, (3,3), 1.1)

	edges = cv.Canny(gray, 5, 50)

	out = cv.cvtColor(gray,cv.COLOR_GRAY2BGR)

	for x,y,w,h in rects:
		cv.rectangle(out, (x,y), (x+w, y+h), (0,0,255), 2)

	cv.imshow("Edges+face", out)

	if cv.waitKey(30) != 255:
		break