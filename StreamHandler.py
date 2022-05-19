import cv2 #импортирование библиотеки
import time
import numpy as np

class StreamHandler:
	stream = None
	width = 0
	height = 0
	fps = 0
	controlFrames = []
	def __init__(self, stream):
		self.stream = stream
		self.width = round(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.height = round(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
		self.fps = stream.get(cv2.CAP_PROP_FPS)

	def getSize(self):
		return [self.width, self.height]

	def getState(self):
		return self.stream.isOpened()

	def contourProcess(self, contours):
		sumLeftZone = 0
		sumRightZone = 0
		for contour in contours:
			for dot_packed in contour:
				#print(dot_packed[0][0])
				dot = []
				dot.append(int(dot_packed[0][0]))
				dot.append(int(dot_packed[0][1]))
				#print(dot)
				if(dot[0] >= int(self.width*0.3) and dot[0] <= int(self.width*0.5)):
					sumLeftZone += 1
				elif(dot[0] >= int(self.width*0.75) and dot[0] <= int(self.width*0.95)):
					sumRightZone += 1


		#print("SumLeftZone: ", sumLeftZone)
		#print("SumRightZone: ", sumRightZone)
		#time.sleep(5)
		return [sumLeftZone, sumRightZone]

	def frameProccess(self, init_frame):
		init_frame = cv2.rectangle(init_frame,(0, self.height//2),(self.width, round((self.height//4)*2.8)),(0,0,255),2)
		#print(frame) #вывод массива в консоль
		proc_frame = init_frame[self.height//2:round((self.height//4)*2.8), 0:self.width]

		#cv2.imshow("proc_frame1",proc_frame)
		proc_frame = cv2.medianBlur(proc_frame,5)

		#cv2.imshow("proc_frame2",proc_frame)
		ret, proc_frame = cv2.threshold(proc_frame, 210, 255, 0)
		#cv2.imshow("proc_frame3",proc_frame)
		proc_frame = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
		#cv2.imshow("proc_frame4",proc_frame)
		proc_frame = cv2.GaussianBlur(proc_frame, (7, 7), 1.5) # Параметры позволяют регулировать шумность
		#cv2.imshow("proc_frame5",proc_frame)
		proc_frame = cv2.Canny(proc_frame, 1, 50)

		

		init_frame = cv2.rectangle(init_frame,(int(self.width*0.3), self.height//2),(int(self.width*0.5), round((self.height//4)*2.8)),(0,255,255),2)
		init_frame = cv2.rectangle(init_frame,(int(self.width*0.75), self.height//2),(int(self.width*0.95), round((self.height//4)*2.8)),(0,255,255),2)
		cv2.line(proc_frame,(int(self.width * 0.6), 0), (int(self.width * 0.6), int(self.height/4)),(255, 255, 255), 3, lineType=cv2.LINE_AA);
		contours, hierarchy = cv2.findContours(proc_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

		cv2.drawContours(init_frame[self.height//2:round((self.height//4)*2.8), 0:self.width], contours, -1, (255,0,0), 5, cv2.LINE_AA, hierarchy, 1 )
		#init_frame = cv2.putText(init_frame, "min_thrhold: %d, max_thrhold: %d" % (min_threshold, max_threshold), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
		
		if len(self.controlFrames) != 10:
			self.controlFrames.append(self.contourProcess(contours))
		else:
			self.controlFrames = self.controlFrames[1:]
			self.controlFrames.append(self.contourProcess(contours))
			avgLeftZone = 0
			avgRightZone = 0
			for i in range(10):
				avgLeftZone += self.controlFrames[i][0]
				avgRightZone += self.controlFrames[i][1]

			avgLeftZone /= 10
			avgRightZone /= 10

			init_frame = cv2.putText(init_frame, "LeftZone: %d, RightZone: %d" % (int(avgLeftZone), int(avgRightZone)), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
			out = ['','']
			if avgLeftZone > 1000:
				out[0] = "Solid"
			else:
				out[0] = "Intermittent"
			if avgRightZone > 1000:
				out[1] = "Solid"
			else:
				out[1] = "Intermittent"

			init_frame = cv2.putText(init_frame, "LeftZone: %s, RightZone: %s" % (out[0], out[1]), (20,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
		return [proc_frame, init_frame]
		            
		            
	def startStream(self):
		try:
		    while self.getState():
		        ret, init_frame = self.stream.read()
		        if ret:
		            proc_frame, init_frame = self.frameProccess(init_frame)
		            
		            cv2.imshow("proc_frame",proc_frame) 
		            cv2.imshow("frame",init_frame)
		        if cv2.waitKey(1) & 0xFF == ord('q'):
		            break
		        else:
		            time.sleep(1/self.fps)
		except KeyboardInterrupt:
		    print("Принудительная остановка")
		    self.stream.release()
		    cv2.destroyAllWindows()
		else:
		    print("Остановка видеопотока")
		    self.stream.release()
		    cv2.destroyAllWindows()
