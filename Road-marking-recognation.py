import cv2 #импортирование библиотеки
from StreamHandler import *

print("Захват видеопотока...")
try:
    stream = cv2.VideoCapture("example.mp4") #импорт видеопотока (входного файла)
except:
    print("Ошибка")

if(stream.isOpened()):
    print("Успешный захват")
else:
    print("Ошибка захвата")

strHandler = StreamHandler(stream)
print("Размеры видео потока: %d x %d" % (strHandler.getSize()[0], strHandler.getSize()[1])) #вывод размеров
print("Кадров в секунду: ", strHandler.fps)


strHandler.startStream()

