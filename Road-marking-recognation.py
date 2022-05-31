import cv2 #импорт библиотеки cv2
from StreamHandler import * #импорт обработчика потока

print("Захват видеопотока...")

try: #попытка открытия видеопотока
    stream = cv2.VideoCapture("example.mp4") #импорт видеопотока (входного файла)
except: #вывод ошибки при любых исключениях
    print("Ошибка")

if(stream.isOpened()): #проверка работы потока
    print("Успешный захват")
else:
    print("Ошибка захвата")

strHandler = StreamHandler(stream) #создание экземпляра обработчика потока
print("Размеры видео потока: %d x %d" % (strHandler.getSize()[0], strHandler.getSize()[1])) #вывод размеров потока
print("Кадров в секунду: ", strHandler.fps) #вывод fps потока


strHandler.startStream() #запуск обработки потока

