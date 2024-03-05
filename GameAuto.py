# Game Frame Size 512 * 384
#Graphics Fantastic

from ultralytics import YOLO
from cv2 import imshow,circle,line,putText,rectangle,waitKey,destroyAllWindows
from numpy import array,transpose
from time import sleep, time
from windowcapture import WindowCapture
from keyboard import send
i=0
loop_time = time()

lst=array([ [0,0, 1, 2, 3],
            [1,0, 20, 2, 30],
            [2,0, 30, 2, 30]])

model = YOLO('model/e300(n)_224.pt')
wincap = WindowCapture("Flappy Bird New")

while(True):
    image = wincap.get_screenshot() # capture window

    # image=image[0:800,300:900]
    image=image[0:348,232:435]
    image = transpose(image, (0, 1, 2))[:, :, :3]

    results = model(image,conf=0.7,verbose=False,)  # predict on an image
    image=array(image)
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        for box,name in zip(result.boxes,result.names):
            class_id = box.cls.item()
            nameTry=box.cls.item()
            x1, y1, x2, y2 =box.xyxy[0].tolist()
            x1, y1, x2, y2=int(x1), int(y1), int(x2),int(y2)

            
            if class_id==0:
                lst[0][1],lst[0][2],lst[0][3],lst[0][4]=x1, y1, x2, y2
                # rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            if class_id==1:
                lst[1][1],lst[1][2],lst[1][3],lst[1][4]=x1, y1, x2, y2
                # rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            if class_id==2:
                lst[2][1],lst[2][2],lst[2][3],lst[2][4]=x1, y1, x2, y2
                # rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        bird=lst[0][4]
        pipedown=lst[1][2]-29                   
        pipeup=lst[2][4]+30             
        # circle(image,(lst[0][3],bird),2,(255,0,0),2)
        # line(image,(lst[1][1],pipedown),(  lst[1][3],pipedown),(0,0,255),2)
        # line(image,(lst[2][1],pipeup) ,(lst[2][3],pipeup),(0,0,255),2)
        # circle(image,(lst[1][1],pipedown),2,(0,255,0),2)
        # circle(image,(lst[2][1],pipeup),2,(0,0, 255),2)

        if  bird<pipeup:
            sleep(.01)
            break

        if bird-pipedown > 50:
            send('space')
            sleep(.0001)
            break
        
        if bird>pipedown :                                                                                                    
            sleep(.255)                                                        

    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    imshow("Image",image)

    if waitKey(1) == ord('q'):
        destroyAllWindows()
        break


# roboticverse 
# Chetan Parihar
# 05-03-2024