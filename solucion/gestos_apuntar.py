# El ejercicio, para realizar en grupos, consiste comentar este código y sugerir mejoras

import numpy as np
import cv2  
import mediapipe as mp

mpHands = mp.solutions.hands
mp_dedos = mpHands.HandLandmark
hands = mpHands.Hands(max_num_hands=1,min_detection_confidence=0.8,min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
(ancho,alto)=(720,1280)
cap = cv2.VideoCapture('mano_bien_4.mp4')
print('fps',cap.get(cv2.CAP_PROP_FPS))
lista_imagenes_nombres=['_intro.png','0_codigo.PNG','1_puntos.PNG','2_flanges.PNG','3_coseno.PNG','4_apuntar.PNG','5_puntero.PNG']
lista_imagenes= [cv2.resize(cv2.imread('imagenes/'+nombre_imagen).transpose((1,0,2,)), (720,1280), interpolation = cv2.INTER_AREA) for nombre_imagen in lista_imagenes_nombres]
lista_imagenes_shapes=[imagen.shape for imagen in lista_imagenes]
ret, img = cap.read()
ancho_video,alto_video= int(cap.get(3)), int(cap.get(4))
print('ancho_video,alto_video',ancho_video,alto_video)
(ancho,alto)=(720,1280)
alto_imagen=int(ancho/ancho_video*alto_video)
wk=0
paso=0
repetir=True
if not cap.isOpened():
    print("Cannot open camera")
    repetir=False
contador=0
while repetir:
    contador += 1
    if contador%100==0:print(contador)
    frame=lista_imagenes[paso]
    ret, img = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    centro=[ancho//2,alto//2]
    centro_np=np.array(centro)
    img1=cv2.resize(img, (ancho,alto_imagen), interpolation = cv2.INTER_AREA)
    frame[alto-alto_imagen:,:,:]=img1
    frame=cv2.flip(frame,1)
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results =hands.process(imgRGB) # realiza el procesamiento con mediapipe
                                   # para ubicar las manos en la imagen

    if results.multi_hand_landmarks: # si se encontraron manos ...
        for handLms in results.multi_hand_landmarks: # para cada mano ...
            if paso == 1:
                mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS) # dibuja la mano

            puntos=np.array([[int(punto.x*ancho),int(punto.y*alto)] for punto in handLms.landmark ]).T
            
            if paso == 2:
                for i in range(puntos.shape[1]):
                    l_punto=list(puntos[:,i])
                    cv2.line(frame,centro,l_punto,(255,0,0),1)
                    cv2.putText(frame,str(i),l_punto, cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)

            if paso == 3:
                for i in [2,3,6,7,10,11,14,15,18,19]:
                    l_punto=list(puntos[:,i])
                    cv2.line(frame,centro,l_punto,(255,0,0),1)

            falanges_medias = np.zeros((2,5),dtype=np.int16)  
            coss_con_indice = np.zeros(5)  
            dedo_indice = puntos[:,4+3]-puntos[:,4+2]
            for i in range(5): #para cada dedo
                vector_inicial=puntos[:,4*i+2]
                vector_final=puntos[:,4*i+3]
                falanges_medias[:,i] = vector_final-vector_inicial
                cos_con_indice=round(falanges_medias[:,i].dot(dedo_indice)/(np.linalg.norm(falanges_medias[:,i])*np.linalg.norm(dedo_indice)),2)
                coss_con_indice[i]=cos_con_indice
                if paso >= 3:
                    cv2.arrowedLine(frame, list(vector_inicial), list(vector_final),(255-(i*60),(i*180)%256,i*60), 6)
                    cv2.arrowedLine(frame, centro, list(falanges_medias[:,i]+centro_np),(255-(i*60),(i*180)%256,i*60), 6)
                if paso >= 4 :
                    cv2.putText(frame,str(cos_con_indice),list(vector_final), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
            
            suma_cos = np.sum(coss_con_indice[2:])
            if paso == 5:
                cv2.putText(frame,str(suma_cos),(10,alto-10), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
                if suma_cos < -2.7:
                    cv2.circle(frame,list((puntos[:,8]).astype(np.int16)), 10, (0,0,255), 5)
            if paso == 6:
                cv2.putText(frame,str(suma_cos),(10,alto-10), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
                if suma_cos < -2.7:
                    cv2.circle(frame,list((puntos[:,8]+(1-coss_con_indice[0])*dedo_indice*20).astype(np.int16)), 10, (0,0,255), 5)

    resized =   cv2.resize(frame, (360,640), interpolation = cv2.INTER_AREA) 
    cv2.imshow('frame', resized)
    wk=cv2.waitKey(50)
    if wk == ord('q'):
        break
    elif wk == ord('a'):
        paso -= 1
    elif wk == ord('d'):
        paso += 1
    factor=756/1582
    if   contador ==  int(factor*100):
        paso=2
    elif contador ==  int(factor*190):
        paso=3
    elif contador ==  int(factor*550):
        paso=4
    elif contador ==  int(factor*800):
        paso=5
    elif contador == int(factor*1010):
        paso=6


print(contador)
cap.release()
cv2.destroyAllWindows()