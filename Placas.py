#Importamos las librerias
import cv2
import numpy as np
import pytesseract
from PIL import Image

#Realizamos la VideoCaptura
cap = cv2.VideoCapture('Placas.mp4')
Ctexto = '' #Caracteres de las placas, incialmente no se detecta nada

#Realizamos nuestro While True
while True:
    #Se ejecuta todos los fotogramas del video
    #Realizamos la lectura de la VideoCaptura
    ret, frame = cap.read() #Se almacena de frame los fotogramas

    if ret == False:
        break

    #Dibujamos un rectangulo
    cv2.rectangle(frame, (870, 750), (1070, 850), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, Ctexto[0:7], (900, 810), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #Ctexto[0:7] String en Python se trabaja como si fuera un arreglo, 7 por la cantidad
    #de caracteres de la placa

    #Extraemos el ancho y el alto de los fotogramas
    al, an, c = frame.shape

    #Tomar el centro de la imagen
    #Segmentamos o extraemos la región de interes que queremos
    #En x:
    x1 = int(an / 3) #Tomamos 1/3 de la imagen
    x2 = int(x1 * 2) #Hasta el inicio del 3/3 de la imagen

    #En y:
    y1 = int(al / 3) #Tomamos 1/3 de la imagen
    y2= int(y1 * 2) #Hasta el inicio del 3/3 de la imagen

    #Texto
    cv2.rectangle(frame, (x1 + 160, y1 + 500), (1120, 940), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, 'Procesando Placa', (x1 + 180, y1 + 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    #Ubicamos el rectangulo en las zonas extraídas
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    #Ubicamos un recorte a nuestra zona de interes
    #Se extraen los pixeles del rectangulo extraido
    recorte = frame[y1:y2, x1:x2]

    #Procesamiento de la zona de interes
    #La matriz esta en RGB
    #Se separa
    nB = np.matrix(recorte[:, :, 0]) #Matriz azul
    nG = np.matrix(recorte[:, :, 1]) #Matriz verde
    nR = np.matrix(recorte[:, :, 2]) #Matriz roja

    #Color
    #Color amarillo resta entre la matriz verde y azul con esto se detecta el color amarillo
    #Color = cv2.absdiff(nG, nB)
    #Colo blanco la suma de azul, verde y rojo
    Color = cv2.add(nB, nG, nR)

    #Binarizar la imagen para que los colores queden en negro y el blanco en blanco
    _, umbral = cv2.threshold(Color, 40, 255, cv2.THRESH_BINARY)

    #Extraemos y organizamos los contornos para dibujar el más grande
    contornos, _ = cv2.findContours(umbral, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #Ordenar del más grande al más pequeño
    contornos = sorted(contornos, key=lambda x: cv2.contourArea(x), reverse=True)

    #Dibujamos los contornos extraidos
    #Para mostrar los contornos se usa el for
    for contorno in contornos:
        #Detectamos la placa
            x, y, ancho, alto = cv2.boundingRect(contorno)
            #Extraemos las coordenadas para dibujar un rectangulo sobre esos contornos
            #Extraemos las coordenadas
            xpi = x + x1        #Coordenadas de la placa en x inicial
            ypi = y + y1        #Coordenadas de la placa en y inicial

            xpf = x + ancho + x1        #Coordenadas de la placa en x final
            ypf = y + alto + y1         #Coordenadas de la placa en y final

            #x1 y y1 coordenadas de la imagen original

            #Dibujamos el rectangulo
            cv2.rectangle(frame, (xpi, ypi), (xpf, ypf), (255, 255, 0), 2)

            #Extraemos a los píxeles que pertenecen a la placa pero con los coloresoriginales
            placa = frame[ypi:ypf, xpi:xpf] #frame para que quede en RGB como la imagen original

            #Extraemos el ancho y el alto de los fotogramas de la placa
            alp, anp, cp = placa.shape
            #print(alp, anp)

            #Se usara para saber en que momento utilizar la extracción de caracteres
            #Cuando la placa esta en un tamaño determinado, dira si se va haciendo más grande
            #cuando se considere que esta lo suficientemente grande para realizar la lectura del texto
            #y se exitosa ahí se realizará el procesamiento

            #Procesando los píxeles para extraer los valores de las placas
            Mva = np.zeros((alp, anp))

            #Normalizamos las matrices
            #Se extraen las matrices RGB pero de la placa pequeña que se extrajo
            mBp = np.matrix(recorte[:, :, 0])  # Matriz azul
            mGp = np.matrix(recorte[:, :, 1])  # Matriz verde
            mRp = np.matrix(recorte[:, :, 2])  # Matriz roja

            #Creamos una máscara
            for col in range(0, alp):
                for fil in range(0, anp):
                    #Se resta 255 al máximo de la matriz R,G y B con el fin de resaltar el color negro
                    #de los caracteres, quedaran solo los caracteres el resto de colores se borra
                    Max = max(mRp[col, fil], mGp[col, fil], mBp[col, fil])
                    Mva[col, fil] = 255-Max #Resaltamos el negro

            #Binarizamos la imagen
            #Los colores que hayan quedado lo eliminamo con un umbral con el fin
            #para que me quede solo los caracteres
            _, bin = cv2.threshold(Mva, 150, 255, cv2.THRESH_BINARY)

            #Convertimos la matriz en imagen
            bin = bin.reshape(alp, anp)
            bin = Image.fromarray(bin)
            bin = bin.convert("L")

            #Nos aseguramos de tener un buen tamaño de placa
            if alp >= 36 and anp >= 82:

                #Declaramos la dirección de Pytesseract
                pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract_OCR_Python\tesseract.exe'

                #Extraemos el texto
                config = "--psm 1"
                texto = pytesseract.image_to_string(bin, config=config)

                #If para no mostrar basura
                if len(texto) >=7: #Queremos que este la placa completa
                    #print(texto[0:7])

                    Ctexto = texto #textotodo el tiempo se va actualizando y queremos
                    #una copia de seguridad y se muestra en la parte inicial de este código

                    #Mostramos los valores que nos interesan
                    #cv2.putText(frame, Ctexto[0:7], (900, 810), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                break

                #Mostramos el recorte
                #cv2.imshow("Recorte", bin)

    #Mostramos el recorte en gris

    cv2.imshow("Vehiculos", frame)

    #Leemos una tecla#
    t = cv2.waitKey(1) #Con Esc se cierra el programa

    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()