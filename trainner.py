import cv2,os
import numpy as np
from PIL import Image

recognizer = cv2.createLBPHFaceRecognizer()
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

def getImagesAndLabels(path):
    #obtenir le schéma pour les images
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #la création d'une liste de visage
    faceSamples=[]
    #initialiser l'identifiant
    Ids=[]
    #boucle pour charger les identifiants
    for imagePath in imagePaths:
        #conversion vers l'espace de niveau de gris
        pilImage=Image.open(imagePath).convert('L')
        #convertir la liste en une liste numpy(vecteur d'images)
        imageNp=np.array(pilImage,'uint8')
        #obtenir l'dentifiant de chaque image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces=detector.detectMultiScale(imageNp)
        #charger la liste
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
            cv2.imshow("trainning",imageNp)
            cv2.waitKey(10)
    return faceSamples,Ids


faces,Ids = getImagesAndLabels('dataSet')
recognizer.train(faces, np.array(Ids))
recognizer.save('trainner/trainner.yml')
cv2.destroyAllWindows()

