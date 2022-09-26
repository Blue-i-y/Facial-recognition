import face_recognition as fr
import cv2

import numpy as np
import os
#TAIN MODEL

"""
Ici j'ai 2 listes qui stockent 
les noms des images (personnes) et leurs 
encodages de visage respectifs.

"""


path = "./photos/train/"
known__name = []
Known_name_encoding = []

images = os.listdir(path)

"""
face_encodings est un vecteur de valeurs représentant 
les mesures importantes entre les caractéristiques distinctives 
d'un visage comme la distance entre les yeux, la largeur 
du front, etc.
---------------------------------------------------------------
Nous parcourons chacune des images de notre répertoire train,
extrayons le nom de la personne dans l'image, calculons son vecteur 
d'encodage de visage et stockons les informations dans les listes respectives.
"""

for _ in images:
    image = fr.load_image_file(path + _)
    image_path = path + _
    encoding = fr.face_encodings(image)[0]

Known_name_encoding.append(encoding)
known__name.append(os.path.splitext(os.path.basename(image_path))[0].capitalize())


#TEST MODEL

test_image = "./test/test.jpg"
image = cv2.imread(test_image)

face_locations = fr.face_locations(image)
face_encoding = fr.face_encoding(image, face_locations)

for (top, right, bottom, left), face_encoding in 
