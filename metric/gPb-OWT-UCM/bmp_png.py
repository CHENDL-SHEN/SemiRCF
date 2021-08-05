import cv2
import os

names = os.listdir('../gPb-OWT-UCM')
for i in names:
    if '.bmp' in i:
        pred = cv2.imread('../gPb-OWT-UCM/' + i, 0)
        cv2.imwrite('../gPb-OWT-UCM/' + i.replace('.bmp','.png'), pred)