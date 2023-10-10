import os
import glob
import tqdm
import cv2
import numpy as np
inputpath = r"E:\pwg\FDDAFFNet\Levir_CD\test"
outputpath = r"E:\pwg\FDDAFFNet\Levir_CD_256\test_2"

if not os.path.exists(outputpath):
    os.makedirs(outputpath)
if not os.path.exists(outputpath + "/A/"):
    os.makedirs(outputpath+ "/A/")
if not os.path.exists(outputpath+ "/B/"):
    os.makedirs(outputpath+ "/B/")
if not os.path.exists(outputpath+ "/label/"):
    os.makedirs(outputpath+ "/label/")
Apaths = glob.glob(os.path.join(inputpath,"A/*.png"))
Bpaths = glob.glob(os.path.join(inputpath,"B/*.png"))
labelpaths = glob.glob(os.path.join(inputpath,"label/*.png"))

for aApath in tqdm.tqdm(Apaths):
    Aimg = cv2.imread(aApath)
    Bimg = cv2.imread(Bpaths[Apaths.index(aApath)])
    labelimg = cv2.imread(labelpaths[Apaths.index(aApath)])
    for x in range(0,4):
        for y in range(0, 4):
            if np.max(labelimg[x * 256:(x + 1) * 256, y * 256:(y + 1) * 256])>=0:
                cv2.imwrite(
                    outputpath + "/A/" + aApath.split("\\")[-1].split(".")[0] + "_" + str(x) + "_" + str(y) + ".png",
                    Aimg[x * 256:(x + 1) * 256, y * 256:(y + 1) * 256])
                cv2.imwrite(
                    outputpath + "/B/" + aApath.split("\\")[-1].split(".")[0] + "_" + str(x) + "_" + str(y) + ".png",
                    Bimg[x * 256:(x + 1) * 256, y * 256:(y + 1) * 256])
                cv2.imwrite(
                    outputpath + "/label/" + aApath.split("\\")[-1].split(".")[0] + "_" + str(x) + "_" + str(
                        y) + ".png",
                    labelimg[x * 256:(x + 1) * 256, y * 256:(y + 1) * 256])



