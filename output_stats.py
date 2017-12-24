import os
import numpy as np

all_labels = {'Reggae': 0, 'Latin': 1, 'RnB': 2, 'Jazz': 3, 'Metal': 4, 'Pop': 5, 'Punk': 6, 'Country': 7, 'New Age': 8, 'Rap': 9, 'Rock': 10, 'World': 11, 'Blues': 12, 'Electronic': 13, 'Folk': 14}
root_path = "/media/mcr222/First_Backup/output/"
labels_totals = [0]*15

for dirName, subdirList, fileList in os.walk(root_path):
    print('Found directory: %s' % dirName)
    first = True
    for fname in fileList:
        if("labels_10batch" in fname):
            #print dirName+"/"+fname
            labels = np.load(dirName+"/"+fname)
            for lab in labels:
                labels_totals[lab] +=1
                

print labels_totals
print sum(labels_totals)

#[94, 180, 605, 366, 298, 2519, 22, 1069, 94, 172, 4273, 28, 46, 660, 74]
#10500
