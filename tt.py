import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')


part = 'train'
all_files = os.listdir('C:/project1/'+part+'/audio')
labels =[]
label_list = open('C:/project1/labels.txt').read().strip().split('\n')
for i in all_files:
    label = i.split('_')[0]
    #print(label)
    if label == 'reed':
        labels.append(label_list.index(label))
        for a in enumerate(labels):
            print(a)
