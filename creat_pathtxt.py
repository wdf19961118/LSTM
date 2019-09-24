import os

dir='/home/lab226/wdf/imgsrc'
fp = open('./img_path.txt','w+')
imgfile_list = os.listdir('/home/lab226/wdf/imgsrc')
imgfile_list.sort(key= lambda x:int(x[:]))
#print(img_list)
seqsize =17
for imgfile in imgfile_list:
    filepath = os.path.join(dir,imgfile)
    img_list = os.listdir(filepath)
    img_list.sort(key=lambda x: int(x[:-4]))
    #滑窗取序列，步长为8
    for i in range(0, len(img_list)-seqsize, 8):
        for j in range(i,i+seqsize):
             img = img_list[j]
             path = os.path.join(filepath, img)
             if j == i+seqsize-1:
                fp.write(path+'\n')
             else:
                fp.write(path+' ')
fp.close()