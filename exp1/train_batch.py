import os

for i in range(1, 5):
    for j in range(1, 5):
        os.system('python train.py ../imgs/content/content{}.jpg ../imgs/style/style{}.jpg -o ./output/output{}_{}.png -s 500'.format(i,j,i,j))    