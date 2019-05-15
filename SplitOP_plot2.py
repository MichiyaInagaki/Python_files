import cvxopt
from cvxopt import matrix
import pandas as pd
import csv
from scipy.fftpack import fftn, ifftn # n次元離散フーリエ・逆フーリエ変換 
from mpl_toolkits.mplot3d.axes3d import Axes3D
import cvxpy
import pywt
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
 
fig = plt.figure()
ax = Axes3D(fig)

#初期設定------------------------------------------------------------
#csvファイルからリスト型で行列を取得  #rstrip("末尾の不要な文字")=>末尾の改行コードを削除
csv_mesh = [list(map(float,line.rstrip(",\n").split(","))) for line in open('C:/Users/SENS/source/repos/Control_PTU/Control_PTU/csv/mesh_test/mesh11_hoge.csv').readlines()]
#行列のサイズ確認//////////////////
N=len(csv_mesh)
M=len(csv_mesh[0])
print('csv_mesh_index', N)
print('csv_mesh_colmn', M)
#END初期設定--------------------------------------------------------


#ppmm計算処理---------------------------------------------------
#ndarray型にキャスト
mesh = np.array(csv_mesh)
#listの定義
intersection=[]
temp_ppm=[]
ppmm_list=[]
x_inter=[]
y_inter=[]
z_inter=[]

for i in range(0,N):          #行
    optical_len = mesh[i][0]  #光路長
    for j in range(1,M,3):    #列：3ずつ増加
        #要素がなくなったらbreak
        if mesh[i][j]==-1:
            break
        #交点格納 
        x_inter.append(mesh[i][j])
        y_inter.append(mesh[i][j+1])
        z_inter.append(mesh[i][j+2])
    if i>=848:    #表示するパスを絞る
        #出力
        ax.plot(x_inter, y_inter, z_inter, "o-", color="#ff8c00", ms=0.5, mew=0.5)
    if i>=1000:
        break
    #初期化
    x_inter.clear()
    y_inter.clear() 
    z_inter.clear() 



# 軸ラベル
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()