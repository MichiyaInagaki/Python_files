#メッシュ座標からシミュレーション上のLMmの値[ppm・m]をCSV出力させるプログラム

from matplotlib import pyplot as plt
import numpy as np
import cvxopt
from cvxopt import matrix
import pandas as pd
import csv
from scipy.fftpack import fftn, ifftn # n次元離散フーリエ・逆フーリエ変換 
from mpl_toolkits.mplot3d.axes3d import Axes3D
import cvxpy
import pywt
import scipy.stats

#初期設定------------------------------------------------------------
#csvファイルからリスト型で行列を取得  #rstrip("末尾の不要な文字")=>末尾の改行コードを削除
csv_mesh = [list(map(float,line.rstrip(",\n").split(","))) for line in open('C:/Users/SENS/source/repos/Control_PTU/Control_PTU/csv/mesh_test/mesh6.csv').readlines()]
#3次元ガウス分布の平均，分散////////
mean = np.array([1.0, 1.6, 0.0])    #***
cov  = np.diag([0.05,0.05,0.05])    #***
#行列のサイズ確認//////////////////
N=len(csv_mesh)
M=len(csv_mesh[0])
print('csv_mesh_index', N)
print('csv_mesh_colmn', M)
#ロボットプラットフォームの寸法/////
Height = 0.273 + 0.01 + 0.091
length_tilt = 0.038 + 0.01 + 0.02
#計測範囲[m]//////////////////////
delta = 0.2
xmin = -1.0
xmax = 1.0
ymin = 0.6
ymax = 2.6
#数値の丸め込み関数///////////////////////////
def RoundUp(data, interval):
    return int((data + interval - 0.001) / interval) * interval
#int型へのキャスト問題を解消するための関数：(int)(0.3/0.1)=0.2となる問題
def RangetoNum(d_num):
    if d_num >= 0:
        return int(d_num/delta+0.0001)
    else:
        return int(d_num/delta-0.0001)
#ボクセル関係////////////////////////////////
Xrange = RangetoNum(xmax) - RangetoNum(xmin)
Yrange = RangetoNum(ymax)
Zrange = RangetoNum(RoundUp(Height + length_tilt, delta))
cell_size = Xrange * Yrange * Zrange
print('x_r',Xrange,'y_r',Yrange,'z_r',Zrange,'call_size',cell_size)
#END初期設定--------------------------------------------------------


#ppmm計算処理---------------------------------------------------
#ndarray型にキャスト
mesh = np.array(csv_mesh)
#listの定義
intersection=[]
temp_ppm=[]
ppmm_list=[]

for i in range(0,N):          #行
    optical_len = mesh[i][0]  #光路長
    for j in range(1,M,3):    #列：3ずつ増加
        #要素がなくなったらbreak
        if mesh[i][j]==-1:
            break
        #交点格納
        intersection = [mesh[i][j], mesh[i][j+1], mesh[i][j+2]] 
        #print(intersection)  
        #3次元ガウス分布の計算
        temp_ppm.append(scipy.stats.multivariate_normal(mean,cov).pdf(intersection))
    if optical_len == -1:   #例外処理
        ppmm_list.append(0)
    else:    
        s = sum(temp_ppm)
        data_num = len(temp_ppm)
        ppmm = float(s)/data_num*optical_len   #ppmm計算→平均×長さ
        ppmm_list.append(ppmm)  #最終配列に格納
        temp_ppm.clear()        #初期化

#変形
ppmm_list_csv = np.reshape(ppmm_list,(N,1))
#print(ppmm_list_csv)    

#書き出し
with open("C:/Users/SENS/source/repos/Control_PTU/Control_PTU/csv/mesh_test/ppmm6.csv", "w", encoding="Shift_jis") as f: 
    writer = csv.writer(f, lineterminator="\n") 
    writer.writerows(ppmm_list_csv)    
