from matplotlib import pyplot as plt
import numpy as np
import cvxopt
from cvxopt import matrix
import pandas as pd
import csv
from scipy.fftpack import fftn, ifftn # n次元離散フーリエ・逆フーリエ変換 
from mpl_toolkits.mplot3d.axes3d import Axes3D
import cvxpy


#初期設定------------------------------------------------------------
#csvファイルからリスト型で行列を取得  #rstrip("末尾の不要な文字")=>末尾の改行コードを削除
csv_L=[list(map(float,line.rstrip(",\n").split(","))) for line in open('C:/Users/SENS/source/repos/Control_PTU/Control_PTU/csv/1123_dif_Ex1/Split_OP.csv').readlines()]
csv_y=[list(map(float,line.rstrip().split(","))) for line in open('C:/Users/SENS/source/repos/Control_PTU/Control_PTU/csv/1123_dif_Ex1/Split_OP_y.csv').readlines()]
#行列のサイズ確認
N=len(csv_L)
M=len(csv_L[0])
print('L_index', N)
print('L_colmn', M)
print('y_index', len(csv_y))
print('y_colmn', len(csv_y[0]))
#xの事前分布の強度/////////////////
ramuda = 1.0
#ノイズに関する定数
Epsilon = 30
#ロボットプラットフォームの寸法***
Height = 0.273 + 0.01 + 0.091
length_tilt = 0.038 + 0.01 + 0.02
#計測範囲[m]//////////////////////
delta = 0.2
xmin = -1.0
xmax = 1.0
ymin = 0.6
ymax = 2.6
#END初期設定--------------------------------------------------------

#数値の丸め込み関数--------------------------------------------------
def RoundUp(data, interval):
    return int((data + interval - 0.001) / interval) * interval
#int型へのキャスト問題を解消するための関数：(int)(0.3/0.1)=0.2となる問題
def RangetoNum(d_num):
    if d_num >= 0:
        return int(d_num/delta+0.0001)
    else:
        return int(d_num/delta-0.0001)  

#ボクセル関係/////////////////////
Xrange = RangetoNum(xmax) - RangetoNum(xmin)
Yrange = RangetoNum(ymax)
Zrange = RangetoNum(RoundUp(Height + length_tilt, delta))
cell_size = Xrange * Yrange * Zrange
print('x_r',Xrange,'y_r',Yrange,'z_r',Zrange,'call_size',cell_size)
#------------------------------------------------------------------


#最適化問題を解く---------------------------------------------------
#ndarray型にキャスト
L=np.array(csv_L)
y=np.array(csv_y)

###### xの定義 #########################
x = cvxpy.Variable(cell_size)
###### objective: 目的関数 #############
#error = cvxpy.sum_squares(L*x-y)   #二乗和
error = cvxpy.norm((L*x-y),2)       #2ノルム（最小二乗解）
#objective = cvxpy.Minimize(error+ramuda*cvxpy.norm(x,1))  #Lasso回帰
objective = cvxpy.Minimize(error+ramuda*cvxpy.norm(x,2))   #Redge回帰
#objective = cvxpy.Minimize(cvxpy.norm(x,1))
###### constraints: 制約条件 ###########
constraints = [x >= 0]
#constraints = [cvxpy.sum_squares(y-L*x) <= Epsilon]
#最適化問題を解く
prob = cvxpy.Problem(objective, constraints)
print("Optimal value", prob.solve())
print("Optimal var")
print(x.value) # A numpy ndarray.
#------------------------------------------------------------------


#結果をマッピングする------------------------------------------------
#解をリスト形式にキャスト
temp_concentration_list = np.ravel(x.value)
#マッピング用リストの定義
x_list=[]
y_list=[]
z_list=[]
x_fine_list=[]
y_fine_list=[]
z_fine_list=[]
concentration_list = []

#フーリエ変換用のデータを作る  f[i1][i2][i3]
f=[[[0 for i3 in range(Xrange)] for i2 in range(Yrange)] for i1 in range(Zrange)]    #フーリエ変換されるデータの宣言
print("X",Xrange,"Y",Yrange,"Z",Zrange)

for num in range(cell_size):
    #ボクセル座標出す
    x_voxel = int(((num % (Xrange*Yrange)) % Xrange))
    y_voxel = int(((num % (Xrange*Yrange)) / Xrange))
    z_voxel = int((num / (Xrange*Yrange)))
    #print('num',num, 'xv',x_voxel,'yv',y_voxel,'zv',z_voxel)
    f[z_voxel][y_voxel][x_voxel]=temp_concentration_list[num]

#フーリエ変換前
#print("f",f)
x1 = np.arange(0, 390, 1)
y1 = np.ravel(f)
plt.figure(1)
plt.title("Before Fourier Transform")
plt.plot(x1, y1)

#フーリエ変換後
g = fftn(f)  # 多次元高速フーリエ変換
#print("g",g)
g_real=np.real(g) # 実部
g_imag=np.imag(g) # 虚部
g_abs=np.abs(g)   #絶対値
y2 = np.ravel(g)
plt.figure(2)
plt.title("After Fourier Transform")
plt.plot(x1, y2)

g2=np.copy(g) #フーリエ変換後のコピー
g2[(g_abs)<300]=0 #ノイズを切る
y3 = np.ravel(g2)
#print("0_count",y3.tolist().count(0))
plt.figure(3)
plt.title("After Fourier Transform: noise cut")
plt.plot(x1, y3)

#フーリエ逆変換
ff = ifftn(g2)
y4 = np.ravel(ff)
plt.figure(4)
plt.title("Inverse Fourier Transform: noise cut")
plt.plot(x1, y4)



#マッピング用リストの作成 #計測範囲内のみマップを作成する
for num in range(cell_size):    
    #点群でボクセルを表現する
    x_point = int(((num % (Xrange*Yrange)) % Xrange)) * delta #+ xmin
    y_point = int(((num % (Xrange*Yrange)) / Xrange)) * delta -ymin
    z_point = int((num / (Xrange*Yrange))) * delta
    print('num',num, 'x',x_point,'y',y_point,'z',z_point, 'concentration', temp_concentration_list[num])
    range_num = 5
    #if int(num % (Xrange*Yrange) / Xrange) >= RangetoNum(ymin) and temp_concentration_list[num] >= 61 :    #閾値設定 default
    #if int(num % (Xrange*Yrange) / Xrange) >= RangetoNum(ymin) and (y2[num] >= 300 or y2[num] <= -300) :    #閾値設定 Fourier変換後を見る
    if int(num % (Xrange*Yrange) / Xrange) >= RangetoNum(ymin) and y4[num] >= 51 :    #閾値設定 Fourier逆変換後を見る
        for numz in range(range_num):
            for numy in range(range_num):
                for numx in range(range_num):
                    x_fine_list.append(x_point+numx*float(delta/range_num))
                    y_fine_list.append(y_point+numy*float(delta/range_num))
                    z_fine_list.append(z_point+numz*float(delta/range_num))
                    #concentration_list.append(temp_concentration_list[num])     #default
                    concentration_list.append(y4[num]) #Fourier逆変換後


#リストの最大値，最小値
#print('max',max(concentration_list),'min',min(concentration_list))

# 散布図を表示、各点の色を濃度に対応させる
fig = plt.figure()
ax = Axes3D(fig)

# 表示範囲の設定
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.set_zlim(0, 0.6)

#刻み幅の設定
ax.set_xticks(np.arange(0.0, 2.0, 0.2))
ax.set_yticks(np.arange(0.0, 2.0, 0.2))
ax.set_zticks(np.arange(0.0, 0.6, 0.2))

#背景色：引数RGBA
ax.w_xaxis.set_pane_color((0., 0., 0., 0.5))
ax.w_yaxis.set_pane_color((0., 0., 0., 0.5))
ax.w_zaxis.set_pane_color((0., 0., 0., 0.5))

#縦横比 np.diag([x軸, y軸, z軸, 1])
aff = np.diag([1.0, 1.0, 0.3, 1])
aff[0][3] = 0#x軸担当
aff[1][3] = 0#y軸担当
aff[2][3] = 0#z軸担当
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), aff)

#p = ax.scatter(x_list, y_list, z_list, s=10, c=concentration_list, cmap='hot')
p = ax.scatter(x_fine_list, y_fine_list, z_fine_list, s=1, c=concentration_list, cmap='Blues')
# ラベルの設定
#plt.title("Graph Title")
#ax.set_xlabel("X-axis")
#ax.set_ylabel("Y-axis")
#ax.set_zlabel("Z-axis")

# カラーバーを表示
plt.colorbar(p,shrink=0.8)

#実際のガス源の位置プロット
x_resorce = [0.5, 1.0]
y_resorce = [0.5, 1.0]
z_resorce = [0.1, 0.1]
q = ax.scatter(x_resorce, y_resorce, z_resorce, s=30, c="red")

# プロット
plt.figure(5)
plt.show()
#----------------------------------------------------------------------