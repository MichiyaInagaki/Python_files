from matplotlib import pyplot as plt
import numpy as np
import cvxopt
from cvxopt import matrix
import pandas as pd
import csv
from mpl_toolkits.mplot3d.axes3d import Axes3D


#初期設定------------------------------------------------------------
#csvファイルからリスト型で行列を取得  #rstrip("末尾の不要な文字")=>末尾の改行コードを削除
csv_L=[list(map(float,line.rstrip(",\n").split(","))) for line in open('C:/Users/SENS/source/repos/Control_PTU/Control_PTU/csv/Ex1/Split_OP.csv').readlines()]
csv_y=[list(map(float,line.rstrip().split(","))) for line in open('C:/Users/SENS/source/repos/Control_PTU/Control_PTU/csv/Ex1/Split_OP_y.csv').readlines()]
#xの事前分布の強度/////////////////
ramuda = 1.0
#ロボットプラットフォームの寸法
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

#行列のサイズ確認
N=len(csv_L)
M=len(csv_L[0])
print('L_index', N)
print('L_colmn', M)
print('y_index', len(csv_y))
print('y_colmn', len(csv_y[0]))

#ndarray型にキャストしてからmatrix型にキャスト
L=matrix(np.array(csv_L))
y=matrix(np.array(csv_y))

#必要な行列を作成
I=matrix(np.eye(M))     #単位行列

#最後にmatrix型にキャスト
P=2*(L.T)*L+2*ramuda*I
q=(-1)*L.T*y
r=(y.T)*y
#ndarray型にキャストしてからmatrix型にキャスト
G=matrix(-np.eye(M))
h=matrix(np.zeros((M,1,)))

sol=cvxopt.solvers.qp(P,q,G,h)

print(sol)
print(sol["x"])
print("primal objective", sol["primal objective"])

#pandasのDataFrame型に変換
res_x=pd.DataFrame(list(sol["x"]))
#CSV出力
#res_x.to_csv('C:/Users/SENS/source/repos/Control_PTU/Control_PTU/csv/result_05_sample.csv')

#解をリスト形式にキャスト
temp_concentratin_list = list(sol["x"])

x_list=[]
y_list=[]
z_list=[]
x_fine_list=[]
y_fine_list=[]
z_fine_list=[]
concentration_list = []

#マッピング用リストの作成 #計測範囲内のみマップを作成する
for num in range(cell_size):
    '''
    #ボクセル座標出す => xyz座標に直す => 位置補正する => ボクセルの中心に点を置く
    x_point = int(((num % (Xrange*Yrange)) % Xrange)) * delta + xmin + (delta/2)
    y_point = int(((num % (Xrange*Yrange)) / Xrange)) * delta + (delta/2) -ymin
    z_point = int((num / (Xrange*Yrange))) * delta + (delta/2)
    print('num',num, 'x',x_point,'y',y_point,'z',z_point)
    if int(num % (Xrange*Yrange) / Xrange) >= RangetoNum(ymin): #and temp_concentratin_list[num] >= 200 :
        x_list.append(x_point)
        y_list.append(y_point)
        z_list.append(z_point)
        concentration_list.append(temp_concentratin_list[num])
    '''

    #点群でボクセルを表現する
    x_point = int(((num % (Xrange*Yrange)) % Xrange)) * delta + xmin
    y_point = int(((num % (Xrange*Yrange)) / Xrange)) * delta -ymin
    z_point = int((num / (Xrange*Yrange))) * delta
    print('num',num, 'x',x_point,'y',y_point,'z',z_point)
    range_num = 5
    if int(num % (Xrange*Yrange) / Xrange) >= RangetoNum(ymin) and temp_concentratin_list[num] >= 400 : 
        for numz in range(range_num):
            for numy in range(range_num):
                for numx in range(range_num):
                    x_fine_list.append(x_point+numx*float(delta/range_num))
                    y_fine_list.append(y_point+numy*float(delta/range_num))
                    z_fine_list.append(z_point+numz*float(delta/range_num))
                    concentration_list.append(temp_concentratin_list[num])


# 散布図を表示、各点の色を濃度に対応させる
fig = plt.figure()
ax = Axes3D(fig)

# 表示範囲の設定
ax.set_xlim(-1.0, 1.0)
ax.set_ylim(0, 2)
ax.set_zlim(0, 0.6)

#刻み幅の設定
ax.set_xticks(np.arange(-1.0, 1.0, 0.2))
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
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
# カラーバーを表示
plt.colorbar(p)

# プロット
plt.show()