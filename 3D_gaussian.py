from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
import scipy.stats
import pandas as pd
import csv

#初期設定--------------------------------------------------------
#ロボットプラットフォームの寸法////////////////
Height = 0.273 + 0.01 + 0.091
length_tilt = 0.038 + 0.01 + 0.02
#計測範囲[m]/////////////////////////////////
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

#3次元ガウス分布の平均，分散
mean = np.array([1.0, 1.0, 0.0])
cov  = np.diag([0.05,0.05,0.05])

#マッピング用リストの定義
x_list=[]
y_list=[]
z_list=[]
concentration_list = []
temp_concentration_list = []

#マッピング用リストの作成 #計測範囲内のみマップを作成する
for num in range(cell_size):
    #ボクセル座標出す => xyz座標に直す => 位置補正する => ボクセルの中心に点を置く
    x_point = int(((num % (Xrange*Yrange)) % Xrange)) * delta + (delta/2)
    y_point = int(((num % (Xrange*Yrange)) / Xrange)) * delta + (delta/2) -ymin
    z_point = int((num / (Xrange*Yrange))) * delta + (delta/2)
    print('num',num, 'x',x_point,'y',y_point,'z',z_point)
    #GT書き出し用
    temp_grid = [x_point,y_point,z_point]
    temp_concentration_list.append(scipy.stats.multivariate_normal(mean,cov).pdf(temp_grid)*10)
    if int(num % (Xrange*Yrange) / Xrange) >= RangetoNum(ymin): 
        x_list.append(x_point)
        y_list.append(y_point)
        z_list.append(z_point)
        grid = [x_point,y_point,z_point]
        #3次元ガウス分布の計算
        concentration_list.append(scipy.stats.multivariate_normal(mean,cov).pdf(grid)*10)  
        #表示するセルの閾値
        if(scipy.stats.multivariate_normal(mean,cov).pdf(grid)<0.1):
            x_list.pop(-1)
            y_list.pop(-1)
            z_list.pop(-1)
            concentration_list.pop(-1)
            
#Ground Truesの書き出し
#pandasのDataFrame型に変換
GT_list = pd.DataFrame(temp_concentration_list)
#CSV出力
GT_list.to_csv('C:/Users/SENS/source/repos/Control_PTU/Control_PTU/csv/simulation/GT_var005.csv')

# 散布図を表示、各点の色を濃度に対応させる
fig = plt.figure(1)
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

p = ax.scatter(x_list, y_list, z_list, s=10, c=concentration_list, cmap='Blues')

# ラベルの設定
#plt.title("Graph Title")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")

# カラーバーを表示
plt.colorbar(p,shrink=0.8)

# プロット
plt.show()