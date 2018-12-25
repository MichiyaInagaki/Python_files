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
csv_L=[list(map(float,line.rstrip(",\n").split(","))) for line in open('C:/Users/SENS/source/repos/Control_PTU/Control_PTU/csv/1123_dif_Ex1_dec/Split_OP.csv').readlines()]
csv_y=[list(map(float,line.rstrip().split(","))) for line in open('C:/Users/SENS/source/repos/Control_PTU/Control_PTU/csv/1123_dif_Ex1_dec/Split_OP_y.csv').readlines()]
#行列のサイズ確認
N=len(csv_L)
M=len(csv_L[0])
print('L_index', N)
print('L_colmn', M)
print('y_index', len(csv_y))
print('y_colmn', len(csv_y[0]))
#xの事前分布の強度/////////////////
ramuda = 1.0
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


#ISTA-----------------------------------------------------
#軟しきい値関数
def SoftThr(xt, lam):
    yt = np.zeros(xt.shape)
    for i in range(xt.shape[0]):
        if xt[i]>lam:
            yt[i]=xt[i]-lam
        elif xt[i]<lam:
            yt[i]=xt[i]+lam
    return yt

#初期条件
x_ista = (1/N)*fftn(L.T @ y)
lamuda = 1.0    #L1ノルム係数
Lip0 = 1        #リプシッツ定数
min = 1000      #終了条件用

for p in range(2000):
    kx_ista = x_ista                                            #kx前回の結果 for Lip
    #three_x_ista = np.reshape(x_ista,(Zrange,Yrange,Xrange))    #3次元配列版x^^^
    #z_ista = ifftn(three_x_ista)                                #3次元逆フーリエ変換^^^
    #z_ista = np.reshape(z_ista,(cell_size,1))                   #1次元に直す^^^
    z_ista = ifftn(x_ista)  #^^^
    ktemp = 0.5*np.linalg.norm(y-L@z_ista, ord=2)*np.linalg.norm(y-L@z_ista, ord=2)      #前回結果におけるコスト関数の計算 for Lip
    Lip = Lip0                                                  #リプシッツ定数をリセット
    grad_x = (-1)*(1/N)*fftn(L.T@(y-L@z_ista))                  #勾配計算 ***
    MMx = x_ista - grad_x*(1/Lip)                               #更新点候補 ***
    #three_MMx = np.reshape(MMx,(Zrange,Yrange,Xrange))          #3次元配列版MMx^^^
    #MMz = ifftn(three_MMx)                                      #3次元逆フーリエ変換^^^
    #MMz = np.reshape(MMz,(cell_size,1))                         #1次元に直す^^^
    MMz = ifftn(MMx)    #^^^
    temp = 0.5*np.linalg.norm(y-L@MMz, ord=2)*np.linalg.norm(y-L@MMz, ord=2)   #コスト関数の一部を計算 for Lip
    MMtemp = ktemp + grad_x.T@(MMx-kx_ista) + 0.5 * Lip * np.linalg.norm(MMx-kx_ista, ord=2)*np.linalg.norm(MMx-kx_ista, ord=2)    #メジャライザーを計算 for Lip
    #メジャライザーがコスト関数を下回るようにLipを大きくする
    while MMtemp < temp:
        Lip=Lip*1.1                                             #リプシッツ定数を大きくしてみる
        MMx = x_ista - grad_x*(1/Lip)                           #更新候補位置再計算
        three_MMx = np.reshape(MMx,(Zrange,Yrange,Xrange))      #3次元配列版MMx^^^
        MMz = ifftn(three_MMx)                                  #3次元逆フーリエ変換^^^
        MMz = np.reshape(MMz,(cell_size,1))                     #1次元に直す^^^
        MMz = ifftn(MMx)    #^^^
        temp = 0.5*np.linalg.norm(y-L@MMz, ord=2)*np.linalg.norm(y-L@MMz, ord=2)    #コスト関数の一部を再計算
        MMtemp = ktemp + grad_x.T@(MMx-kx_ista) + 0.5*Lip*np.linalg.norm(MMx-kx_ista, ord=2)*np.linalg.norm(MMx-kx_ista, ord=2)
    #軟判定しきい値関数の適用
    x_ista=SoftThr(MMx.real, lamuda/Lip)
    print("step",p,"d",np.linalg.norm(x_ista-kx_ista,ord=2),"d2",(np.linalg.norm(x_ista-kx_ista,ord=2)/np.linalg.norm(kx_ista, ord=2)),"min",min)
    #更新幅が小さくなれば更新終了
    #if (np.linalg.norm(x_ista-kx_ista,ord=2)/np.linalg.norm(kx_ista, ord=2)) < 0.0015:
        #break
    #if min > np.linalg.norm(x_ista-kx_ista,ord=2)/np.linalg.norm(kx_ista, ord=2):
        #min = np.linalg.norm(x_ista-kx_ista,ord=2)/np.linalg.norm(kx_ista, ord=2)   
    #相対誤差が大きいほうに転じたとき反復を終了する
    #if min+0.000015 < np.linalg.norm(x_ista-kx_ista,ord=2)/np.linalg.norm(kx_ista, ord=2):
        #break    

#END ISTA-------------------------------------------------

#プロットしてみよー
#print("x_ista",x_ista)
xx1 = np.arange(0, 390, 1)
yy1 = x_ista
plt.figure(1)
plt.title("x_ista")
plt.plot(xx1, yy1)

g2=np.copy(yy1)     #フーリエ変換後のコピー
g_abs=np.abs(g2)    #絶対値

#n番目に大きい要素を取得するためにソートする
sort_x = sorted(g_abs)

g2[(g_abs) < sort_x[int(cell_size/100*70)]] = 0 #ノイズを切る 70%スパース
plt.figure(2)
plt.title("x_ista noise_cut")
plt.plot(xx1, g2)

#x_ista_3d = np.reshape(x_ista,(Zrange,Yrange,Xrange))   #3次元配列にする 
#yy2_3d = ifftn(x_ista_3d)                               #逆フーリエ変換をかける
#yy2 = np.reshape(yy2_3d,(cell_size,1))                  #1次元に直す
#print("yy2",yy2_3d)
yy2 = ifftn(g2)
plt.figure(3)
plt.title("ifftn_x_ista")
plt.plot(xx1, yy2)



#結果をマッピングする------------------------------------------------
#解をリスト形式にキャスト
temp_concentration_list = np.ravel(yy2)
#マッピング用リストの定義
x_list=[]
y_list=[]
z_list=[]
x_fine_list=[]
y_fine_list=[]
z_fine_list=[]
concentration_list = []

#見やすいマッピング用の閾値を取るためにソートする
sort_concentration_list = sorted(temp_concentration_list)
cut_concentration = sort_concentration_list[int(cell_size/100*93)]
print("voxel_cut_concentration", cut_concentration.real)

#マッピング用リストの作成 #計測範囲内のみマップを作成する
for num in range(cell_size):    
    #点群でボクセルを表現する
    x_point = int(((num % (Xrange*Yrange)) % Xrange)) * delta #+ xmin
    y_point = int(((num % (Xrange*Yrange)) / Xrange)) * delta -ymin
    z_point = int((num / (Xrange*Yrange))) * delta
    #print('num',num, 'x',x_point,'y',y_point,'z',z_point, 'concentration', temp_concentration_list[num])
    range_num = 5
    if int(num % (Xrange*Yrange) / Xrange) >= RangetoNum(ymin) and temp_concentration_list[num] >= cut_concentration and num < (cell_size/2):    #閾値設定 default
    #if int(num % (Xrange*Yrange) / Xrange) >= RangetoNum(ymin) and (y2[num] >= 300 or y2[num] <= -300) :    #閾値設定 Fourier変換後を見る
    #if int(num % (Xrange*Yrange) / Xrange) >= RangetoNum(ymin) and yy2[num] >= 51 :    #閾値設定 Fourier逆変換後を見る
        for numz in range(range_num):
            for numy in range(range_num):
                for numx in range(range_num):
                    x_fine_list.append(x_point+numx*float(delta/range_num))
                    y_fine_list.append(y_point+numy*float(delta/range_num))
                    z_fine_list.append(z_point+numz*float(delta/range_num))
                    concentration_list.append(temp_concentration_list[num])     #default
                    #concentration_list.append(y4[num]) #Fourier逆変換後


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
plt.figure(4)
plt.show()
#----------------------------------------------------------------------