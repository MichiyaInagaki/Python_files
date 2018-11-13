from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# これをカラム名にする
col_names = [ 'c{0:02d}'.format(i) for i in range(5) ]
#csvファイルの読み込み*** names=>カラム名，header=>始まりの行指定
df1 = pd.read_csv(filepath_or_buffer="C:/Users/SENS/Documents/My_folder/research/progress_memo/2018_11_12_methane.csv", encoding="ms932", sep=",", names=col_names, header=5)
#df2 = pd.read_csv(filepath_or_buffer="C:/Users/SENS/Documents/My_folder/research/progress_memo/2018_11_10 123547_1.csv", encoding="ms932", sep=",", names=col_names, header=5)
#df2 = pd.read_csv(filepath_or_buffer="C:/Users/SENS/Documents/My_folder/research/progress_memo/2018_11_05_nomal2.csv", encoding="ms932", sep=",", names=col_names, header=5)

#データの抽出（100行までの計測値のみを取得）***
data_df1 = df1.loc[:6000,['c02']]
#data_df2 = df2.loc[:6000,['c02']]

#データの統合
#data_df12=pd.concat([data_df1,data_df2], ignore_index=True)

# データの統計量の表示
print(data_df1.describe())
#rint(data_df2.describe())

# 描画範囲の指定
# x = np.arange(x軸の最小値, x軸の最大値, 刻み)***
x = np.arange(0, 600, 0.1)
y1=data_df1
#y2=data_df2

# 横軸の変数。縦軸の変数。
plt.figure(1)
plt.plot(x, y1)
#plt.title("Graph Title")
plt.xlabel("Time[sec]")
plt.ylabel("Measured value[ppm-m]")

#plt.figure(2)
#plt.plot(x, y2)

#ヒストグラム***
plt.figure(2)
data_df1.c02.plot.hist(bins=20, color='gray', rwidth=.8)
#plt.title("Graph Title")
plt.xlabel("Measured value[ppm-m]")
#plt.ylabel("")

#横に並べて表示
#fig = plt.figure()
#ax1 = fig.add_subplot(211)  #引数はこの場合，2*1の領域の1つめの領域に書くことを示す
#ax1.plot(x, y1)
#ax2 = fig.add_subplot(212)
#ax2.plot(x, y2)

#グラフの描画
plt.show()