import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# make data
begin = 1555
final = 2324 #1554
limit = final - begin
data = pd.read_csv('./dataset')
x1 = np.linspace(1, limit, limit)
y1 = data.iloc[begin - 1:final - 1, 4].values
z1 = data.iloc[begin - 1:final - 1, 7].values

plt.plot(x1, y1, color='red', label='Cases')
plt.title('Europe COVID-19 Climate')
plt.xlabel('Time')
plt.ylabel('Cases')
plt.legend()
plt.xticks([])
plt.yticks()
plt.show()

plt.plot(x1, z1, color='grey', label='Deaths')
plt.title('Europe COVID-19 Climate')
plt.xlabel('Time')
plt.ylabel('Deaths')
plt.legend()
plt.xticks([])
plt.yticks()
plt.show()