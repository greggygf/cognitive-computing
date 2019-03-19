# untuk operasi matriks
import numpy as np

# untuk visualisasi
import matplotlib.pyplot as plt

# untuk menghitung akurasi score dari perhitungan manual
from sklearn.metrics import accuracy_score

# Inisialisasi bias, bobot, perulangan, learning rate
bias = np.random.random_sample()
weight = np.random.random_sample((4,)).astype(float)

perulangan = 1000
learning_rate = 0.8

# Tugas 4

# learning_rate = 0.1

# mengatur path sesuai dengan path csv
path = str(
    '../')
data_training = np.genfromtxt(
    path+'IRIS-edit-training.csv', delimiter=',', dtype='unicode')
data_testing = np.genfromtxt(
    path+'IRIS-edit-testing.csv', delimiter=',', dtype='unicode')

# preprocess data iris
# mengubah datairis yang berlabel Iris-versicolor menjadi 1
# mengubah datairis yang berlabel Setosa menjadi 0


def Preprocess(dataIris):
    for singleData in dataIris:
        if (singleData[4] == 'Iris-versicolor'):
            singleData[4] = 1
        else:
            singleData[4] = 0
    return dataIris

# Fungsi aktivasi dengan hard limit


def FungsiAktivasi(y):
    if (y > 0):
        return 1
    elif(y <= 0):
        return 0

# Tugas 2 - Fungsi Aktivasi dengan symetric hard limit

# def FungsiAktivasi(y):
#     if (y > 0):
#         return 1
#     elif(y == 0):
#         return 0
#     else:
#         return -1


# data training dan testing diubah kedalam bentuk float
data_training = Preprocess(data_training).astype(float)
data_testing = Preprocess(data_testing).astype(float)

# training
for i in range(perulangan):
    for data in data_training:
        # print(type (data[0]))
        y = bias + np.dot(data[0:4], weight)
        y_prediksi = FungsiAktivasi(y)
        error = data[4] - y_prediksi
        if (error != 0):
            weight[0] = weight[0] + (learning_rate*error*data[0])
            weight[1] = weight[1] + (learning_rate*error*data[1])
            weight[2] = weight[2] + (learning_rate*error*data[2])
            weight[3] = weight[3] + (learning_rate*error*data[3])
            # boleh diupdate atau tidak berdasarkan sumber
            bias = bias + learning_rate*error

# # testing
y_predict = []
for data in data_testing:
    y = bias + np.dot(data[0:4], weight)
    y_prediksi = FungsiAktivasi(y)
    y_predict.append(y_prediksi)

print(accuracy_score(data_testing[:, 4], np.array(y_predict)))


# tugas1

print("Weight 1 : " + str(weight[0]))
print("Weight 2 : " + str(weight[1]))
print("Weight 3 : " + str(weight[2]))
print("Weight 4 : " + str(weight[3]))
