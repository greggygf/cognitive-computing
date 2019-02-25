# untuk operasi matriks
import numpy as np
# untuk melakukan visualisasi
import matplotlib.pyplot as plt

bias = 1
weight = np.array([1, 1])

perulangan = 100
learning_rate = 0.5
errorArray = []

data_training = np.array([
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1)
])
data_testing = np.array([1, 1, 1])


def FungsiAktivasi(y):
    if (y > 0):
        return 1
    elif(y == 0):
        return 0
    else:
        return -1


# training
for i in range(perulangan):
    for data in data_training:
        y = bias+(data[0] * weight[0]) + (data[1]*weight[1])
        y_prediksi = FungsiAktivasi(y)
        error = data[2] - y_prediksi
        errorArray.append(error)
        if (error != 0):
            # print (error)
            weight[0] += (learning_rate*error*data[0])
            weight[1] += (learning_rate*error*data[1])
            bias += learning_rate*error

print("model = " + str(weight[0]) + ";"+str(weight[1])+";")

plt.ylim([-1, 1])
plt.plot(errorArray)
plt.show()

# testing
for data in data_testing:
    y = bias+(data[0]*weight[0]) + (data[1]*weight[0])
    y_prediksi = FungsiAktivasi(y)
    error = data[2] - y_prediksi

    print('y_prediksi = '+str(y_prediksi))
    print('error = '+str(error))
