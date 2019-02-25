import matplotlib.pyplot as plt
import numpy as np
# membuat list angka berjarak dalam range yang ditentukan
x = np.linspace(0, 20, 100)
plt.plot(x, np.sin(x))  # Plot sinusoida dari tiap nilai
plt.show()              # tampilkan plot
