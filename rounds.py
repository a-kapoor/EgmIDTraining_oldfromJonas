import numpy as np
import matplotlib.pyplot as plt
import sys

dirname = sys.argv[-1]

arr = np.loadtxt(dirname + "/rounds.txt").T

plt.figure(figsize=(6.4,4.8))

plt.plot(1 - arr[1], label="eval")
plt.plot(1 - arr[0], label="train")

plt.ylabel("1 - AUC")
plt.xlabel("Rounds")
plt.legend(loc="upper right")
plt.show()
