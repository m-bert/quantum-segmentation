import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Get absolute path to the CSV file
csv_path = (Path(__file__) / 'results_lanczos.csv').resolve()
data = np.genfromtxt(str(csv_path), delimiter=';', dtype=float)

N = data[:, 0]
graph_time = data[:, 1]
krylov_time = data[:, 2]


fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Krylov reconstruction time plot
axs[0].plot(N, krylov_time, 'o-', color='tab:blue')
axs[0].set_xlabel('Image size (N)')
axs[0].set_ylabel('Time (seconds)')
axs[0].set_title('Krylov reconstruction time')
axs[0].grid(True)

# Graph generation time plot
axs[1].plot(N, graph_time, 's-', color='tab:orange')
axs[1].set_xlabel('Image size (N)')
axs[1].set_ylabel('Time (seconds)')
axs[1].set_title('Graph generation time')
axs[1].grid(True)

fig.tight_layout()
plt.show()