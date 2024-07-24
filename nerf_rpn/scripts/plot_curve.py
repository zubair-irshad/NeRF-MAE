import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

# Data
labels = ['10%', '25%', '50%', '100%']
pretrained_encoder = ['MAE', 'MAE']
ap_50 = [[0.175, 0.36, 0.42, 0.54], [0.152, 0.29, 0.303, 0.41]]

# Set LaTeX font
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Plot style
plt.style.use('seaborn-whitegrid')
plt.style.use('viridis')

# Plot setup
fig, ax = plt.subplots()
fig.patch.set_facecolor('white')  # Set the background color to white

# Plot lines and markers
ax.plot(labels, ap_50[0], marker='o', markersize=8, linewidth=2, color='tab:blue', linestyle='-', label='MAE pretrained encoder')
ax.plot(labels, ap_50[1], marker='o', markersize=8, linewidth=2, color='tab:orange', linestyle='--', label='MAE pretrained encoder (start from scratch)')

# Set line width for better visibility
ax.spines['left'].set_linewidth(2.5)
ax.spines['right'].set_linewidth(2.5)
ax.spines['top'].set_linewidth(2.5)
ax.spines['bottom'].set_linewidth(2.5)

# Draw dashed arrow and annotate improvement
arrow_style = dict(arrowstyle='->', color='black', linestyle='--', linewidth=1.5)
# ax.annotate('13\% absolute\nimprovement', xy=(labels[3], ap_50[1][3]), xytext=(labels[3], ap_50[0][3]), ha='center', fontsize=10,
#               arrowprops=arrow_style)

print(labels[3], ap_50[1][3])
ax.annotate('13\% absolute\nimprovement', xy=(labels[3], ap_50[1][3]), xytext=(labels[3], ap_50[0][3]), ha='center', fontsize=10,
              arrowprops=arrow_style)

# Axis labels and title (LaTeX rendered)
ax.set_xlabel(r'Percentage of labelled scenes', fontsize=12, fontweight='bold')
ax.set_ylabel(r'Average precision (AP 50)', fontsize=12, fontweight='bold')
ax.set_title(r'\textbf{NeRF-MAE} outperforms the SoTA on NeRF 3D Object Detection', fontsize=14, fontweight='bold')

# Legend
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='lower right')

# Show the plot
plt.show()