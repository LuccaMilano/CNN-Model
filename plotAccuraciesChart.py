import numpy as np
import matplotlib.pyplot as plt

# Example data: Accuracies for split and k-fold validation
configurations = np.arange(1, 14)
split_accuracies = [81.6,75.42,79.87,79.68,78.60,80.57,68.52,79.18,80.63,77.17,54.87,77.25,55.63]
kfold_accuracies = [82.2,72.82,80.54,78.49,79.62,79.60,66.17,79.01,79.13,75.85,55.65,72.54,56.39]

#,81.6,75.42,79.87,79.68,78.60,80.57,68.52,79.18,80.63,77.17,54.87,77.25,55.63
#,82.2,72.82,80.54,78.49,79.62,79.60,66.17,79.01,79.13,75.85,55.65,72.54,56.39\\

bar_width = 0.35  # Width of the bars
bar_positions_split = configurations
bar_positions_kfold = configurations + bar_width

# Define colors for the bars
split_color = '#6a3d9a'  # color for split validation
kfold_color = '#993399'  # color purple for k-fold validation
#kfold_color = '#4cb1b1'
# Create a figure and axis
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})

plt.figure(figsize=(10, 6))
ax = plt.gca()

# Plot the data
ax.bar(bar_positions_split, split_accuracies, width=bar_width, label='Validação Split', color=split_color)
ax.bar(bar_positions_kfold, kfold_accuracies, width=bar_width, label='Validação 5-Fold', color=kfold_color)

# Set labels and title
plt.xlabel('Configuração')
plt.ylabel('Acurácia')
plt.title('Métodos de validação')
plt.xticks(configurations + bar_width/2, configurations)
plt.legend()

# Show the plot
plt.tight_layout()
plt.savefig("accuracies.pdf",bbox_inches='tight')
plt.show()
