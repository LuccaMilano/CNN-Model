import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})

# Example data: Accuracies for split and k-fold validation
configurations = np.arange(1, 14)
# split_accuracies = [0.8347, 0.77, 0.8166, 0.8189, 0.8124, 0.8012, 0.7, 0.8152, 0.82, 0.8017, 0.60, 0.7583, 0.5534]
# kfold_accuracies = [0.85, 0.7866, 0.8246, 0.8377, 0.8393, 0.8285, 0.7013, 0.8261, 0.8299, 0.815, 0.6082, 0.7789, 0.5673]

split_accuracies = [83, 77, 81, 81, 81, 80, 70, 81, 82, 80, 60, 75, 55]
kfold_accuracies = [85, 78, 82, 83, 84, 82, 70, 82, 83, 81, 60, 77, 56]

bar_width = 0.35  # Width of the bars
bar_positions_split = configurations
bar_positions_kfold = configurations + bar_width

# Define colors for the bars
split_color = '#6a3d9a'  # color for split validation
kfold_color = '#993399'  # color purple for k-fold validation

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})

# Create a figure and axis
plt.figure(figsize=(10, 6))
ax = plt.gca()

# Plot the data
split_bars = ax.bar(bar_positions_split, split_accuracies, width=bar_width, label='Validação Split', color=split_color)
kfold_bars = ax.bar(bar_positions_kfold, kfold_accuracies, width=bar_width, label='Validação 5-Fold', color=kfold_color)

# Annotate the bars with their values
def annotate_bars(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',  # Format the number to 4 decimal places
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords='offset points',
                    ha='center', va='bottom')

annotate_bars(split_bars)
annotate_bars(kfold_bars)

# Set labels and title
plt.xlabel('Configuração')
plt.ylabel('Acurácia')
plt.title('Métodos de validação')
plt.xticks(configurations + bar_width / 2, configurations)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()
