import matplotlib.pyplot as plt

# data
n_values = [500, 700, 1000, 1200, 1300, 1500, 2000]
precision_values = [0.942, 0.96, 0.961, 0.952, 0.954, 0.957, 0.956]
recall_values = [0.942, 0.96, 0.962, 0.954, 0.956, 0.957, 0.955]
f1_score_values = [0.942, 0.96, 0.961, 0.953, 0.955, 0.957, 0.955]
accuracy_values = [0.944, 0.96, 0.962, 0.953, 0.955, 0.957, 0.955]

# window
fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

# titles
axs[0, 0].set_title('Macro averaged precision')
axs[0, 1].set_title('Macro averaged recall')
axs[1, 0].set_title('Macro averaged F1-Score')
axs[1, 1].set_title('Accuracy')

# lines
axs[0, 0].plot(n_values, precision_values, color="#D89C7A", marker='o', linestyle='-')
axs[0, 1].plot(n_values, recall_values, color="#99857E", marker='s', linestyle='-')
axs[1, 0].plot(n_values, f1_score_values, color="#849B91", marker='^', linestyle='-')
axs[1, 1].plot(n_values, accuracy_values, color="#8A95A9", marker='d', linestyle='-')

# coordinate axis
for ax in axs.flat:
    ax.set(xlabel='Maximum Words in frequency dictionary (n)', ylabel='values')
    ax.grid(linestyle='--', alpha=0.5)

plt.tight_layout()

plt.show()