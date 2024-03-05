import matplotlib.pyplot as plt

# data is from the result in part2.py of this project
metrics = ['Macro avg Precision', 'Macro avg Recall', 'Macro avg F1', 'Accuracy']
values = [0.961, 0.962, 0.961, 0.962]
colors =['#D89C7A', '#CFC3A9', '#D4BAAD', '#8A95A9']

# draw the bar
plt.figure(figsize=(8, 6))
plt.bar(metrics, values, color=colors,width=0.3)

# information showed near x and y
plt.xlabel('Performance metric')
plt.ylabel('Values')
plt.title('Performance of this Model on test set')
plt.ylim(0.93, 0.97)  # range of y
plt.gca().axes.get_xaxis().set_ticklabels([])  # remove x axis labels
plt.grid(axis='x', linestyle='--', alpha=0.5)  # grid line

# legend
legend_bars = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
plt.legend(legend_bars, metrics, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)

# show the number on each bar
for index, value in enumerate(values):
    plt.text(index, value + 0.001, f"{value:.3f}", ha='center', color=colors[index])

plt.tight_layout()
plt.show()
