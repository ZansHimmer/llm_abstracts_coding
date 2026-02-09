import matplotlib.pyplot as plt

# positions
positions = [1, 2, 3, 4, 5]

# example metric values (replace with yours)
accuracy  = [0.90, 0.88, 0.88, 0.86, 0.87]
recall    = [0.90, 0.91, 0.90, 0.89, 0.90]
precision = [0.81, 0.77, 0.75, 0.73, 0.72]

fig = plt.gcf()
ax = plt.gca()

fig.patch.set_alpha(0)
ax.set_facecolor('none') 

plt.figure()

plt.plot(positions, accuracy,  marker='o', label='Accuracy')
plt.plot(positions, recall,    marker='s', label='Recall')
plt.plot(positions, precision, marker='^', label='Precision')

plt.xlabel('Position')
plt.ylabel('Score')
plt.xticks(positions)
plt.ylim(0, 1)

plt.legend()
plt.grid(False)

plt.show()
