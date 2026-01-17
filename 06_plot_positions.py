import matplotlib.pyplot as plt

# positions
positions = [1, 2, 3, 4, 5]

# example metric values (replace with yours)
accuracy  = [0.91, 0.91, 0.91, 0.91, 0.91]
recall    = [0.94, 0.96, 0.95, 0.93, 0.93]
precision = [0.80, 0.82, 0.79, 0.82, 0.79]

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
