import matplotlib.pyplot as plt


with open('resnet_eval.txt', 'r') as file:
    lines = file.readlines()

accuracy = []
val_accuracy = []
train_loss = []
val_los = []

lines.pop(0)
for line in range(50):
  accuracy.append(float(lines[0]))
  lines.pop(0)
lines.pop(0)
lines.pop(0)
for line in range(50):
  val_accuracy.append(lines[0])
  lines.pop(0)
lines.pop(0)
lines.pop(0)
for line in range(50):
  train_loss.append(float(lines[0]))
  lines.pop(0)
lines.pop(0)
lines.pop(0)
for line in range(50):
  val_los.append(lines[0])
  lines.pop(0)


x = range(1, len(accuracy) + 1)

plt.plot(x, accuracy, label='Accuracy')
plt.plot(x, train_loss, label='Train Loss')

# Set the axis labels and title
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Accuracy and Train Loss')

# Add a legend
plt.legend()

# Display the chart
plt.show()