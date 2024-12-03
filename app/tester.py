from clearml import Task
from loguru import logger

task = Task.init(project_name="cs-gy-6613-rag", task_name="tester")

logger.info("Hello, ClearML!")

# Generate a random plot
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title("Random Plot")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid()
plt.show()


logger.info("Goodbye, ClearML!")
