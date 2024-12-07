from clearml import Task, StorageManager
from loguru import logger

# create an dataset experiment
task = Task.init(project_name="cs-gy-6613-rag", task_name="Pipeline Task 1")

# only create the task, we will actually execute it later
# task.execute_remotely()

# # simulate local dataset, download one, so we have something local
# local_iris_pkl = StorageManager.get_local_copy(
#     remote_url="https://github.com/allegroai/events/raw/master/odsc20-east/generic/iris_dataset.pkl"
# )

# # add and upload local file containing our toy dataset
# task.upload_artifact("dataset", artifact_object=local_iris_pkl)

# print("uploading artifacts in the background")

# # we are done
# print("Done")


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
