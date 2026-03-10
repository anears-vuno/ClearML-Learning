from clearml import Task
import socket
import time


task = Task.init(
    project_name="remote_execution_test",
    task_name="remote_execution2"
)

task.execute_remotely(queue_name="default")

print("Running on host:", socket.gethostname())

for i in range(10):
    print("step", i)
    time.sleep(1)