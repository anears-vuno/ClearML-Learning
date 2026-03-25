# ct_controller.py
from clearml import Task, Dataset

PROJECT = "MNIST_CT"
DATASET_PROJECT = "MNIST_CT"
DATASET_NAME = "mnist_dataset"
QUEUE = "default"

# 베이스 학습 task id를 넣으세요.
BASE_TASK_ID = "29e64976081640d0bfe9b6e3015b1ab7"

def main():
    controller = Task.init(project_name=PROJECT, task_name="mnist_ct_controller")

    # 최신 dataset version 가져오기
    latest_ds = Dataset.get(
        dataset_project=DATASET_PROJECT,
        dataset_name=DATASET_NAME,
    )

    latest_dataset_id = latest_ds.id

    # controller가 마지막으로 처리한 dataset_id 기억
    state = {"last_trained_dataset_id": ""}
    state = controller.connect(state, name="ct_state")

    if state["last_trained_dataset_id"] == latest_dataset_id:
        print("새 dataset version 없음. 아무 작업도 하지 않음.")
        return

    # base task clone
    cloned = Task.clone(
        source_task=BASE_TASK_ID,
        name=f"mnist_train_on_{latest_dataset_id[:8]}",
        comment=f"Triggered by dataset version {latest_dataset_id}",
    )

    cloned_task = Task.get_task(task_id=cloned.id)
    params = cloned_task.get_parameters()

    # train.py의 task.connect(params) 기준으로 General 섹션에 들어갑니다.
    params["General/dataset_id"] = latest_dataset_id
    cloned_task.set_parameters(params)
    cloned_task.add_tags(["ct", "dataset-triggered", "mnist"])

    Task.enqueue(cloned_task, queue_name=QUEUE)
    print(f"Enqueued task {cloned_task.id} with dataset {latest_dataset_id}")

    state["last_trained_dataset_id"] = latest_dataset_id
    controller.connect(state, name="ct_state")

if __name__ == "__main__":
    main()