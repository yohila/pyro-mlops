
# %%
from ultralytics import YOLO
import ultralytics
import os, shutil
import yaml
import mlflow

# %% [markdown]
# Load configuration yaml files (data and model)

# %%
# Load model configuration yaml file
with open(r"model_configuration.yaml") as f:
    yolo_params = yaml.safe_load(f)

# %%
# Load data configuration yaml file
with open(r"data_configuration.yaml") as f:
    data_params = yaml.safe_load(f)

# Check label names
data_params["names"]

# %%
# Print chosen yolo parameters:
print("YOLOv8 PARAMETERS:")
print(f"model: {yolo_params['model_type']}")
print(f"imgsz: {yolo_params['imgsz']}")
print(f"lr0: {yolo_params['learning_rate']}")
print(f"batch: {yolo_params['batch']}")
print(f"name: {yolo_params['experiment_name']}")

# %%
# Define the YOLO model
model = YOLO(yolo_params['model_type'])

# %%
EXPERIMENT_NAME = "pyronear"
# mlflow.set_tracking_uri('http://localhost')
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
mlflow.set_experiment(EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
print("experiment_id:", experiment.experiment_id)


# %%
dirpath = os.path.join('./runs/detect/', yolo_params['experiment_name'] )
if os.path.exists(dirpath) and os.path.isdir(dirpath):
    shutil.rmtree(dirpath)

# %%

dirpath = os.path.join('./runs/detect/', yolo_params['experiment_name'] )
if os.path.exists(dirpath) and os.path.isdir(dirpath):
    shutil.rmtree(dirpath)

with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="pyronear_yolo") as dl_model_tracking_run:
    model.train(data="data_configuration.yaml",
        imgsz=yolo_params['imgsz'],
        batch=yolo_params['batch'],
        epochs=yolo_params['epochs'],
        optimizer=yolo_params['optimizer'],
        lr0=yolo_params['learning_rate'],
        pretrained=yolo_params['pretrained'],
        name=yolo_params['experiment_name'],
        seed=0)

    model.val()


# %%
path = f"./runs/detect/{yolo_params['experiment_name']}"

# %%
run_id = dl_model_tracking_run.info.run_id
print("run_id: {}; lifecycle_stage: {}".format(run_id,mlflow.get_run(run_id).info.lifecycle_stage))

# %%
logged_model = f'runs:/{run_id}/model'

# %%
# register the model
model_registry_version = mlflow.register_model(logged_model, 'pyronear_dl_model')
print(f'Model Name: {model_registry_version.name}')
print(f'Model Version: {model_registry_version.version}')