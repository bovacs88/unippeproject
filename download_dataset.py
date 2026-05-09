from roboflow import Roboflow

rf = Roboflow(api_key="Ys40jZxToM7Uh8PrGmcP")
project = rf.workspace("man-1o9pj").project("ppe-detection-6kous")
version = project.version(2)
dataset = version.download(model_format="yolov8", location="./dataset")
