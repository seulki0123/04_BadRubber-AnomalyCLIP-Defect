from roboflow import Roboflow
rf = Roboflow(api_key="b78VK50s6AszQw6gstvg")
project = rf.workspace("infrax-zc1kh").project("f2150_1classtest")
version = project.version(1)
dataset = version.download("yolov11")