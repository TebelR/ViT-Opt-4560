import os

project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project, "data", "classification")



for label in os.listdir(data_dir):
    os.rename(os.path.join(data_dir, label), os.path.join(data_dir, label.replace(u'\xa0', " ")))