import os

project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project, "data", "classification")



for label in os.listdir(data_dir):
    if(label.__contains__(u'\xa0')):
        os.rename(os.path.join(data_dir, label), os.path.join(data_dir, label.replace(u'\xa0', "_")))
    else:
        os.rename(os.path.join(data_dir, label), os.path.join(data_dir, label.replace(" ", "_")))