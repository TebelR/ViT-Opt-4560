import os

#This script needs to be inside one of the labels folders, it will update every line in every label file to have the class set to 0

os.chdir("data/syntheticData/labels/test")
for filename in os.listdir("."):
    if filename.endswith(".txt"):
        with open(filename, "r") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                tokens = lines[i].split(" ")
                tokens[0] = "0"
                lines[i] = " ".join(tokens)
            print("updated " + filename)
        with open(filename, "w") as f:
            f.writelines(lines)