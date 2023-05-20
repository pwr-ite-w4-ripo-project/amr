import os

amr_dataset = "amr_dataset/labels"
target = "dataset/labels.txt"

labels = os.listdir(amr_dataset)

target_file = open(target, "+w")

for label in labels:
    file = open(f"{amr_dataset}/{label}")
    filename = label.replace("txt", "png")
    lines = file.readlines()
    
    for line in lines:
        content = line.strip()
        target_file.write(f"{filename} {content}\n")
   
    file.close()

target_file.close()