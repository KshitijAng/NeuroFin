import os
import random
import shutil

# 1. Extract classes from the dataset
split_size = .80
categories = []

source_folder = "/Users/kshitija/Desktop/AI/transfer/Fish_Dataset/Fish_Dataset"
folders = os.listdir(source_folder)
# print(folders)

for subfolder in folders:
    if os.path.isdir(source_folder + "/" + subfolder):
        categories.append(subfolder)

categories.sort()
print(categories)

# 2. Create a Target Folder
target_folder =  "/Users/kshitija/Desktop/AI/transfer/Fish_Dataset/dataset_for_model"
existDataSetPath = os.path.isdir(target_folder)

if existDataSetPath == False:
    os.mkdir(target_folder)

# 3. Create a Function to split data between Train and Validation
def split_data(source,training,validation,split_size):
    files = []
    
    for filename in os.listdir(source):
        file = os.path.join(source, filename)
        print(file)
        
        if os.path.getsize(file) > 0:
            files.append(file)
        else:
            print(filename + " is 0 length, ignore it.")
    
    print(len(files))
    
    trainingLength = int(len(files) * split_size)
    shuffleSet = random.sample(files,len(files))
    trainingSet = shuffleSet[0:trainingLength]
    validationSet = shuffleSet[trainingLength:]
    
    # Copy the Train images
    for filename in trainingSet:
        thisFile = os.path.join(source, os.path.basename(filename))
        destination = os.path.join(training, os.path.basename(filename))
        shutil.copyfile(thisFile,destination)
        
    # Copy the Validation images
    for filename in validationSet:
        thisFile = os.path.join(source, os.path.basename(filename))
        destination = os.path.join(validation, os.path.basename(filename))
        shutil.copyfile(thisFile,destination)
        
trainPath = target_folder + "/train"
validatePath = target_folder + "/validate"

# Create Target Folders
existDataSetPath = os.path.exists(trainPath)
if existDataSetPath == False:
    os.mkdir(trainPath)
    
existDataSetPath = os.path.exists(validatePath)
if existDataSetPath == False:
    os.mkdir(validatePath)
    
# Run the function
for category in categories:
    trainDestPath = trainPath + "/" + category 
    validDestPath = validatePath + "/" + category
    
    if os.path.exists(trainDestPath) == False:
        os.mkdir(trainDestPath)
    if os.path.exists(validDestPath) == False:
        os.mkdir(validDestPath)
        
    sourcePath = source_folder + "/" + category + "/"
    trainDestPath = trainDestPath + "/"
    validDestPath = validDestPath + "/"
    
    print("Copy from " + sourcePath + " to " + trainDestPath + " and " + validDestPath)
    
    split_data(sourcePath,trainDestPath,validDestPath,split_size)