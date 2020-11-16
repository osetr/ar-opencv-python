import os
from natsort import natsorted

path_to_directory = input("Enter path to directory: ") + "/"
new_name = input("Enter new name for files: ")

try:
    i = 0
    list_of_files = natsorted(os.listdir(path_to_directory))
    for file in list_of_files:
        i += 1
        extension = file.split(".")[1]
        os.rename(
            path_to_directory + file,
            path_to_directory + new_name + str(i) + "." + extension,
        )
except FileNotFoundError:
    print("Got unccorect directory path")
