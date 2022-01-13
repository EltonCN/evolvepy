from typing import List


FILES_PATHS = ["../README.md", "../examples/Examples.md"]

replaces = {"](/examples": "](https://github.com/EltonCN/evolvepy/blob/main/examples",
            "](/Thumbnail.png":"](https://github.com/EltonCN/evolvepy/blob/main/Thumbnail.png"}

def parse_files():
    for path in FILES_PATHS:

        file = open(path, "r")
        lines = file.readlines()
        file.close()

        file_name = path.split("/")[-1]

        print(file_name)

        new_file = open(file_name, "w")


        for line in lines:
            for key in replaces:
                line = line.replace(key, replaces[key])
            new_file.write(line)
        
        new_file.close()
