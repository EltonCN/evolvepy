from typing import List


README_PATH = "../README.md"

replaces = {"](examples": "](https://github.com/EltonCN/evolvepy/blob/main/examples",}

def parse_readme():
    file = open("../README.md", "r")
    lines = file.readlines()
    file.close()

    new_file = open("README.md", "w")


    for line in lines:
        for key in replaces:
            line = line.replace(key, replaces[key])
        new_file.write(line)
    
    new_file.close()

if __name__ == "__main__":
    parse_readme()
