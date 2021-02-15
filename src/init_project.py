import os

paths = [
    "./data/parsed/train",
    "./data/parsed/validation",
    "./data/models",
    "./data/games",
    "./data/experiments",
]


if __name__ == "__main__":
    for path in paths:
        try:
            os.makedirs(path)
            print("Path Created: " + path)
        except FileExistsError:
            print("Path Exists:" + path)
