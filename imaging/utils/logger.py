import os

class Logger:
    def __init__(self, filename: str):
        if os.path.isfile(filename):
            self.f = open(filename, "a")
        else:
            self.f = open(filename, "w+")
    def log(self, string):
        self.f.write(string)
        self.f.write("\n")
        self.f.flush()
    def close(self):
        self.f.close()
