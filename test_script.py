import platform
import os
import sys

print("Python version:", sys.version)
print("Platform:", platform.platform())
print("Current directory:", os.getcwd())
print("Content of current directory:", os.listdir("."))
print("Hello from HPCC!") 