from setuptools import setup
from Cython.Build import cythonize

print("Starting")

setup(
    name='Hello world app',
    ext_modules=cythonize("runnerQueueSplit27.pyx"),
)

print("Done")