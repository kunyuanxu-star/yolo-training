from setuptools import setup, find_packages

setup(
    name="training_engine",
    version="0.1.0",
    packages=find_packages(exclude=["frontend*", "scripts*", "docs*", "tests*"]),
    python_requires=">=3.10",
)
