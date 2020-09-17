import os

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


repo_root = os.path.dirname(os.path.realpath(__file__))
requirements_path = repo_root + "/requirements.txt"
install_requires = []  # Examples: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if os.path.isfile(requirements_path):
    with open(requirements_path) as f:
        install_requires = f.read().splitlines()

setuptools.setup(
    name="scos_actions",
    version="0.0.0",
    author="The Institute for Telecommunication Sciences",
    # author_email="author@example.com",
    description="Base actions and hardware support library for scos-sensor",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NTIA/scos-actions",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Public Domain",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=install_requires,
    package_data={"scos_actions": ["configs/actions/*.yml"]},
)
