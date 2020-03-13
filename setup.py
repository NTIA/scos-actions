import setuptools

# with open("README.md", "r") as fh:
#     long_description = fh.read()


setuptools.setup(
    name="scos_actions",
    version="0.0.0",
    author="The Institute for Telecommunication Sciences",
    # author_email="author@example.com",
    description="Base actions library for scos-sensor",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    #url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)