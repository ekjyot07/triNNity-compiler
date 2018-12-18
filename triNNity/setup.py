import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="triNNity",
    version="0.0.1",
    author="Andrew Anderson",
    author_email="aanderso@tcd.ie",
    description="triNNity compiler internals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://bitbucket.org/STG-TCD/triNNity-compiler",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
