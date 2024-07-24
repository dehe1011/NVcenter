from setuptools import setup, find_packages

setup(
    name="ExampleProject_11",
    version="0.1.1",
    author="Dennis Herb",
    author_email="dennis.herb@uni-ulm.de",
    description="An example project for testing.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/dehe1011/ExampleProject",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'ExampleProject_11': ['data/*.json'],
    },
)
