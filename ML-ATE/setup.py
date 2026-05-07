from setuptools import setup, find_packages
import version

fixed_requirements = ["py4j"]

with open('requirements.txt') as requirements:
    added_requirements = [line.rstrip() for line in requirements]

fixed_requirements += added_requirements

with open("README.md", "r", encoding="utf-8") as readme:
    long_description = readme.read()

setup(
    name="causal-ai-ate",
    version=version.__version__,
    description=version.__description__,
    install_requires=fixed_requirements,
    long_description=long_description,
    packages=find_packages(exclude=['tests', 'docs']),
    python_requires=">=3.10",
    classifiers=['Programming Language :: Python :: 3.10',],
    include_package_data=True,
)
