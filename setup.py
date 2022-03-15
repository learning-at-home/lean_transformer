import codecs
import os
import re

from pkg_resources import parse_requirements, parse_version
from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

with open("requirements.txt") as requirements_file:
    install_requires = list(map(str, parse_requirements(requirements_file)))

# loading version from setup.py
with codecs.open(os.path.join(here, "lean_transformer/__init__.py"), encoding="utf-8") as init_file:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", init_file.read(), re.M)
    version_string = version_match.group(1)

extras = {}

with open("requirements-dev.txt") as dev_requirements_file:
    extras["dev"] = list(map(str, parse_requirements(dev_requirements_file)))

with open("requirements-docs.txt") as docs_requirements_file:
    extras["docs"] = list(map(str, parse_requirements(docs_requirements_file)))

extras["all"] = extras["dev"] + extras["docs"]

setup(
    name="lean_transformer",
    version=version_string,
    description="PyTorch transformers that don't hog your GPU memory.",
    long_description="Memory-efficient transformers with optional sparsity, reversible layers, checkpoints, etc.",
    author="Learning@home & contributors",
    author_email="hivemind-team@hotmail.com",
    url="https://github.com/learning-at-home/lean_transformer",
    packages=find_packages(exclude=["tests"]),
    license="MIT",
    install_requires=install_requires,
    extras_require=extras,
    classifiers=[
        "Development Status :: 4 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="pytorch, deep learning, machine learning, gpu, efficient training, efficient inference",
)
