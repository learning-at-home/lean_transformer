import codecs
import glob
import hashlib
import os
import re
import shlex
import subprocess
import tarfile
import tempfile
import urllib.request

from pkg_resources import parse_requirements, parse_version
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop

with open("requirements.txt") as requirements_file:
    install_requires = list(map(str, parse_requirements(requirements_file)))

# loading version from setup.py
with codecs.open(os.path.join(here, "hivemind/__init__.py"), encoding="utf-8") as init_file:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", init_file.read(), re.M)
    version_string = version_match.group(1)

extras = {}

with open("requirements-dev.txt") as dev_requirements_file:
    extras["dev"] = list(map(str, parse_requirements(dev_requirements_file)))

with open("requirements-docs.txt") as docs_requirements_file:
    extras["docs"] = list(map(str, parse_requirements(docs_requirements_file)))

extras["all"] = extras["dev"] + extras["docs"]

setup(
    name="hivemind",
    version=version_string,
    cmdclass={"build_py": BuildPy, "develop": Develop},
    description="Decentralized deep learning in PyTorch",
    long_description="Decentralized deep learning in PyTorch. Built to train models on thousands of volunteers "
    "across the world.",
    author="Learning@home & contributors",
    author_email="hivemind-team@hotmail.com",
    url="https://github.com/learning-at-home/hivemind",
    packages=find_packages(exclude=["tests"]),
    package_data={"hivemind": ["proto/*", "hivemind_cli/*"]},
    include_package_data=True,
    license="MIT",
    setup_requires=["grpcio-tools"],
    install_requires=install_requires,
    extras_require=extras,
    classifiers=[
        "Development Status :: 4 - Beta",
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
    entry_points={
        "console_scripts": [
            "hivemind-server = hivemind.hivemind_cli.run_server:main",
        ]
    },
    # What does your project relate to?
    keywords="pytorch, deep learning, machine learning, gpu, distributed computing, volunteer computing, dht",
)
