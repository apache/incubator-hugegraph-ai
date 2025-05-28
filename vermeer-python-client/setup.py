# !/usr/bin/env python3
"""
file:setup.py
author: wenyuxuan@baidu.com
"""
import setuptools
from pkg_resources import parse_requirements

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", encoding="utf-8") as fp:
    install_requires = [str(requirement) for requirement in parse_requirements(fp)]

setuptools.setup(
    name="vermeer-python",
    version="0.1.0",
    install_requires=install_requires,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(where="src", exclude=["tests"]),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
