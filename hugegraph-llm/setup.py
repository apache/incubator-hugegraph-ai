import setuptools
from pkg_resources import parse_requirements

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", encoding="utf-8") as fp:
    install_requires = [str(requirement) for requirement in parse_requirements(fp)]

setuptools.setup(
    name="hugegraph-llm",
    version="1.3.0",
    author="Apache HugeGraph Team",
    author_email="dev@hugegraph.apache.org",
    install_requires=install_requires,
    include_package_data=True,
    description="Integrating Apache HugeGraph with LLM.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/incubator-hugegraph-ai",
    packages=setuptools.find_packages(where="src", exclude=["tests"]),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
