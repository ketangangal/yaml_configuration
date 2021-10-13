import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="src",
    version="0.0.1",
    author="BlackPanther",
    author_email="ketangangal@gmail.com",
    description="A small example package Ann Implementations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ketangangal/yaml_configuration.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=["src"],
    python_requires=">=3.6",
    install_requires=["tensorflow",
                      "matplotlib",
                      "seaborn",
                      "numpy",
                      "pandas"
                      ]

)