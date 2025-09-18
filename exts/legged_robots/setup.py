"""Installation script for the 'legged_robots' python package."""

import os
from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))

# Lazy import toml to avoid ImportError before installation
def load_extension_data():
    import toml
    return toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

EXTENSION_TOML_DATA = load_extension_data()

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "psutil",
    "toml",          # required to parse extension.toml
    "tensordict",    # PyTorch TensorDict library
    "torchrl",       # Reinforcement Learning extensions for PyTorch
]

# Installation operation
setup(
    name="legged_robots",
    packages=["legged_robots"],
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="MIT",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 4.5.0",
    ],
    zip_safe=False,
)
