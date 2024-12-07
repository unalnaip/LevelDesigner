from setuptools import setup, find_packages

setup(
    name="level_designer",
    version="0.1.0",
    description="AI-powered level design using conditional VAE",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "tensorboard>=2.9.0",
        "numpy>=1.19.0",
        "pytest>=7.0.0",
        "setuptools>=61.0.0",
    ],
    python_requires=">=3.7",
) 