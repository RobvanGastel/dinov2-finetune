from setuptools import setup, find_packages

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()[1:]

setup(
    name="dino_finetune",
    version="0.1.1",
    packages=find_packages(),
    url="https://github.com/robvangastel/dinov2-finetune",
    license="",
    install_requires=install_requires,
    extras_require={
        "dev": [
            "black",
        ]
    },
    author="Rob van Gastel",
    description="A package for finetuning DINOv2, DINOv3 models with Low Rank Adaptation (LoRA).",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
