from setuptools import setup, find_packages

setup(
    name="nerf_pytorch_truedepth",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "imageio",
        "matplotlib",
        "tqdm",
        "configargparse",
        "wandb",
        # torch 单独说明，见下面
        # "torch==2.3.0",  # 如果你确认两个平台都用同一大版本，可以写死
    ],
)
