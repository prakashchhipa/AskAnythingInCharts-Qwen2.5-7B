from setuptools import setup, find_packages

setup(
    name="askanything-in-charts",
    version="1.0.0",
    description="AskAnything in Charts - Powered by Qwen 2.5",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Prakash Chandra Chhipa",
    author_email="prakashchhipa@github.io",
    url="https://github.com/prakashchhipa/AskAnythingInCharts-Qwen2.5-7B",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.45.0",
        "peft>=0.12.0",
        "Pillow>=10.0.0",
        "gradio>=4.0.0",
        "accelerate>=0.20.0",
        "datasets>=2.0.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
