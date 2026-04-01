from setuptools import setup, find_packages

setup(
    name="privacy-mcps",
    version="1.0.0",
    description="Privacy-Preserving Microservices ML Framework for Medical Cyber-Physical Systems",
    author="Author Name",
    author_email="author@university.edu",
    url="https://github.com/<username>/privacy-mcps",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "tensorflow>=2.12.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "phe>=1.5.0",
        "imbalanced-learn>=0.11.0",
        "scipy>=1.11.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security :: Cryptography",
    ],
)
