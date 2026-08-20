from setuptools import setup, find_packages

setup(
    name="phm_windturbine",
    version="0.1.0",
    description="Sparse sensing and physics-informed learning for wind turbine structural health monitoring",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "torch>=2.0.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
)
