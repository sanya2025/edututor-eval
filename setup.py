from setuptools import setup, find_packages

setup(
    name="edututor-eval",
    version="0.1.0",
    description="Evaluation framework for AI tutoring responses",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pydantic>=2.5",
        "pyyaml>=6.0",
        "pandas>=2.1",
        "numpy>=1.24",
        "scikit-learn>=1.3",
        "scipy>=1.11",
    ],
    extras_require={
        "llm": ["openai>=1.6", "anthropic>=0.18"],
        "ml": ["torch>=2.1", "transformers>=4.36"],
        "dev": ["pytest>=7.4", "ruff>=0.1"],
    },
)
