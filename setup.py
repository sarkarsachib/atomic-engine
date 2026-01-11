#!/usr/bin/env python3
"""Setup script for Manus LLM Agent Layer"""

from setuptools import setup, find_packages

with open("app/requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="manus-llm",
    version="1.0.0",
    description="Manus LLM Agent Layer for Atomic Engine",
    author="Atomic Engine",
    license="MIT",
    packages=find_packages(where="app", include=["app.llm", "app.llm.*"]),
    package_dir={"": "app"},
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "pytest-asyncio", "mypy"],
    },
    entry_points={
        "console_scripts": [
            "manus-llm=app.llm.client:main",
        ],
    },
    include_package_data=True,
    package_data={
        "app.llm": ["py.typed"],
    },
)
