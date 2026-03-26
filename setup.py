from setuptools import setup

setup(
    name="amaf",
    version="1.0.0",
    description="Audio Multi-Method Assessment Fusion",
    py_modules=["web", "measure", "report", "generate_reference"],
    install_requires=[
        "numpy",
        "scipy",
        "soundfile",
        "pesq",
        "matplotlib",
        "flask",
    ],
    entry_points={
        "console_scripts": [
            "amaf=web:main",
        ],
    },
)
