from setuptools import setup, find_packages

setup(
    name="expl_annotator",
    version="0.1.0",
    description="Explanation Annotator used to create a KG from provenance execution traces",
    author="User",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "rdflib",
        "pyyaml",
    ],
    python_requires=">=3.6",
)
