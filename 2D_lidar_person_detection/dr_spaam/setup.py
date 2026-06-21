from setuptools import setup, find_packages

setup(
    name="dr_spaam",
    version="1.2.0",
    author="Pavan Sai",
    author_email="18pavansai@gmail.com",
    packages=find_packages(include=["dr_spaam", "dr_spaam.*", "dr_spaam.*.*"]),
    license="LICENSE.txt",
    description="DR-SPAAM, a deep-learning based person detector for 2D range data.",
)
