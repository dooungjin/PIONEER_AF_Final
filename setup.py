import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pioneer",
    version="0.0.1",
    author="Dongjin Lee, Charles Liang",
    author_email="dl953@cornell.edu, sl2678@cornell.edu",
    description="PIONEER: protein-protein interaction interface prediction with deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hyulab/PIONEER",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True,
    package_data={'': ['data/*.*']},
)
