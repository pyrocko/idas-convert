import setuptools

setuptools.setup(
    name="das-convert",
    version="0.1.0",
    author="Marius Paul Isken",
    author_email="mi@gfz-potsdam.de",
    description="Convert seismic DAS data to anything",
    url="https://git.pyrocko.org/pyrocko/idas-convert",
    package_dir={"das_convert": "src"},
    packages=[
        "das_convert",
    ],
    entry_points={
        "console_scripts": [
            "das_convert = das_convert.app:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
