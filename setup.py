import setuptools

setuptools.setup(
    name="idas-convert",
    version="0.0.1",
    author="Marius Paul Isken",
    author_email="isken@example.com",
    description="Convert iDAS TDMS data to anything",
    url="https://git.pyrocko.org/pyrocko/idas-tdms-converter",
    package_dir={
        'idas_convert': 'src'
    },
    packages=[
        'idas_convert',
    ],
    entry_points={
        'console_scripts': [
            'idas_convert = idas_convert:main',
        ]},

    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
