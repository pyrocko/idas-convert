import setuptools

setuptools.setup(
    name="idas-convert",
    version="0.1.0",
    author="Marius Paul Isken",
    author_email="mi@gfz-potsdam.de",
    description="Convert iDAS TDMS data to anything",
    url="https://git.pyrocko.org/pyrocko/idas-convert",
    package_dir={
        'idas_convert': 'src'
    },
    packages=[
        'idas_convert',
    ],
    entry_points={
        'console_scripts': [
            'idas_convert = idas_convert.app:main',
        ]},

    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
