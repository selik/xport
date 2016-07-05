#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open('README.rst') as f:
    readme = f.read()

setup(
    name='xport',
    version='0.3.4',

    author='Michael Selik',
    author_email='michael.selik@gmail.com',
    url='https://github.com/selik/xport',

    description='SAS XPORT file reader',
    long_description=readme,
    keywords='sas xport xpt',

    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Text Processing",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        ],

    license='MIT',
    
    py_modules=['xport'],
    #test_suite='test_xport',
    )