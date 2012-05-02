# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name='xport',
    version='0.1.0',
    author=u'Jack Cushman',
    author_email='jcushman@gmail.com',
    packages=find_packages(),
    url='https://github.com/jcushman/xport',
    license='MIT',
    description='SAS XPORT data file reader.',
    keywords='sas xport',
    long_description=open('README.rst').read(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Text Processing",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        ],
)