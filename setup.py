import io
import os
import re

from setuptools import setup
from setuptools import find_packages


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())

setup(
    name="shepherd_gym",
    version="0.1.0",
    url="https://github.com/buntyke/shepherd_gym",
    license='MIT',

    author="Nishanth Koganti",
    author_email="buntyke@gmail.com",

    description="Gym environment implementation of dog shepherding task",
    long_description=read("README.md"),

    packages=find_packages(exclude=('tests','examples')),

    install_requires=['gym>=0.10.8',
                      'numpy>=1.15.0',
                      'matplotlib>=2.2.2'],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
