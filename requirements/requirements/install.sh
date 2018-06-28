#!/bin/bash

### Here is package reference list in Centos 7.2 64bit.
### Pick some accourding to local OS.
yum -y install gcc
yum -y install python-devel
yum -y install python-pip
pip install --upgrade pip
pip install --upgrade python-dateutil

### Some machine Learning related packages.
yum -y install tkinter
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install sklearn
