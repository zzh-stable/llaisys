#!/bin/bash
set -e

xmake
xmake install
pip install ./python/ 