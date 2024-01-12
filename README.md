# PRESEIS seismic source location

## Description

This module provides functionality to compute proability distributions of seismic source locations based on (first) arrival time data, using Bayesian inference.

## Configuration

A python environment can be created using conda from [environment.yml](environment.yml), i.e.:
```bash
conda env create --file environment.yml
conda activate source-location
```

In addition, the package [pykonal](https://github.com/malcolmw/pykonal) needs to be installed from source, i.e.:
```bash
git clone https://github.com/malcolmw/pykonal
cd pykonal
pip install .
```


## Usage

The package provides basic functionality for computing probabilistic source location distributions based on first arrival time data.
See [examples.ipnb](examples.ipynb) for a usage example.

## License

MIT License

Copyright (c) 2023, 2024 TNO

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
