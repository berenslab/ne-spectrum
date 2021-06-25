from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy

bh_umap = Extension(
    name="bhumap",
    sources=["jnb_msc/transformer/bhumap.pyx"],
    language="c++",
    library_dirs=["jnb_msc"],
    ## This needs to be adapted for the respective location
    include_dirs=["../../../msc/vendor/python/openTSNE/", numpy.get_include()],
)

bh_noack = Extension(
    name="bhnoack",
    sources=["jnb_msc/transformer/bhnoack.pyx"],
    language="c++",
    library_dirs=["jnb_msc"],
    ## This needs to be adapted for the respective location
    include_dirs=["../../../msc/vendor/python/openTSNE/", numpy.get_include()],
)

setup(
    name="jnb_msc",
    version="0.0.1",
    description="Python code associated with my master thesis",
    url="https://jnboehm.com/",
    author="Niklas Böhm",
    author_email="jan-niklas.boehm@student.uni-tuebingen.de",
    packages=["jnb_msc"],
    ext_modules=cythonize([bh_umap, bh_noack]),
)
