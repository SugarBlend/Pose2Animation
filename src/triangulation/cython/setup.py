from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="adaptive_triangulation",
        sources=["adaptive_triangulation.pyx"],
        include_dirs=[np.get_include()],
    ),
    Extension(
        name="linear_triangulation",
        sources=["linear_triangulation.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="triangulation",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
    zip_safe=False
)
