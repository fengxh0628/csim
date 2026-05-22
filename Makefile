PYTHON ?= python3

define CYTHON_SETUP
import sysconfig, os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np

class build_ext_short_suffix(build_ext):
    """Strip platform suffix: .cpython-3xx-xxx.so -> .so"""
    def get_ext_filename(self, ext_name):
        filename = super().get_ext_filename(ext_name)
        name, ext = os.path.splitext(filename)
        ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
        if ext_suffix == ext:
            return filename
        ext_suffix = ext_suffix.replace(ext, '')
        idx = name.find(ext_suffix)
        return filename if idx == -1 else name[:idx] + ext

extensions = [Extension('fast', ['fast.pyx'], include_dirs=[np.get_include()])]
setup(
    name='fast',
    cmdclass={'build_ext': build_ext_short_suffix},
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': 3, 'boundscheck': False},
    ),
)
endef
export CYTHON_SETUP

.PHONY: all lib clean

all: lib

lib: lib/fast.so

lib/fast.so: lib/fast.pyx
	@echo "  BUILD   lib/fast.so"
	@cd lib && printf '%s\n' "$$CYTHON_SETUP" > setup.py \
	    && $(PYTHON) setup.py build_ext --inplace \
	    && rm -f setup.py
	@rm -rf lib/build lib/fast.c

clean:
	rm -f lib/fast.so lib/fast.c
	rm -rf lib/build
