try:
    import fast
except ImportError:
    from . import fast_py as fast
