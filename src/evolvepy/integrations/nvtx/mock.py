import contextlib


class annotate(contextlib.nullcontext):
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, func):
            return func


def mark(*args, **kwargs): pass
def push_range(*args, **kwargs): pass
def pop_range(*args, **kwargs): pass
def start_range(*args, **kwargs): pass
def end_range(*args, **kwargs): pass