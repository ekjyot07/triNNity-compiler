import sys

class CompilerError(Exception):
    pass

def print_stderr(msg):
    sys.stderr.write('%s\n' % msg)
