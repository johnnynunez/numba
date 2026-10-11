"""Exercise the actual legacy calling-convention selection expression."""
import ast
import ctypes
from pathlib import Path
import platform
import sys
import types
import unittest
from unittest.mock import patch
root=Path(sys.argv.pop(1))
class ABI(unittest.TestCase):
    def test_arm64_is_not_x86_stdcall(self):
        tree=ast.parse((root/'numba/core/typing/ctypes_utils.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_function_type')
        branch=next(n for n in ast.walk(fn) if isinstance(n,ast.If) and any(isinstance(x,ast.Constant) and x.value=='x86_stdcallcc' for x in ast.walk(n)))
        code=compile(ast.Expression(branch.test),'<upstream condition>','eval')
        with patch.object(sys,'platform','win32'),patch.object(platform,'machine',return_value='ARM64'):
            self.assertFalse(eval(code,dict(sys=sys,platform=platform,ctypes=ctypes,cfnptr=types.SimpleNamespace(_flags_=0))))
if __name__=='__main__': unittest.main()
