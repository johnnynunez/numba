"""Native WinARM64 CPU/JIT acceptance; no architecture or NumPy overrides."""
import ctypes
import json
import os
import platform
import sys
import sysconfig
import unittest
from pathlib import Path

import numpy as np
import numba
from numba import njit
import llvmlite
from llvmlite import binding as llvm


class NativeAcceptance(unittest.TestCase):
    def test_native_interpreter(self):
        self.assertEqual(sys.platform, 'win32')
        self.assertEqual(sys.version_info[:2], (3, 12))
        self.assertEqual(sysconfig.get_platform(), 'win-arm64')

    def test_mcjit(self):
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        llvm.check_jit_execution()
        target = llvm.Target.from_default_triple()
        self.assertTrue(llvm.get_default_triple().startswith('aarch64-'))
        mod = llvm.parse_assembly('define double @add(double %a, double %b) { %r = fadd double %a, %b\n ret double %r }')
        mod.verify()
        tm = target.create_target_machine(codemodel='jitdefault')
        with llvm.create_mcjit_compiler(mod, tm) as engine:
            engine.finalize_object()
            func = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)(engine.get_function_address('add'))
            self.assertEqual(func(1.25, -3.5), -2.25)

    def test_numeric_strides(self):
        @njit
        def f(x):
            return np.sin(x) + x * x - 0.5 * x
        data = np.random.default_rng(84721).normal(size=257)
        for x in [data, data[::3], data[::-1], data[:0]]:
            np.testing.assert_allclose(f(x), np.sin(x) + x*x - .5*x, rtol=1e-13, atol=1e-13)
        self.assertTrue(f.nopython_signatures)

    def test_ieee_adversaries(self):
        @njit
        def f(x):
            return np.sqrt(x), np.isfinite(x), np.signbit(x)
        x = np.array([0., -0., np.nan, np.inf, -np.inf, -1., 1e-300, 1e300])
        with np.errstate(invalid='ignore'):
            a,b,c = f(x)
            np.testing.assert_allclose(a, np.sqrt(x), equal_nan=True)
            np.testing.assert_array_equal(b, np.isfinite(x))
            np.testing.assert_array_equal(c, np.signbit(x))

    def test_integer_boundaries(self):
        @njit
        def f(x):
            return x + np.int64(1)
        x = np.array([np.iinfo(np.int64).min, -1, 0, np.iinfo(np.int64).max], dtype=np.int64)
        np.testing.assert_array_equal(f(x), x + np.int64(1))

    def test_40_argument_abi(self):
        names = [f'a{i}' for i in range(40)]
        ns = {}
        exec('def f(' + ','.join(names) + '):\n return ' + '+'.join(names), ns)
        compiled = njit(ns['f'])
        args = tuple(float(i) / 7 for i in range(40))
        self.assertAlmostEqual(compiled(*args), ns['f'](*args), places=12)

    def test_unicode_sets(self):
        @njit
        def f(s):
            values = set(s)
            return len(values), 'é' in values
        s = 'éλ🦊aéλ'
        self.assertEqual(f(s), (len(set(s)), True))

    def test_bounds_negative(self):
        @njit(boundscheck=True)
        def f(x, i):
            return x[i]
        with self.assertRaises(IndexError):
            f(np.arange(5.), 5)

    def test_oracle_negative(self):
        # Deliberately poisoned expected value must be rejected.
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(njit(lambda x: x*x)(3.), 8.)

    def test_parallel_workqueue(self):
        @njit(parallel=True)
        def f(x):
            return (x*x).sum()
        x = np.arange(129., dtype=np.float64)
        self.assertEqual(f(x), np.sum(x*x))
        self.assertEqual(numba.threading_layer(), 'workqueue')


if __name__ == '__main__':
    out = Path(sys.argv[1]) if len(sys.argv)>1 else Path('runtime.json')
    metadata = dict(python=sys.version, platform=sysconfig.get_platform(), machine=platform.machine(), numpy=np.__version__, numba=numba.__version__, llvmlite=llvmlite.__version__, llvm=llvm.llvm_version_info, triple=llvm.get_default_triple(), host_cpu=llvm.get_host_cpu_name(), features=llvm.get_host_cpu_features().flatten(), cpu_override=os.environ.get('NUMBA_CPU_NAME'), threading_layer=os.environ.get('NUMBA_THREADING_LAYER'))
    print(json.dumps(metadata, indent=2), flush=True)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(NativeAcceptance))
    metadata.update(tests=result.testsRun, failures=[(str(t),e) for t,e in result.failures], errors=[(str(t),e) for t,e in result.errors], skipped=result.skipped, successful=result.wasSuccessful())
    out.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    sys.exit(not result.wasSuccessful())
