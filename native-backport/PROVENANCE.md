Isolated experimental backport; not an upstream release. No Isaac pins modified.
Backports only ctypes ABI selection from numba/numba#10644; NumPy cap remains <2.5.
Native build/runtime validation is a separate gate. Versioneer generates a PEP440 local commit suffix.
