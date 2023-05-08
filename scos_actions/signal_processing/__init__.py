# Above this number of array elements, perform computations with NumExpr
# instead of with NumPy. NumExpr is slower for small arrays due to its
# setup overhead. Once arrays reach ~100s of thousands of elements, the
# speedup in the computation makes up for this overhead. This constant
# is used within multiple of the signal processing modules.
NUMEXPR_THRESHOLD = 200000
