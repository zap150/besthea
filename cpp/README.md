# Compilation

```
mkdir build
cd build
cmake ..
make
```

```
CXX=icpc CC=icc cmake -DBUILD_TEST=ON -E env CXXFLAGS="-xcore-avx512 -qopt-zmm-usage=high"
```
