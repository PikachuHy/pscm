# Build PikachuHy's Scheme


## CMake (Recommanded)

```shell
mkdir build && cd build
cmake ..
make -j
```

## xmake

```shell
xmake config --yes --mode=Release
xmake build --jobs=4
```

## Bazel

```shell
bazel build :all
```

## pscm-build (WIP)

```shell
pscm-build
```
