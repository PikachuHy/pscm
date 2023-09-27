# Build PikachuHy's Scheme

## Preparation

### Required toolchain

if you build other projects under pscm repo

```shell
sudo apt-get install -y build-essential
sudo apt-get install -y libsdl2-dev libsdl2-ttf-dev libsdl2-image-dev libsdl2-mixer-dev
```

### Required git submodule

```shell
git submodule update --init
```

### Required ninja (version >= 1.11.0)

if you build pscm with C++20 Modules

Ref: https://github.com/ninja-build/ninja/releases

## CMake (Recommanded)

```shell
mkdir build && cd build
cmake ..
make -j
```

## xmake

目前Windows下，推荐使用xmake进行构建。xmake可以管理icu等依赖，无需同步git子模块

```shell
xmake config --yes --mode=Release
xmake build --jobs=4
```

## Bazel

```shell
bazel build :all
```

- build pscm android app

set `ANDROID_HOME` and `ANDROID_NDK_HOME`

```
# for example
# export ANDROID_HOME=$HOME/Library/Android/sdk
# export ANDROID_NDK_HOME=$HOME/Library/Android/sdk/ndk/25.1.8937393
bazel build //android/app/src/main:app --android_crosstool_top=@androidndk//:toolchain --fat_apk_cpu=arm64-v8a
```

- install pscm android app

```
adb install bazel-bin/android/app/src/main/app.apk
```

Note: don't
use `bazel mobile-install //android/app/src/main:app --android_crosstool_top=@androidndk//:toolchain --fat_apk_cpu=arm64-v8a`,
which may cause app crash

- build pscm ios app

```
bazel build //ios/app:ios-app
```

- run pscm ios app on simulator

```
bazel run //ios/app:ios-app
```
## pscm-build (WIP)

```shell
pscm-build
```
