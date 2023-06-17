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
