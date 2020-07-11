# go-onnxruntime

Go binding for Onnxruntime C++ API.
This is used by the [Onnxruntime agent](https://github.com/rai-project/onnxruntime) in [MLModelScope](mlmodelscope.org) to perform model inference in Go.

## Installation

Download and install go-onnxruntime:

```
go get -v github.com/rai-project/go-onnxruntime
```

The binding requires Onnxruntime C++ and other Go packages.

### Onnxruntime C++ Library

The Go binding for Onnxruntime C++ API in this repository is built based on Onnxruntime v1.3.1 .

To install Onnxruntime C++ on your system, you can follow the instruction on [Onnxruntime Installation](https://microsoft.github.io/onnxruntime/).

The Onnxruntime C++ libraries are expected to be under `/opt/onnxruntime/lib`.

The Onnxruntime C++ header files (the [include](https://github.com/microsoft/onnxruntime/tree/master/include) directory from [Onnxruntime](https://github.com/microsoft/onnxruntime)) are expected to be under `/opt/onnxruntime`.

Now there are two directories, `lib` and `include`, under `/opt/onnxruntime/`.

If you get an error about not being able to write to `/opt` then perform the following:

```
sudo mkdir -p /opt/onnxruntime
sudo chown -R `whoami` /opt/onnxruntime
```

### Go Packages

You can install the dependency through using [Dep](https://github.com/golang/dep).

```
dep ensure -v -vendor-only
```

This installs the dependencies in `vendor/`.

### Configure Environmental Variables

Configure the linker environmental variables since the Onnxruntime C++ library is under a non-system directory. Place the following in either your `~/.bashrc` or `~/.zshrc` file:

Linux
```
export LIBRARY_PATH=$LIBRARY_PATH:/opt/onnxruntime/lib
export LD_LIBRARY_PATH=/opt/onnxruntime/lib:$DYLD_LIBRARY_PATH

```

macOS
```
export LIBRARY_PATH=$LIBRARY_PATH:/opt/onnxruntime/lib
export DYLD_LIBRARY_PATH=/opt/onnxruntime/lib:$DYLD_LIBRARY_PATH
```
## Check the Build

Run `go build` to check the dependencies, installation and library paths set-up.
On linux, the default is to use GPU, if you don't have a GPU, do `go build -tags=nogpu` instead of `go build`.

**_Note_** : The CGO interface passes go pointers to the C API. This is an error by the CGO runtime. Disable the error by placing:

```
export GODEBUG=cgocheck=0
```

in your `~/.bashrc` or `~/.zshrc` file and then run either `source ~/.bashrc` or `source ~/.zshrc`.

## Examples

Examples of using the Go Onnxruntime binding to do model inference are under [examples](examples) .

## Credits

Some of the logic of conversion between Go types and Ort::Values is borrowed from [go-pytorch](https://github.com/rai-project/go-pytorch).