package onnxruntime

// #cgo CXXFLAGS: -std=c++11 -I${SRCDIR}/cbits -g -O3 -Wno-unused-result
// #cgo CFLAGS: -I${SRCDIR}/cbits -O3 -Wall -Wno-unused-variable -Wno-deprecated-declarations -Wno-c++11-narrowing -g -Wno-sign-compare -Wno-unused-function
// #cgo LDFLAGS: -lstdc++
// #cgo CFLAGS: -isystem /opt/onnxruntime/include/onnxruntime/core/session/
// #cgo CXXFLAGS: -isystem /opt/onnxruntime/include/onnxruntime/core/session/
// #cgo CXXFLAGS: -isystem /opt/onnxruntime/include/onnxruntime/core/common/
// #cgo CXXFLAGS: -isystem /opt/onnxruntime/include/onnxruntime
// #cgo LDFLAGS: -L/opt/onnxruntime/lib/ -lonnxruntime
// #cgo linux,amd64,!nogpu CXXFLAGS: -isystem /opt/onnxruntime/include/onnxruntime/core/providers/cuda/
// #cgo linux,amd64,!nogpu CXXFLAGS: -I/usr/local/cuda/include -DORT_WITH_GPU
import "C"
