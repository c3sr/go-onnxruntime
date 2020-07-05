package onnxruntime

// #include "cbits/predictor.hpp"
// #include <stdlib.h>
//
import "C"
import (
	"unsafe"

	"github.com/pkg/errors"
)

// Error returned by C++
type Error struct {
	message string
}

func (e *Error) Error() string {
	return e.message
}

func checkError(err C.ORT_Error) *Error {
	if err.message != nil {
		defer C.free(unsafe.Pointer(err.message))
		return &Error{
			message: C.GoString(err.message),
		}
	}

	return nil
}

func HasError() bool {
	return int(C.ORT_HasError()) == 1
}

func GetErrorString() string {
	msg := C.ORT_GetErrorString()
	if msg == nil {
		return ""
	}
	return C.GoString(msg)
}

func ResetError() {
	C.ORT_ResetError()
}

func GetError() error {
	if !HasError() {
		return nil
	}
	err := errors.New(GetErrorString())
	ResetError()
	return err
}

func PanicOnError() {
	msg := C.ORT_GetErrorString()
	if msg == nil {
		return
	}
	panic(C.GoString(msg))
}
