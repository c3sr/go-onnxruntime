package onnxruntime

// #include "cbits/predictor.hpp"
import "C"

import (
	"reflect"

	"gorgonia.org/tensor"
)

/* Description: type conversion between C++ and Golang
 * Reference: https://github.com/microsoft/onnxruntime/blob/master/include/onnxruntime/core/session/onnxruntime_c_api.h
 * Note: Currently, Ort doesn't support complex64, complex128, bfloat16 types
 */

var types = []struct {
	typ      reflect.Type
	dataType C.ONNXTensorElementDataType
}{
	{reflect.TypeOf(float32(0)), C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT},
	{reflect.TypeOf(uint8(0)), C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8},
	{reflect.TypeOf(int8(0)), C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8},
	{reflect.TypeOf(uint16(0)), C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16},
	{reflect.TypeOf(int16(0)), C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16},
	{reflect.TypeOf(int32(0)), C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32},
	{reflect.TypeOf(int64(0)), C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64},
	{reflect.TypeOf(string("")), C.ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING},
	{reflect.TypeOf(bool(false)), C.ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL},
	{reflect.TypeOf(float16(0)), C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16},
	{reflect.TypeOf(float64(0)), C.ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE},
	{reflect.TypeOf(uint32(0)), C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32},
	{reflect.TypeOf(uint64(0)), C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64},
	// {reflect.TypeOf(complex64(0)), C.ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64},
	// {reflect.TypeOf(complex128(0)), C.ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128},
}

func fromType(ten *tensor.Dense) C.ONNXTensorElementDataType {
	for _, t := range types {
		if t.typ == ten.Dtype().Type {
			return t.dataType
		}
	}
	return C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
}
