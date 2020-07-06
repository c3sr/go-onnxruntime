package onnxruntime

// #include "cbits/predictor.hpp"
import "C"

import (
	"unsafe"

	"gorgonia.org/tensor"
)

/* Description: Convert data from a pointer to an int slice
 * Referenced: https://github.com/rai-project/go-pytorch/blob/master/utils.go
 */
func toIntSlice(data []int64) []int {
	res := make([]int, len(data))
	for i, d := range data {
		res[i] = int(d)
	}
	return res
}

/* Description: Get flatten length for a shape
 * Referenced: https://github.com/rai-project/go-pytorch/blob/master/utils.go
 */
func getFlattenedLength(data []int64) int {
	res := 1
	for _, d := range data {
		res *= int(d)
	}
	return res
}

/* Description: Convert Ort_Value from C++ to Go tensor, referenced from ivalueToTensor in go-pytorch
 * Referenced: https://github.com/rai-project/go-pytorch/blob/master/utils.go
 */
func ortValueToTensor(ctx C.ORT_Value) tensor.Tensor {
	shapeLength := int64(ctx.shape_len)
	ptr := ctx.data_ptr
	cShape := ctx.shape_ptr
	ty := ctx.otype

	cShapeSlice := (*[1 << 30]int64)(unsafe.Pointer(cShape))[:shapeLength:shapeLength]

	shape := tensor.Shape(toIntSlice(cShapeSlice))
	flattenedLength := getFlattenedLength(cShapeSlice)

	switch ty {
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
		{
			panic("undefined data type!")
		}
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
		{
			cData := (*[1 << 30]float32)(unsafe.Pointer(ptr))[:flattenedLength:flattenedLength]
			data := make([]float32, flattenedLength)
			copy(data, cData)
			return tensor.NewDense(
				tensor.Float32,
				shape,
				tensor.WithBacking(data),
			)
		}
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
		{
			cData := (*[1 << 30]uint8)(unsafe.Pointer(ptr))[:flattenedLength:flattenedLength]
			data := make([]uint8, flattenedLength)
			copy(data, cData)
			return tensor.NewDense(
				tensor.Uint8,
				shape,
				tensor.WithBacking(data),
			)
		}
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
		{
			cData := (*[1 << 30]int8)(unsafe.Pointer(ptr))[:flattenedLength:flattenedLength]
			data := make([]int8, flattenedLength)
			copy(data, cData)
			return tensor.NewDense(
				tensor.Int8,
				shape,
				tensor.WithBacking(data),
			)
		}
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
		{
			cData := (*[1 << 30]uint16)(unsafe.Pointer(ptr))[:flattenedLength:flattenedLength]
			data := make([]uint16, flattenedLength)
			copy(data, cData)
			return tensor.NewDense(
				tensor.Uint16,
				shape,
				tensor.WithBacking(data),
			)
		}
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
		{
			cData := (*[1 << 30]int16)(unsafe.Pointer(ptr))[:flattenedLength:flattenedLength]
			data := make([]int16, flattenedLength)
			copy(data, cData)
			return tensor.NewDense(
				tensor.Int16,
				shape,
				tensor.WithBacking(data),
			)
		}
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
		{
			cData := (*[1 << 30]int32)(unsafe.Pointer(ptr))[:flattenedLength:flattenedLength]
			data := make([]int32, flattenedLength)
			copy(data, cData)
			return tensor.NewDense(
				tensor.Int32,
				shape,
				tensor.WithBacking(data),
			)
		}
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
		{
			cData := (*[1 << 30]int64)(unsafe.Pointer(ptr))[:flattenedLength:flattenedLength]
			data := make([]int64, flattenedLength)
			copy(data, cData)
			return tensor.NewDense(
				tensor.Int64,
				shape,
				tensor.WithBacking(data),
			)
		}
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
		{
			cData := (*[1 << 30]bool)(unsafe.Pointer(ptr))[:flattenedLength:flattenedLength]
			data := make([]bool, flattenedLength)
			copy(data, cData)
			return tensor.NewDense(
				tensor.Bool,
				shape,
				tensor.WithBacking(data),
			)
		}
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
		{
			cData := (*[1 << 30]float64)(unsafe.Pointer(ptr))[:flattenedLength:flattenedLength]
			data := make([]float64, flattenedLength)
			copy(data, cData)
			return tensor.NewDense(
				tensor.Float64,
				shape,
				tensor.WithBacking(data),
			)
		}
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
		{
			cData := (*[1 << 30]uint32)(unsafe.Pointer(ptr))[:flattenedLength:flattenedLength]
			data := make([]uint32, flattenedLength)
			copy(data, cData)
			return tensor.NewDense(
				tensor.Uint32,
				shape,
				tensor.WithBacking(data),
			)
		}
	case C.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
		{
			cData := (*[1 << 30]uint64)(unsafe.Pointer(ptr))[:flattenedLength:flattenedLength]
			data := make([]uint64, flattenedLength)
			copy(data, cData)
			return tensor.NewDense(
				tensor.Uint64,
				shape,
				tensor.WithBacking(data),
			)
		}
	default:
		panic("invalid data type")
	}
}
