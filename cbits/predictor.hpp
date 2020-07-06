#ifndef __PREDICTOR_HPP__
#define __PREDICTOR_HPP__

#ifdef __cplusplus
#include <onnxruntime_cxx_api.h>
extern "C" {
#else
#include <onnxruntime_c_api.h>
#endif  /* __cplusplus */

  typedef struct ORT_Error {
    char* message;
  } ORT_Error;

  typedef struct ORT_Value {
    ONNXTensorElementDataType otype;
    void *data_ptr;
    int64_t *shape_ptr;
    size_t shape_len;
  } ORT_Value;

  extern ORT_Error ORT_GlobalError;
  typedef enum { UNKNOWN_DEVICE_KIND = -1, CPU_DEVICE_KIND = 0, CUDA_DEVICE_KIND = 1 } ORT_DeviceKind;
  typedef void* ORT_PredictorContext;
  typedef void* ORT_TensorContext;

  // Predictor + Profiling interface for Go

  ORT_PredictorContext ORT_NewPredictor(const char *model_file, ORT_DeviceKind device);

  void ORT_PredictorRun(ORT_PredictorContext pred);

  void ORT_PredictorConvertOutput(ORT_PredictorContext pred);

  int ORT_PredictorNumOutputs(ORT_PredictorContext pred);

  ORT_Value ORT_PredictorGetOutput(ORT_PredictorContext pred, int index);

  void ORT_PredictorDelete(ORT_PredictorContext pred);

  char *ORT_ProfilingRead(ORT_PredictorContext pred);

  int64_t ORT_ProfilingGetStartTime(ORT_PredictorContext pred);

  void ORT_AddInput(ORT_PredictorContext pred, void *input, int64_t *dimensions,
                    int n_dim, ONNXTensorElementDataType dtype);

  // Error interface for Go

  int ORT_HasError();

  const char* ORT_GetErrorString();

  void ORT_ResetError();

#ifdef __cplusplus
}
#endif  /* __cplusplus */

#endif /* __PREDICTOR_HPP__ */