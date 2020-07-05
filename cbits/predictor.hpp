#ifndef __PREDICTOR_HPP__
#define __PREDICTOR_HPP__

#include <string.h>
#include <onnxruntime_cxx_api.h>

#ifdef __cplusplus
extern "C" {
#endif  /* __cplusplus */

  typedef struct ORT_Error {
    std::string message;
  } ORT_Error;

  extern ORT_Error ORT_GlobalError;
  typedef enum { UNKNOWN_DEVICE_KIND = -1, CPU_DEVICE_KIND = 0, CUDA_DEVICE_KIND = 1 } ORT_DeviceKind;
  typedef void* ORT_PredictorContext;
  typedef void* ORT_TensorContext;

#ifdef __cplusplus
}
#endif  /* __cplusplus */


#endif /* __PREDICTOR_HPP__ */