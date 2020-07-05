#ifndef __ERROR_HPP__
#define __ERROR_HPP__

#include <exceptions.h>

/* Description: try and catch MACROS used for C++ functions
 * Referenced: 
 * https://github.com/rai-project/go-pytorch/blob/master/cbits/error.hpp
 * https://github.com/microsoft/onnxruntime/blob/master/include/onnxruntime/core/common/exceptions.h
 * NOTE: Put a semicolumn after using these MACROS, as a C++ function.
 */

#define HANDLE_ORT_ERRORS(errVar)      \
  try {                                \
    if (errVar.message != nullptr) {   \
      free(errVar.message);            \
    }                                  \
    errVar.message = nullptr


#define END_HANDLE_ORT_ERRORS(errVar, retVal)                    \
  }                                                              \
  catch (const onnxruntime::OnnxRuntimeException &e) {           \
    auto msg = e.what();                                         \
    std::cout << "Onnxruntime Exception msg = " << msg << "\n";  \
    errVar.message = strdup(msg);                                \
  }                                                              \
  catch (const std::exception &e) {                              \
    auto msg = e.what();                                         \
    std::cout << "Std Exception msg = " << msg << "\n";          \
    errVar.message = strdup(msg);                                \
  }                                                              \
  return retVal

#endif /* __ERROR_HPP__ */