#include "error.hpp"
#include "predictor.hpp"

/* Description: The interface for getting errors in C++, used by Go functions.
 * Referenced: https://github.com/rai-project/go-pytorch/blob/master/error.cpp
 */

ORT_Error ORT_GlobalError{.message = nullptr};

int ORT_HasError() {
  if (ORT_GlobalError.message == nullptr) {
    return 0;
  }
  return 1;
}

const char* ORT_GetErrorString() {
  if (!ORT_HasError()) {
    return nullptr;
  }
  return ORT_GlobalError.message;
}

void ORT_ResetError() {
  if (ORT_HasError()) {
    free(ORT_GlobalError.message);
    ORT_GlobalError.message = nullptr;
  }
}