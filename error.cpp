#include "error.hpp"
#include "predictor.hpp"
#include <string>

/* Description: The interface for getting error in C++, used by Go functions.
 * Referenced: 
 * https://github.com/rai-project/go-pytorch/blob/master/error.cpp
 */

ORT_Error ORT_GlobalError{.message = ""};

int ORT_HasError() {
  if (ORT_GlobalError.message.size() == 0) {
    return 0;
  }
  return 1;
}

const char* ORT_GetErrorString() {
  if (!ORT_HasError()) {
    return nullptr;
  }
  return ORT_GlobalError.message.c_str();
}

void ORT_ResetError() {
  if (ORT_HasError()) {
    ORT_GlobalError.message = "";
  }
}