#include "error.hpp"
#include "predictor.hpp"

#include <cassert>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <onnxruntime_cxx_api.h>

#ifdef ORT_WITH_GPU
#include <cuda_provider_factory.h>
#endif

using std::string;

/* Description: The structure to handle the predictor for onnxruntime
 * Note: Call ConvertOutput before you want to read the outputs
 */ 
struct Predictor {
  Predictor(const string &model_file, ORT_DeviceKind device);
  ~Predictor();
  void Predict(void);
  void ConvertOutput(void);
  void AddOutput(Ort::Value&);
  void Clear(void);
  void *ConvertTensorToPointer(Ort::Value&, size_t);
  void EndProfiling(void);
  struct Onnxruntime_Env {
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    /* Description: Follow the sample given in onnxruntime to initialize the environment
     * Referenced: https://github.com/microsoft/onnxruntime/blob/master/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp
     */
    Onnxruntime_Env(ORT_DeviceKind device) : env_(ORT_LOGGING_LEVEL_ERROR, "ort_predict") {
      // Initialize environment, could use ORT_LOGGING_LEVEL_VERBOSE to get more information
      // NOTE: Only one instance of env can exist at any point in time
      
      // enable profiling, the argument is the prefix you want for the file
      session_options_.EnableProfiling("onnxruntime");
      
      #ifdef ORT_WITH_GPU
      if (device == CUDA_DEVICE_KIND) {
        OrtSessionOptionsAppendExecutionProvider_CUDA(session_options_, 0 /* device id */);
      }
      #endif

      // Sets graph optimization level
      // Available levels are
      // ORT_DISABLE_ALL -> To disable all optimizations
      // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
      // ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
      // ORT_ENABLE_ALL -> To Enable All possible opitmizations
      session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    }
  } ort_env_;
  // Order matters when using member initializer lists
  int64_t profile_start_;
  Ort::Session session_;
  Ort::AllocatorWithDefaultOptions allocator_;
  string profile_filename_;
  std::vector<const char*> input_node_;
  std::vector<Ort::Value> input_;
  std::vector<const char*> output_node_;
  std::vector<Ort::Value> output_;
  std::vector<ORT_Value> converted_output_;
};

/* Description: Follow the sample given in onnxruntime to initialize the predictor
 * Referenced: https://github.com/microsoft/onnxruntime/blob/master/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp
 */
Predictor::Predictor(const string &model_file, ORT_DeviceKind device)
  : ort_env_(device), 
    session_(ort_env_.env_, model_file.c_str(), ort_env_.session_options_) {
    //profile_start_ = static_cast<int64_t>(session_.GetProfilingStartTimeNs());

  // get input info
  size_t num_input_nodes = session_.GetInputCount();

  for (size_t i = 0; i < num_input_nodes; i++) {
    // get input node names and dimensions
    input_node_.push_back(session_.GetInputName(i, allocator_));
  }

  // get output info
  size_t num_output_nodes = session_.GetOutputCount();

  for (size_t i = 0; i < num_output_nodes; i++) {
    // get output node names
    output_node_.push_back(session_.GetOutputName(i, allocator_));
  }

}

/* Description: clean up the predictor for next prediction */
void Predictor::Clear() {
  for(size_t i = 0; i < converted_output_.size(); i++) {
    free(converted_output_[i].data_ptr);
    free((void*) converted_output_[i].shape_ptr);
    converted_output_[i].data_ptr = nullptr;
    converted_output_[i].shape_ptr = nullptr;
  }
  converted_output_.clear();
  input_.clear();
}

/* Description: Destructor of the predictor to clean up dynamic allocated momory */
Predictor::~Predictor() {
  Clear();
}

/* Description: Do the inference in onnxruntime */
void Predictor::Predict(void) {
  // check invalid dims size
  if (input_.size() != input_node_.size()) {
    throw std::runtime_error(std::string("Invalid number of input tensor in Predictor::Predict."));
  }

  output_ = session_.Run(Ort::RunOptions{nullptr}, input_node_.data(), input_.data(),
                         input_.size(), output_node_.data(), output_node_.size());

  //profile_filename_ = session_.EndProfiling(allocator_);

}

/* Description: Convert Ort::Value to an array pointed by the pointer */
void *Predictor::ConvertTensorToPointer(Ort::Value& value, size_t size) {
  void *res = nullptr;
  switch (value.GetTensorTypeAndShapeInfo().GetElementType()) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
      throw std::runtime_error(std::string("undefined data type detected in ConvertTensorToPointer."));
    break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      res = (void*) malloc(sizeof(float) * size);
      memcpy(res, value.GetTensorMutableData<float>(), sizeof(float) * size);
    break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      res = (void*) malloc(sizeof(uint8_t) * size);
      memcpy(res, value.GetTensorMutableData<uint8_t>(), sizeof(uint8_t) * size);
    break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      res = (void*) malloc(sizeof(int8_t) * size);
      memcpy(res, value.GetTensorMutableData<int8_t>(), sizeof(int8_t) * size);
    break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      res = (void*) malloc(sizeof(uint16_t) * size);
      memcpy(res, value.GetTensorMutableData<uint16_t>(), sizeof(uint16_t) * size);
    break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      res = (void*) malloc(sizeof(int16_t) * size);
      memcpy(res, value.GetTensorMutableData<int16_t>(), sizeof(int16_t) * size);
    break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      res = (void*) malloc(sizeof(int32_t) * size);
      memcpy(res, value.GetTensorMutableData<int32_t>(), sizeof(int32_t) * size);
    break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      res = (void*) malloc(sizeof(int64_t) * size);
      memcpy(res, value.GetTensorMutableData<int64_t>(), sizeof(int64_t) * size);
    break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      res = (void*) malloc(sizeof(bool) * size);
      memcpy(res, value.GetTensorMutableData<bool>(), sizeof(bool) * size);
    break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      res = (void*) malloc(sizeof(double) * size);
      memcpy(res, value.GetTensorMutableData<double>(), sizeof(double) * size);
    break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      res = (void*) malloc(sizeof(uint32_t) * size);
      memcpy(res, value.GetTensorMutableData<uint32_t>(), sizeof(uint32_t) * size);
    break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      res = (void*) malloc(sizeof(uint64_t) * size);
      memcpy(res, value.GetTensorMutableData<uint64_t>(), sizeof(uint64_t) * size);
    break;
    default: // c++: FLOAT16; onnxruntime: COMPLEX64, COMPLEX128, BFLOAT16; TODO: Implement String method
      throw std::runtime_error(std::string("unsupported data type detected in Predictor::ConvertTensorToPointer."));
  }
  return res;
}

/* Description: The helper function when calling ConvertOutput for converting all outputs into array form
 *              Since Ort::Value can be a tensor, a map or a sequence, we need to decompose it by recursion
 */
void Predictor::AddOutput(Ort::Value& value) {
  // base case
  if (value.IsTensor()) {
    auto tensor_info = value.GetTensorTypeAndShapeInfo();
    auto dims = tensor_info.GetShape();
    int64_t *shapes = (int64_t*) malloc(sizeof(int64_t) * dims.size());
    size_t size = 1;
    for (size_t i = 0; i < dims.size(); i++) {
      size *= dims[i];
      shapes[i] = dims[i];
    }
    converted_output_.push_back(ORT_Value{
      .otype = tensor_info.GetElementType(),
      .data_ptr = ConvertTensorToPointer(value, size),
      .shape_ptr = shapes,
      .shape_len = dims.size()
    });
    return;
  }
  
  // need to be decomposed, it is a map or a sequence, both can be done in the same way
  size_t length = value.GetCount();

  for (size_t i = 0; i < length; i++) {
    auto cur_val = value.GetValue(static_cast<int>(i), allocator_);
    AddOutput(cur_val);
  }
}

/* Description: The function need to be called before reading outputs from Go */
void Predictor::ConvertOutput(void) {
  for (size_t i = 0; i < output_.size(); i++) {
    AddOutput(output_[i]);
  }
}

void Predictor::EndProfiling(void) {
  profile_filename_ = session_.EndProfiling(allocator_);
}
void ORT_EndProfiling(ORT_PredictorContext pred) {
  HANDLE_ORT_ERRORS(ORT_GlobalError);
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    throw std::runtime_error(std::string("Invalid pointer to the predictor in ORT_PredictorClear."));
  }
  predictor->EndProfiling();
  END_HANDLE_ORT_ERRORS(ORT_GlobalError, void());
}
/* Description: The interface for Go to create a new predictor */
ORT_PredictorContext ORT_NewPredictor(const char *model_file, ORT_DeviceKind device) {
  HANDLE_ORT_ERRORS(ORT_GlobalError);
  const auto ctx = new Predictor(model_file, device);
  return (ORT_PredictorContext) ctx;
  END_HANDLE_ORT_ERRORS(ORT_GlobalError, (ORT_PredictorContext) nullptr);
}

/* Description: The interface for Go to clear the predictor */
void ORT_PredictorClear(ORT_PredictorContext pred) {
  HANDLE_ORT_ERRORS(ORT_GlobalError);
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    throw std::runtime_error(std::string("Invalid pointer to the predictor in ORT_PredictorClear."));
  }
  predictor->Clear();
  END_HANDLE_ORT_ERRORS(ORT_GlobalError, void());
}

/* Description: The interface for Go to do inference */
void ORT_PredictorRun(ORT_PredictorContext pred) {
  HANDLE_ORT_ERRORS(ORT_GlobalError);
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    throw std::runtime_error(std::string("Invalid pointer to the predictor in ORT_PredictorRun."));
  }
  predictor->Predict();
  END_HANDLE_ORT_ERRORS(ORT_GlobalError, void());
}

/* Description: The interface for Go to convert outputs before reading outputs */
void ORT_PredictorConvertOutput(ORT_PredictorContext pred) {
  HANDLE_ORT_ERRORS(ORT_GlobalError);
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    throw std::runtime_error(std::string("Invalid pointer to the predictor in ORT_PredictorConvertOutput."));
  }

  predictor -> ConvertOutput();

  END_HANDLE_ORT_ERRORS(ORT_GlobalError, void());
}

/* Description: The interface for Go to know the number of converted outputs */
int ORT_PredictorNumOutputs(ORT_PredictorContext pred) {
  HANDLE_ORT_ERRORS(ORT_GlobalError);
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    throw std::runtime_error(std::string("Invalid pointer to the predictor in ORT_PredictorNumOutputs."));
  }
  return (int) ((predictor -> converted_output_).size());
  END_HANDLE_ORT_ERRORS(ORT_GlobalError, 0);
}

/* Description: The interface for Go to get the number of converted outputs */
ORT_Value ORT_PredictorGetOutput(ORT_PredictorContext pred, int index) {
  HANDLE_ORT_ERRORS(ORT_GlobalError);
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    throw std::runtime_error(std::string("Invalid pointer to the predictor in ORT_PredictorGetOutput."));
  }

  return (predictor -> converted_output_)[index];

  END_HANDLE_ORT_ERRORS(ORT_GlobalError, ORT_Value{});
}

/* Description: The interface for Go to delete the dynamic allocated predictor
 *              The destructor for the predictor will be called when deleting the predictor
 */
void ORT_PredictorDelete(ORT_PredictorContext pred) {
  HANDLE_ORT_ERRORS(ORT_GlobalError);
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    throw std::runtime_error(std::string("Invalid pointer to the predictor in ORT_PredictorDelete."));
  }
  delete predictor;
  END_HANDLE_ORT_ERRORS(ORT_GlobalError, void());
}

/* Description: The interface for Go to read the profile in framework level from onnxruntime */
char *ORT_ProfilingRead(ORT_PredictorContext pred) {
  HANDLE_ORT_ERRORS(ORT_GlobalError);
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    throw std::runtime_error(std::string("Invalid pointer to the predictor in ORT_ProfilingRead."));
  }
  
  std::stringstream ss;
  std::ifstream in(predictor -> profile_filename_);
  ss << in.rdbuf();
  return strdup(ss.str().c_str());

  END_HANDLE_ORT_ERRORS(ORT_GlobalError, strdup(""));
}

/* Description: The interface for Go to get the start time of the profiler */
int64_t ORT_ProfilingGetStartTime(ORT_PredictorContext pred) {
  HANDLE_ORT_ERRORS(ORT_GlobalError);
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    throw std::runtime_error(std::string("Invalid pointer to the predictor in ORT_ProfilingGetStartTime."));
  }

  return static_cast<int64_t>(predictor->session_.GetProfilingStartTimeNs());
  END_HANDLE_ORT_ERRORS(ORT_GlobalError, -1);
}

/* Description: The interface for Go to add inputs into the predictor */
void ORT_AddInput(ORT_PredictorContext pred, void *input, int64_t *dimensions,
                  int n_dim, ONNXTensorElementDataType dtype) {
  HANDLE_ORT_ERRORS(ORT_GlobalError);
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    throw std::runtime_error(std::string("Invalid pointer to the predictor in ORT_AddInput."));
  }
  std::vector<int64_t> dims;
  dims.assign(dimensions, dimensions + n_dim);
  size_t size = 1;
  for (int i = 0; i < n_dim; i++)
    size *= dims[i];

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  switch (dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
      throw std::runtime_error(std::string("undefined data type detected in ORT_AddInput."));
    break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      (predictor -> input_).emplace_back(Ort::Value::CreateTensor<float>(memory_info, static_cast<float*>(input) , size, dims.data(), dims.size()));
    break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      (predictor -> input_).emplace_back(Ort::Value::CreateTensor<uint8_t>(memory_info, static_cast<uint8_t*>(input) , size, dims.data(), dims.size()));
    break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      (predictor -> input_).emplace_back(Ort::Value::CreateTensor<int8_t>(memory_info, static_cast<int8_t*>(input) , size, dims.data(), dims.size()));
    break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      (predictor -> input_).emplace_back(Ort::Value::CreateTensor<uint16_t>(memory_info, static_cast<uint16_t*>(input) , size, dims.data(), dims.size()));
    break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      (predictor -> input_).emplace_back(Ort::Value::CreateTensor<int16_t>(memory_info, static_cast<int16_t*>(input) , size, dims.data(), dims.size()));
    break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      (predictor -> input_).emplace_back(Ort::Value::CreateTensor<int32_t>(memory_info, static_cast<int32_t*>(input) , size, dims.data(), dims.size()));
    break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      (predictor -> input_).emplace_back(Ort::Value::CreateTensor<int64_t>(memory_info, static_cast<int64_t*>(input) , size, dims.data(), dims.size()));
    break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      (predictor -> input_).emplace_back(Ort::Value::CreateTensor<bool>(memory_info, static_cast<bool*>(input) , size, dims.data(), dims.size()));
    break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      (predictor -> input_).emplace_back(Ort::Value::CreateTensor<double>(memory_info, static_cast<double*>(input) , size, dims.data(), dims.size()));
    break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      (predictor -> input_).emplace_back(Ort::Value::CreateTensor<uint32_t>(memory_info, static_cast<uint32_t*>(input) , size, dims.data(), dims.size()));
    break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      (predictor -> input_).emplace_back(Ort::Value::CreateTensor<uint64_t>(memory_info, static_cast<uint64_t*>(input) , size, dims.data(), dims.size()));
    break;
    default: // c++: FLOAT16; onnxruntime: COMPLEX64, COMPLEX128, BFLOAT16; TODO: Implement String method
      throw std::runtime_error(std::string("unsupported data type detected in ORT_AddInput."));
  }
  END_HANDLE_ORT_ERRORS(ORT_GlobalError, void());
}
