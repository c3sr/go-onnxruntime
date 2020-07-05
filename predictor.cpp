#include "error.hpp"
#include "predictor.hpp"

#include <cassert>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>

#ifdef ORT_WITH_GPU
#include <cuda_provider_factory.h>
#endif

using std::string;

// TODO:ADD GPU
struct Predictor {
  Predictor(const string &model_file, ORT_DeviceKind device);
  void Predict(void);
  struct Onnxruntime_Env {
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    /* Description: Follow the sample given in onnxruntime to initialize the environment
     * Referenced: https://github.com/microsoft/onnxruntime/blob/master/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp
     */
    Onnxruntime_Env(ORT_DeviceKind device) : env_(ORT_LOGGING_LEVEL_WARNING, "ort_predict") {
      // Initialize environment, could use ORT_LOGGING_LEVEL_VERBOSE to get more information
      // NOTE: Only one instance of env can exist at any point in time
      
      session_options_.SetIntraOpNumThreads(1);

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
      session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    }
  } ort_env_;
  Ort::Session session_;
  Ort::AllocatorWithDefaultOptions allocator_;
  string profile_filename_;
  int64_t profile_start_;
  std::vector<const char*> input_node_;
  std::vector<Ort::Value> input_;
  std::vector<const char*> output_node_;
  std::vector<Ort::Value> output_;
};


/* Description: Follow the sample given in onnxruntime to initialize the predictor
 * Referenced: https://github.com/microsoft/onnxruntime/blob/master/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp
 */
Predictor::Predictor(const string &model_file, ORT_DeviceKind device)
  : ort_env_(device), session_(ort_env_.env_, model_file.c_str(), ort_env_.session_options_) {
  // TODO: find an adequate timestamp
  profile_start_ = static_cast<int64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count());

  // get input info
  size_t num_input_nodes = session_.GetInputCount();

  for (size_t i = 0; i < num_input_nodes; i++) {
    // get input node names and dimensions
    input_node_.push_back(session_.GetInputName(i, allocator_));
    std::cout << "Input " << i << " : name = " << input_node_[i] << "\n";
  }

  // get output info
  size_t num_output_nodes = session_.GetOutputCount();

  for (size_t i = 0; i < num_output_nodes; i++) {
    // get output node names
    output_node_.push_back(session_.GetOutputName(i, allocator_));
    std::cout << "Output " << i << " : name = " << output_node_[i] << "\n";
  }

}

void Predictor::Predict(void) {
  HANDLE_ORT_ERRORS(ORT_GlobalError);

  // check invalid dims size
  if (input_.size() != input_node_.size()) {
    throw std::runtime_error(std::string("Invalid number of input tensor in Predictor::Predict."));
  }

  output_ = session_.Run(Ort::RunOptions{nullptr}, input_node_.data(), input_.data(),
                         input_.size(), output_node_.data(), output_node_.size());

  profile_filename_ = session_.EndProfiling(allocator_);

  END_HANDLE_ORT_ERRORS(ORT_GlobalError, void());
}

ORT_PredictorContext ORT_NewPredictor(const char *model_file, ORT_DeviceKind device) {
  HANDLE_ORT_ERRORS(ORT_GlobalError);
  const auto ctx = new Predictor(model_file, device);
  return (ORT_PredictorContext) ctx;
  END_HANDLE_ORT_ERRORS(ORT_GlobalError, (ORT_PredictorContext) nullptr);
}

void ORT_PredictorRun(ORT_PredictorContext pred) {
  HANDLE_ORT_ERRORS(ORT_GlobalError);
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    throw std::runtime_error(std::string("Invalid pointer to the predictor in ORT_PredictorRun."));
  }
  predictor->Predict();
  END_HANDLE_ORT_ERRORS(ORT_GlobalError, void());
}

// int Torch_PredictorNumOutputs(Torch_PredictorContext pred) {
//   HANDLE_TH_ERRORS(Torch_GlobalError);
//   auto predictor = (Predictor *)pred;
//   if (predictor == nullptr) {
//     return 0;
//   }
//   if (predictor->output_.isTensor()) {
//     return 1;
//   }
//   if (predictor->output_.isTuple()) {
//     return predictor->output_.toTuple()->elements().size();
//   }

//   return 0;
//   END_HANDLE_TH_ERRORS(Torch_GlobalError, 0);
// }

// Torch_IValue Torch_PredictorGetOutput(Torch_PredictorContext pred) {
//   HANDLE_TH_ERRORS(Torch_GlobalError);
//   auto predictor = (Predictor *)pred;
//   if (predictor == nullptr) {
//     return Torch_IValue{};
//   }

//   return Torch_ConvertIValueToTorchIValue(predictor->output_);

//   END_HANDLE_TH_ERRORS(Torch_GlobalError, Torch_IValue{});
// }

void ORT_PredictorDelete(ORT_PredictorContext pred) {
  HANDLE_ORT_ERRORS(ORT_GlobalError);
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    throw std::runtime_error(std::string("Invalid pointer to the predictor in ORT_PredictorDelete."));
  }
  delete predictor;
  END_HANDLE_ORT_ERRORS(ORT_GlobalError, void());
}

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

int64_t ORT_ProfilingGetStartTime(ORT_PredictorContext pred) {
  HANDLE_ORT_ERRORS(ORT_GlobalError);
  auto predictor = (Predictor *)pred;
  if (predictor == nullptr) {
    throw std::runtime_error(std::string("Invalid pointer to the predictor in ORT_ProfilingGetStartTime."));
  }

  return predictor -> profile_start_;
  END_HANDLE_ORT_ERRORS(ORT_GlobalError, -1);
}

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