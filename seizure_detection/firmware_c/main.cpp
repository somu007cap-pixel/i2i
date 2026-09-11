// Static-memory TFLite Micro inference harness for the seizure detector.
//
// This file is a board-porting starting point. It intentionally avoids dynamic
// allocation. The exact resolver must be validated against the TFLite Micro
// version and vendor toolchain used for PSoC Edge E84.

#include <cstdint>
#include <cstdio>

#include "model_data.h"
#include "sensor_dsp.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

constexpr int kTensorArenaSize = 384 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

constexpr float kInputScale = 0.08671709895133972f;
constexpr int kInputZeroPoint = -6;
constexpr float kOutputScale = 0.00390625f;
constexpr int kOutputZeroPoint = -128;

using Resolver = tflite::MicroMutableOpResolver<16>;

TfLiteStatus RegisterOps(Resolver* resolver) {
  if (resolver == nullptr) {
    return kTfLiteError;
  }
  TF_LITE_ENSURE_STATUS(resolver->AddAdd());
  TF_LITE_ENSURE_STATUS(resolver->AddConcatenation());
  TF_LITE_ENSURE_STATUS(resolver->AddDequantize());
  TF_LITE_ENSURE_STATUS(resolver->AddExpandDims());
  TF_LITE_ENSURE_STATUS(resolver->AddFullyConnected());
  TF_LITE_ENSURE_STATUS(resolver->AddGather());
  TF_LITE_ENSURE_STATUS(resolver->AddLogistic());
  TF_LITE_ENSURE_STATUS(resolver->AddMean());
  TF_LITE_ENSURE_STATUS(resolver->AddMul());
  TF_LITE_ENSURE_STATUS(resolver->AddPack());
  TF_LITE_ENSURE_STATUS(resolver->AddQuantize());
  TF_LITE_ENSURE_STATUS(resolver->AddReduceMax());
  TF_LITE_ENSURE_STATUS(resolver->AddReshape());
  TF_LITE_ENSURE_STATUS(resolver->AddShape());
  TF_LITE_ENSURE_STATUS(resolver->AddStridedSlice());
  TF_LITE_ENSURE_STATUS(resolver->AddTranspose());
  return kTfLiteOk;
}

float DequantizeOutput(int8_t score) {
  return ((int)score - kOutputZeroPoint) * kOutputScale;
}

}  // namespace

int main() {
  const tflite::Model* model = tflite::GetModel(g_seizure_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    std::printf("Model schema mismatch\n");
    return 1;
  }

  Resolver resolver;
  if (RegisterOps(&resolver) != kTfLiteOk) {
    std::printf("Operator registration failed\n");
    return 1;
  }

  tflite::MicroInterpreter interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);

  if (interpreter.AllocateTensors() != kTfLiteOk) {
    std::printf("Tensor allocation failed; increase kTensorArenaSize\n");
    return 1;
  }

  TfLiteTensor* primary = interpreter.input(0);
  TfLiteTensor* secondary = interpreter.input(1);
  if (primary == nullptr || secondary == nullptr) {
    std::printf("Missing model inputs\n");
    return 1;
  }

  for (int t = 0; t < SEIZURE_WINDOW_SAMPLES; ++t) {
    for (int c = 0; c < SEIZURE_PRIMARY_CHANNELS; ++c) {
      primary->data.int8[t * SEIZURE_PRIMARY_CHANNELS + c] =
          SeizureQuantizeInt8(0.0f, kInputScale, kInputZeroPoint);
    }
    for (int c = 0; c < SEIZURE_SECONDARY_CHANNELS; ++c) {
      secondary->data.int8[t * SEIZURE_SECONDARY_CHANNELS + c] =
          SeizureQuantizeInt8(0.0f, kInputScale, kInputZeroPoint);
    }
  }

  if (interpreter.Invoke() != kTfLiteOk) {
    std::printf("Inference failed\n");
    return 1;
  }

  const TfLiteTensor* output = interpreter.output(0);
  const float probability = DequantizeOutput(output->data.int8[0]);
  std::printf("Seizure event score: %.6f\n", probability);
  return 0;
}
