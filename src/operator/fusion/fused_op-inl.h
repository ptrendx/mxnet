/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#ifndef MXNET_OPERATOR_FUSION_FUSED_OP_INL_H_
#define MXNET_OPERATOR_FUSION_FUSED_OP_INL_H_

#include <string>
#include <map>
#include <vector>
#include "../nn/activation-inl.h"

namespace mxnet {

namespace detail {

const std::string fp16_support_string = R"code(
#define __HALF_TO_US(var) *(reinterpret_cast<unsigned short *>(&(var)))
#define __HALF_TO_CUS(var) *(reinterpret_cast<const unsigned short *>(&(var)))
#if defined(__cplusplus)
  struct __align__(2) __half {
    __host__ __device__ __half() { }
  protected:
    unsigned short __x;
  };
  /* All intrinsic functions are only available to nvcc compilers */
  #if defined(__CUDACC__)
    /* Definitions of intrinsics */
    __device__ inline __half __float2half(const float f) {
      __half val;
      asm("{  cvt.rn.f16.f32 %0, %1;}\n" : "=h"(__HALF_TO_US(val)) : "f"(f));
      return val;
    }
    __device__ inline float __half2float(const __half h) {
      float val;
      asm("{  cvt.f32.f16 %0, %1;}\n" : "=f"(val) : "h"(__HALF_TO_CUS(h)));
      return val;
    }
  #endif /* defined(__CUDACC__) */
#endif /* defined(__cplusplus) */
#undef __HALF_TO_US
#undef __HALF_TO_CUS
typedef __half half;
)code";

const std::string type_support_string = R"code(
using float32 = float;
using float64 = double;
using float16 = half;
using uint8 = unsigned char;
using int8 = char;
using int32 = int;
using int64 = long long;
)code";

const std::map<std::string, std::string> fused_op_binary_ops = {
  {"elemwise_add", "add"},
  {"_plus"       , "add"},
  {"_Plus"       , "add"},
  {"_add"        , "add"},
  {"elemwise_sub", "sub"},
  {"_minus"      , "sub"},
  {"_Minus"      , "sub"},
  {"_sub"        , "sub"},
  {"elemwise_mul", "mul"},
  {"_mul"        , "mul"},
  {"_Mul"        , "mul"},
  {"elemwise_div", "div"},
  {"_div"        , "div"},
  {"_Div"        , "div"},
  {"_Power"      , "pow"},
  {"_power"      , "pow"},
  {"_Maximum"    , "max"},
  {"_maximum"    , "max"},
  {"_Minimum"    , "min"},
  {"_minimum"    , "min"}
};

const std::map<std::string, std::string> fused_op_unary_ops = {
  {"amp_cast"                          , "identity"},
  {"relu"                              , "relu"},
  {"sigmoid"                           , "sigmoid"},
  {"softsign"                          , "softsign"},
  {"exp"                               , "exp"},
  {"expm1"                             , "expm1"},
  {"log"                               , "log"},
  {"log10"                             , "log10"},
  {"log2"                              , "log2"},
  {"log1p"                             , "log1p"},
  {"degrees"                           , "degrees"},
  {"radians"                           , "radians"},
  {"sin"                               , "sin"},
  {"cos"                               , "cos"},
  {"tan"                               , "tan"},
  {"arcsin"                            , "arcsin"},
  {"arccos"                            , "arccos"},
  {"arccos"                            , "arccos"},
  {"arctan"                            , "arctan"},
  {"sinh"                              , "sinh"},
  {"cosh"                              , "cosh"},
  {"tanh"                              , "tanh"},
  {"arcsinh"                           , "arcsinh"},
  {"arccosh"                           , "arccosh"},
  {"arctanh"                           , "arctanh"},
  {"sqrt"                              , "sqrt"},
  {"rsqrt"                             , "rsqrt"},
  {"cbrt"                              , "cbrt"},
  {"rcbrt"                             , "rcbrt"},
  {"square"                            , "square"},
  {"squeeze"                           , "identity"},
  {"zeros_like"                        , "zero"},
  {"ones_like"                         , "one"},
  {"flatten"                           , "identity"},
  {"Reshape"                           , "identity"},
  {"reshape"                           , "identity"},
  {"expand_dims"                       , "identity"},
  {"round"                             , "round"},
  {"rint"                              , "rint"},
  {"fix"                               , "fix"},
  {"floor"                             , "floor"},
  {"ceil"                              , "ceil"},
  {"trunc"                             , "trunc"},
  {"sign"                              , "sign"},
  {"reciprocal"                        , "reciprocal"},
  {"abs"                               , "abs"},
  {"gamma"                             , "gamma"},
  {"gammaln"                           , "gammaln"},
  {"erf"                               , "erf"},
  {"erfinv"                            , "erfinv"},
  {"_copy"                             , "identity"},
  {"_identity_with_attr_like_rhs"      , "identity"}
};

const std::map<std::string, std::vector<std::string>> fused_op_special_ops = {
  {"_plus_scalar", {"add(%, %)", "_0", "scalar"}},
  {"_PlusScalar", {"add(%, %)", "_0", "scalar"}},
  {"_minus_scalar", {"sub(%, %)", "_0", "scalar"}},
  {"_MinusScalar", {"sub(%, %)", "_0", "scalar"}},
  {"_rminus_scalar", {"(-sub(%, %))", "_0", "scalar"}},
  {"_RMinusScalar", {"(-sub(%, %))", "_0", "scalar"}},
  {"_mul_scalar", {"mul(%, %)", "_0", "scalar"}},
  {"_MulScalar", {"mul(%, %)", "_0", "scalar"}},
  {"_div_scalar", {"div(%, %)", "_0", "scalar"}},
  {"_DivScalar", {"div(%, %)", "_0", "scalar"}},
  {"_rdiv_scalar", {"rdiv(%, %)", "_0", "scalar"}},
  {"_RDivScalar", {"rdiv(%, %)", "_0", "scalar"}},
  {"Cast", {"cast<%>(%)", "dtype", "_0"}},
  {"cast", {"cast<%>(%)", "dtype", "_0"}},
  {"Activation", {"%(%)", "act_type", "_0"}},
  {"clip", {"clip(%, %, %)", "_0", "a_min", "a_max"}},
  {"_zeros", {"zero<%>(0)", "dtype"}},
  {"_ones", {"one<%>(0)", "dtype"}},
  {"negative", {"(-%)", "_0"}},
  {"_hypot", {"hypot(%, %)", "_0", "_1"}},
  {"_hypot_scalar", {"hypot(%, %)", "_0", "scalar"}},
  {"_backward_relu", {"backward_relu(%, %)", "_1", "_0"}},
  {"_backward_sigmoid", {"backward_sigmoid(%, %)", "_1", "_0"}},
  {"_backward_Activation", {"((% == " + std::to_string(mxnet::op::activation::kReLU) +
                            " || % == " + std::to_string(mxnet::op::activation::kSigmoid) +
                            " || % == " + std::to_string(mxnet::op::activation::kTanh) +
                            ") ? backward_%(%, %) : backward_%(%, %))",
                            "act_type", "act_type", "act_type", "act_type",
                            "_1", "_0", "_2", "_0"}},
  {"_backward_expm1", {"backward_expm1(%, %)", "_1", "_0"}},
  {"_backward_log", {"backward_log(%, %)", "_1", "_0"}},
  {"_backward_log10", {"backward_log10(%, %)", "_1", "_0"}},
  {"_backward_log2", {"backward_log2(%, %)", "_1", "_0"}},
  {"_backward_log1p", {"backward_log1p(%, %)", "_1", "_0"}},
  {"_backward_sin", {"backward_sin(%, %)", "_1", "_0"}},
  {"_backward_cos", {"backward_cos(%, %)", "_1", "_0"}},
  {"_backward_tan", {"backward_tan(%, %)", "_1", "_0"}},
  {"_backward_arcsin", {"backward_arcsin(%, %)", "_1", "_0"}},
  {"_backward_arccos", {"backward_arccos(%, %)", "_1", "_0"}},
  {"_backward_arctan", {"backward_arctan(%, %)", "_1", "_0"}},
  {"_backward_sinh", {"backward_sinh(%, %)", "_1", "_0"}},
  {"_backward_cosh", {"backward_cosh(%, %)", "_1", "_0"}},
  {"_backward_tanh", {"backward_tanh(%, %)", "_1", "_0"}},
  {"_backward_arcsinh", {"backward_arcsinh(%, %)", "_1", "_0"}},
  {"_backward_arccosh", {"backward_arccosh(%, %)", "_1", "_0"}},
  {"_backward_arctanh", {"backward_arctanh(%, %)", "_1", "_0"}},
  {"_backward_sqrt", {"backward_sqrt(%, %)", "_1", "_0"}},
  {"_backward_rsqrt", {"backward_rsqrt(%, %)", "_1", "_0"}},
  {"_backward_cbrt", {"backward_cbrt(%, %)", "_1", "_0"}},
  {"_backward_rcbrt", {"backward_rcbrt(%, %)", "_1", "_0"}},
  {"_backward_square", {"backward_square(%, %)", "_1", "_0"}},
  {"_backward_div_scalar", {"(% / %)", "_0", "scalar"}},
  {"_backward_div_scalar", {"(% / %)", "_0", "scalar"}},
  {"_backward_rdiv_scalar", {"(-% * % / (% * %))", "_0", "scalar", "_1", "_1"}},
  {"_backward_hypot_scalar", {"(% * % / hypot(%, %))", "_0", "_1", "_1", "scalar"}}
  // TODO(ptredak): arange
};

// Multiple inputs/multiple outputs
const std::map<std::string, std::vector<std::vector<std::string>>> fused_op_mimo_ops = {
  {"_backward_sub", {{"(%)", "_0"},
                     {"(-(%))", "_0"}}},
  {"_backward_mul", {{"(% * %)", "_0", "_2"},
                     {"(% * %)", "_0", "_1"}}},
  {"_backward_mul_scalar", {{"(% * %)", "_0", "scalar"}}},
  {"_backward_div", {{"(% / %)", "_0", "_2"},
                     {"(-% * % / (% * %))", "_0", "_1", "_2", "_2"}}},
  {"_backward_power", {{"(% * % * powf(%, % - 1))", "_0", "_2", "_1", "_2"},
                       {"(% * powf(%, %) & logf(%))", "_0", "_1", "_2", "_1"}}},
  {"_backward_power_scalar", {{"(% * % * powf(%, % - 1))", "_0", "scalar", "_1", "scalar"}}},
  {"_backward_rpower_scalar", {{"(% * powf(%, %) & logf(%))", "_0", "scalar", "_2", "scalar"}}},
  {"_backward_maximum", {{"((% > %) ? % : 0)", "_1", "_2", "_0"},
                         {"((% > %) ? 0 : %)", "_1", "_2", "_0"}}},
  {"_backward_minimum", {{"((% < %) ? % : 0)", "_1", "_2", "_0"},
                         {"((% < %) ? 0 : %)", "_1", "_2", "_0"}}},
  {"_backward_hypot", {{"(% * % / hypot(%, %))", "_0", "_1", "_1", "_2"},
                       {"(% * % / hypot(%, %))", "_0", "_2", "_1", "_2"}}}
};

const std::vector<std::string> fused_op_variable_io_ops = {
  "add_n"
};

const std::string fused_op_function_definitions = R"code(
template <typename DType>
struct LoadType {
  using Type = DType;
};

template <>
struct LoadType<half> {
  using Type = float;
};

template <typename DType>
inline typename LoadType<DType>::Type load(const DType * input, int i) {
  return input[i];
}

template <>
inline float load(const half * input, int i) {
  return __half2float(input[i]);
}

template <typename DType>
inline void store(const typename LoadType<DType>::Type value, int i, DType * output) {
  output[i] = value;
}

template <>
inline void store(const float value, int i, half * output) {
  output[i] = __float2half(value);
}

template <typename DType>
inline DType identity(const DType val) {
  return val;
}

template <typename DType, typename DType2>
inline DType add(const DType a, const DType2 b) {
  return a + b;
}

template <typename DType, typename DType2>
inline DType sub(const DType a, const DType2 b) {
  return a - b;
}

template <typename DType, typename DType2>
inline DType mul(const DType a, const DType2 b) {
  return a * b;
}

template <typename DType, typename DType2>
inline DType div(const DType a, const DType2 b) {
  return a / b;
}

template <typename DType, typename DType2>
inline DType rdiv(const DType a, const DType2 b) {
  return b / a;
}

template <typename DType, typename DType2>
inline DType pow(const DType a, const DType2 b) {
  return powf(a, b);
}

template <typename DType, typename DType2>
inline DType max(const DType a, const DType2 b) {
  return a > b ? a : b;
}

template <typename DType, typename DType2>
inline DType min(const DType a, const DType2 b) {
  return a < b ? a : b;
}

template <typename DType, typename DType2>
inline DType hypot(const DType a, const DType2 b) {
  return hypotf(a, b);
}

template <typename OutType, typename DType>
inline typename LoadType<OutType>::Type cast(const DType val) {
  return static_cast<typename LoadType<OutType>::Type>(val);
}

// TODO(ptredak): this is not exactly identity, needs type inference
// in the middle of the graph to do it right
template <typename DType>
inline DType amp_multicast(const DType val) {
  return val;
}

// activations

template <typename DType>
inline DType relu(const DType val) {
  return val > 0 ? val : 0;
}

template <typename DType>
inline DType backward_relu(const DType val, const DType grad) {
  return val > 0 ? grad : 0;
}

template <typename DType>
inline DType sigmoid(const DType val) {
  return 1.f/(1 + expf(-val));
}

template <typename DType>
inline DType backward_sigmoid(const DType out, const DType grad) {
  return grad * out * (1 - out);
}

template <typename DType>
inline DType softrelu(const DType val) {
  return logf(1 + expf(val));
}

template <typename DType>
inline DType backward_softrelu(const DType val, const DType grad) {
  return grad * sigmoid(val);
}

template <typename DType>
inline DType softsign(const DType val) {
  return val / (1 + fabsf(val));
}

template <typename DType>
inline DType backward_softsign(const DType val, const DType grad) {
  const DType ap1 = 1 + fabsf(val);
  return grad / (ap1 * ap1);
}

// exp and log

template <typename DType>
inline DType exp(const DType val) {
  return expf(val);
}

template <typename DType>
inline DType backward_exp(const DType val, const DType grad) {
  return grad * expf(val);
}

template <typename DType>
inline DType expm1(const DType val) {
  return expm1f(val);
}

template <typename DType>
inline DType backward_expm1(const DType val, const DType grad) {
  return grad * expf(val);
}

template <typename DType>
inline DType log(const DType val) {
  return logf(val);
}

template <typename DType>
inline DType backward_log(const DType val, const DType grad) {
  return grad / val;
}

template <typename DType>
inline DType log10(const DType val) {
  return log10f(val);
}

template <typename DType>
inline DType backward_log10(const DType val, const DType grad) {
  return grad / (val * logf(10));
}

template <typename DType>
inline DType log2(const DType val) {
  return log2f(val);
}

template <typename DType>
inline DType backward_log2(const DType val, const DType grad) {
  return grad / (val * logf(2));
}

template <typename DType>
inline DType log1p(const DType val) {
  return log1pf(val);
}

template <typename DType>
inline DType backward_log1p(const DType val, const DType grad) {
  return grad / (1 + val);
}

// trigonometric

constexpr double pi = 3.14159265358979323846;

template <typename DType>
inline DType degrees(const DType val) {
  return (val / pi) * 180;
}

template <typename DType>
inline DType radians(const DType val) {
  return (val / 180.0) * pi;
}

template <typename DType>
inline DType sin(const DType val) {
  return sinf(val);
}

template <typename DType>
inline DType backward_sin(const DType val, const DType grad) {
  return grad * cosf(val);
}

template <typename DType>
inline DType cos(const DType val) {
  return cosf(val);
}

template <typename DType>
inline DType backward_cos(const DType val, const DType grad) {
  return -grad * sinf(val);
}

template <typename DType>
inline DType tan(const DType val) {
  return tanf(val);
}

// Uses output from tan
template <typename DType>
inline DType backward_tan(const DType out, const DType grad) {
  return grad * (out * out + 1);
}

template <typename DType>
inline DType arcsin(const DType val) {
  return asinf(val);
}

template <typename DType>
inline DType backward_arcsin(const DType val, const DType grad) {
  return grad / sqrtf(1 - val*val);
}

template <typename DType>
inline DType arccos(const DType val) {
  return acosf(val);
}

template <typename DType>
inline DType backward_arccos(const DType val, const DType grad) {
  return -grad / sqrtf(1 - val*val);
}

template <typename DType>
inline DType arctan(const DType val) {
  return atanf(val);
}

template <typename DType>
inline DType backward_arctan(const DType val, const DType grad) {
  return grad / (1 + val*val);
}

template <typename DType>
inline DType sinh(const DType val) {
  return sinhf(val);
}

template <typename DType>
inline DType backward_sinh(const DType val, const DType grad) {
  return grad * coshf(val);
}

template <typename DType>
inline DType cosh(const DType val) {
  return coshf(val);
}

template <typename DType>
inline DType backward_cosh(const DType val, const DType grad) {
  return grad * sinhf(val);
}

template <typename DType>
inline DType tanh(const DType val) {
  return tanhf(val);
}

// Uses tanh output
template <typename DType>
inline DType backward_tanh(const DType out, const DType grad) {
  return grad * (1 - out * out);
}

template <typename DType>
inline DType arcsinh(const DType val) {
  return asinhf(val);
}

template <typename DType>
inline DType backward_arcsinh(const DType val, const DType grad) {
  return grad / sqrtf(val * val + 1);
}

template <typename DType>
inline DType arccosh(const DType val) {
  return acoshf(val);
}

template <typename DType>
inline DType backward_arccosh(const DType val, const DType grad) {
  return grad / sqrtf(val * val - 1);
}

template <typename DType>
inline DType arctanh(const DType val) {
  return atanhf(val);
}

template <typename DType>
inline DType backward_arctanh(const DType val, const DType grad) {
  return grad / (1 - val * val);
}

// sqrt

template <typename DType>
inline DType sqrt(const DType val) {
  return sqrtf(val);
}

template <typename DType>
inline DType backward_sqrt(const DType out, const DType grad) {
  return 0.5 * grad / out;
}

template <typename DType>
inline DType rsqrt(const DType val) {
  return rsqrtf(val);
}

template <typename DType>
inline DType backward_rsqrt(const DType val, const DType grad) {
  const DType inv = 1 / val;
  return -0.5 * grad * sqrtf(inv) * inv;
}

template <typename DType>
inline DType cbrt(const DType val) {
  return cbrtf(val);
}

template <typename DType>
inline DType backward_cbrt(const DType out, const DType grad) {
  return grad / (3.0f * out * out);
}

template <typename DType>
inline DType rcbrt(const DType val) {
  return rcbrtf(val);
}

template <typename DType>
inline DType backward_rcbrt(const DType val, const DType grad) {
  const DType inv = 1 / val;
  return -1.f/3.f * grad * cbrtf(inv) * inv;
}

template <typename DType>
inline DType square(const DType val) {
  return val * val;
}

template <typename DType>
inline DType backward_square(const DType val, const DType grad) {
  return 2 * val * grad;
}

template <typename DType>
inline DType zero(const DType val) {
  return 0;
}

template <typename DType>
inline DType one(const DType val) {
  return 1;
}

template <typename DType>
inline DType round(const DType val) {
  return roundf(val);
}

template <typename DType>
inline DType rint(const DType val) {
  return rintf(val);
}

template <typename DType>
inline DType fix(const DType val) {
    const auto floor = floorf(val);
    const auto ceil = ceilf(val);
    return (floor > 0 ? floor : -floor) < (ceil > 0 ? ceil : -ceil) ? floor : ceil;
}

template <typename DType>
inline DType floor(const DType val) {
    return floorf(val);
}

template <typename DType>
inline DType ceil(const DType val) {
    return ceilf(val);
}

template <typename DType>
inline DType trunc(const DType val) {
    return truncf(val);
}

template <typename DType>
inline DType clip(const DType val, const float a_min, const float a_max) {
  return max(min(val, a_max), a_min);
}

template <typename DType>
inline DType sign(const DType val) {
  if (val < 0) return -1;
  return val > 0 ? 1 : 0;
}

template <typename DType>
inline DType reciprocal(const DType val) {
  return 1.0f / val;
}

template <typename DType>
inline DType abs(const DType val) {
  return fabsf(val);
}

template <typename DType>
inline DType gamma(const DType val) {
  return tgammaf(val);
}

template <typename DType>
inline DType gammaln(const DType val) {
  return lgammaf(val);
}

template <typename DType>
inline DType erf(const DType val) {
  return erff(val);
}

template <typename DType>
inline DType erfinv(const DType val) {
  return erfinvf(val);
}

)code";

const std::string fused_op_kernel_begin = R"code(
const int tid = threadIdx.x + blockIdx.x * blockDim.x;
for (int i = tid; i < N; i+= gridDim.x * blockDim.x) {
)code";

const std::string fused_op_kernel_end = R"code(
}
}
)code";

}  // namespace detail

}  // namespace mxnet

#endif  // MXNET_OPERATOR_FUSION_FUSED_OP_INL_H_