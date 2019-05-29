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

#ifndef MXNET_OPERATOR_FUSION_FUSED_OP_H_
#define MXNET_OPERATOR_FUSION_FUSED_OP_H_

#include <mxnet/operator.h>
#include <nnvm/graph.h>
#include <vector>
#include <string>

namespace mxnet {

struct FusedOpConfig : public dmlc::Parameter<FusedOpConfig> {
  std::string symbol_json;
  int num_inputs;
  int num_outputs;
  DMLC_DECLARE_PARAMETER(FusedOpConfig) {
    DMLC_DECLARE_FIELD(symbol_json)
    .describe("JSON of the replaced symbol.");
    DMLC_DECLARE_FIELD(num_inputs)
    .describe("Number of inputs.");
    DMLC_DECLARE_FIELD(num_outputs)
    .describe("Number of outputs.");
  }
};

struct FusedOpEntry {
  FusedOpEntry() : dtype(-1) {}
  int dtype;
};

class FusedOp {
 public:
  static const int NTHREADS = 512;

  explicit FusedOp(const nnvm::NodeAttrs* attrs, const FusedOpConfig& config);
  ~FusedOp() {}
  uint32_t num_inputs() const {
    return inputs_.size();
  }
  uint32_t num_outputs() const {
    return outputs_.size();
  }

  template <typename xpu>
  void Forward(const nnvm::NodeAttrs& attrs,
               const OpContext &ctx,
               const std::vector<TBlob> &inputs,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &outputs);

  template <typename xpu>
  bool InferShape(const nnvm::NodeAttrs &attrs,
                  std::vector<mxnet::TShape> *in_attrs,
                  std::vector<mxnet::TShape> *out_attrs);

  template <typename xpu>
  bool InferType(const nnvm::NodeAttrs &attrs,
                 std::vector<int> *in_attrs,
                 std::vector<int> *out_attrs);

  template <typename Attr>
  std::pair<std::vector<Attr>, std::vector<Attr>> GetAttrs(const std::string& attr_name,
                                                           const uint32_t node_id);

 private:
  void GenerateCode();

  std::vector<FusedOpEntry> inputs_;
  std::vector<FusedOpEntry> outputs_;

  std::string code_;
  nnvm::Graph symbol_;
  std::string ptx_;
  std::string kernel_name_;
  bool initialized_;
  CUfunction kernel_;
  int cc_major_;
  int cc_minor_;
};

using FusedOpPtr = std::shared_ptr<FusedOp>;

struct FusedOpHelperParam {
  FusedOpPtr op;
  uint32_t node_id;

  FusedOpHelperParam(FusedOpPtr op, uint32_t node_id) :
    op(op),
    node_id(node_id) {}
};

using FusedOpHelperParamPtr = std::shared_ptr<FusedOpHelperParam>;

}  // namespace mxnet

#endif  // MXNET_OPERATOR_FUSION_FUSED_OP_H_