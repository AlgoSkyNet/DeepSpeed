# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

hpu_lazy_skip_tests = {}

g1_lazy_skip_tests = {
    "unit/inference/test_human_eval.py::test_human_eval[codellama/CodeLlama-7b-Python-hf]":
    "HPU is not supported on deepspeed-mii",
    "unit/inference/test_inference.py::TestLMCorrectness::test[lambada_standard-gpt2-openai-community/gpt2-xl-False]":
    "Skip workload takes longer time to run",
    "unit/inference/test_inference.py::TestLMCorrectness::test[lambada_standard-gpt2-EleutherAI/gpt-neo-2.7B-False]":
    "Skip workload takes longer time to run",
    "unit/linear/test_ctx.py::TestEngine::test_model": "Skip on G1 due to SW-209651",
    "unit/linear/test_linear.py::TestOptimizedLinear::test[qbit6-bws2]": "Skip on G1 due to SW-209651",
    "unit/linear/test_linear.py::TestOptimizedLinear::test[qbit8-bws2]": "Skip on G1 due to SW-209651",
    "unit/linear/test_linear.py::TestLoRALinear::test[2]": "Skip on G1 due to SW-209651",
    "unit/linear/test_ctx.py::TestInitTransformers::test_pretrained_init": "Skip on G1 due to SW-209651",
    "unit/linear/test_linear.py::TestBasicLinear::test": "Skip on G1 due to SW-209651",
    "unit/linear/test_linear.py::TestOptimizedLinear::test[qbit8-bws1]": "Skip on G1 due to SW-209651",
    "unit/linear/test_ctx.py::TestInitTransformers::test_config_init": "Skip on G1 due to SW-209651",
    "unit/linear/test_linear.py::TestLoRALinear::test[1]": "Skip on G1 due to SW-209651",
    "unit/linear/test_linear.py::TestQuantLinear::test[8]": "Skip on G1 due to SW-209651",
    "unit/linear/test_quant_param.py::TestQuantParam::test_move_to_accelerator": "Skip on G1 due to SW-209651",
    "unit/linear/test_linear.py::TestQuantLinear::test[6]": "Skip on G1 due to SW-209651",
    "unit/linear/test_quant_param.py::TestQuantParam::test_unsupported_dtypes[dtype0]": "Skip on G1 due to SW-209651",
    "unit/linear/test_linear.py::TestOptimizedLinear::test[qbit6-bws1]": "Skip on G1 due to SW-209651",
    "unit/linear/test_quant_param.py::TestQuantParam::test_requires_grad": "Skip on G1 due to SW-209651",
    "unit/linear/test_quant_param.py::TestQuantParam::test_unsupported_dtypes[dtype1]": "Skip on G1 due to SW-209651",
    "unit/linear/test_quant_param.py::TestQuantParam::test_hf_clone": "Skip on G1 due to SW-209651",
    "unit/ops/fp_quantizer/test_fp8_gemm.py::test_fp_quant[2048-qbits8-bf16]": "Skip on G1 due to SW-209651",
    "unit/ops/fp_quantizer/test_fp8_gemm.py::test_fp_quant[64-qbits8-bf16]": "Skip on G1 due to SW-209651",
    "unit/ops/fp_quantizer/test_fp8_gemm.py::test_fp_quant[2-qbits8-bf16]": "Skip on G1 due to SW-209651",
    "unit/ops/fp_quantizer/test_fp8_gemm.py::test_fp_quant[256-qbits8-bf16]": "Skip on G1 due to SW-209651",
    "unit/ops/fp_quantizer/test_fp8_gemm.py::test_fp_quant[1-qbits8-bf16]": "Skip on G1 due to SW-209651",
    "unit/ops/fp_quantizer/test_fp8_gemm.py::test_fp_quant[128-qbits8-bf16]": "Skip on G1 due to SW-209651",
    "unit/ops/fp_quantizer/test_fp8_gemm.py::test_fp_quant[1024-qbits8-bf16]": "Skip on G1 due to SW-209651",
    "unit/ops/fp_quantizer/test_fp8_gemm.py::test_fp_quant[8-qbits8-bf16]": "Skip on G1 due to SW-209651",
    "unit/ops/fp_quantizer/test_fp8_gemm.py::test_fp_quant[32-qbits8-bf16]": "Skip on G1 due to SW-209651",
    "unit/ops/fp_quantizer/test_fp8_gemm.py::test_fp_quant[4-qbits8-bf16]": "Skip on G1 due to SW-209651",
    "unit/ops/fp_quantizer/test_fp8_gemm.py::test_fp_quant[512-qbits8-bf16]": "Skip on G1 due to SW-209651",
    "unit/ops/fp_quantizer/test_fp_quant.py::test_fp_quant_selective[bf16]": "Skip on G1 due to SW-209651",
    "unit/ops/fp_quantizer/test_fp_quant.py::test_fp_quant[qbits12-bf16]": "Skip on G1 due to SW-209651",
    "unit/ops/fp_quantizer/test_fp_quant.py::test_fp_quant_meta[bf16]": "Skip on G1 due to SW-209651",
    "unit/ops/fp_quantizer/test_fp_quant.py::test_fp_quant[qbits6-bf16]": "Skip on G1 due to SW-209651",
    "unit/ops/fp_quantizer/test_fp_quant.py::test_fp_quant[qbits8-bf16]": "Skip on G1 due to SW-209651",
}

g2_lazy_skip_tests = {
    "unit/model_parallelism/test_autotp_training.py::TestSave::test_ckpt_save[4-0]": "Skip, due to stuck",
    "unit/model_parallelism/test_autotp_training.py::TestSave::test_ckpt_save[2-0]": "Skip, due to stuck",
    "unit/inference/test_human_eval.py::test_human_eval[codellama/CodeLlama-7b-Python-hf]":
    "HPU is not supported on deepspeed-mii",
    "unit/inference/test_inference.py::TestLMCorrectness::test[lambada_standard-gpt2-openai-community/gpt2-xl-False]":
    "Skip workload takes longer time to run",
    "unit/ops/fp_quantizer/test_fp_quant.py::test_fp_quant[qbits12-bf16]": "Skip on G1 due to SW-209651",
    "unit/ops/fp_quantizer/test_fp_quant.py::test_fp_quant[qbits6-bf16]": "Skip on G1 due to SW-209651",
    "unit/linear/test_linear.py::TestOptimizedLinear::test[qbit6-bws2]": "Skip on G1 due to SW-209651",
    "unit/linear/test_linear.py::TestOptimizedLinear::test[qbit6-bws1]": "Skip on G1 due to SW-209651",
    "unit/checkpoint/test_moe_checkpoint.py::TestMoECheckpoint::test_checkpoint_moe_and_zero[2-False-False]":
    "skip due to no support in Lazy mode for moe SW-220259",
    "unit/checkpoint/test_moe_checkpoint.py::TestMoECheckpoint::test_checkpoint_moe_and_zero[2-True-False]":
    "skip due to no support in Lazy mode for moe SW-220259",
    "unit/checkpoint/test_moe_checkpoint.py::TestMoECheckpoint::test_checkpoint_moe_and_zero[4-True-False]":
    "skip due to no support in Lazy mode for moe SW-220259",
    "unit/checkpoint/test_moe_checkpoint.py::TestMoECheckpoint::test_checkpoint_moe_and_zero[4-False-False]":
    "skip due to no support in Lazy mode for moe SW-220259",
    "unit/moe/test_moe.py::TestExpertWeightGradWithZero::test[0]":
    "skip due to no support in Lazy mode for moe SW-220259",
    "unit/moe/test_moe.py::TestSimpleMoE::test[2]": "skip due to no support in Lazy mode for moe SW-220259",
    "unit/moe/test_moe.py::TestSimpleMoE::test[1]": "skip due to no support in Lazy mode for moe SW-220259",
    "unit/moe/test_moe.py::TestExpertWeightGradWithZero::test[1]":
    "skip due to no support in Lazy mode for moe SW-220259",
    "unit/moe/test_moe.py::TestExpertWeightGradWithZero::test[2]":
    "skip due to no support in Lazy mode for moe SW-220259",
    "unit/moe/test_moe.py::TestMoE::test[False-2-4]": "skip due to no support in Lazy mode for moe SW-220259",
    "unit/moe/test_moe.py::TestMoE::test[False-2-2]": "skip due to no support in Lazy mode for moe SW-220259",
    "unit/moe/test_moe.py::TestMoE::test[False-1-2]": "skip due to no support in Lazy mode for moe SW-220259",
    "unit/moe/test_moe.py::TestMoE::test[False-1-4]": "skip due to no support in Lazy mode for moe SW-220259",
    "unit/moe/test_moe.py::TestMoE::test[True-2-2]": "skip due to no support in Lazy mode for moe SW-220259",
    "unit/moe/test_moe.py::TestMoE::test[True-2-4]": "skip due to no support in Lazy mode for moe SW-220259",
    "unit/moe/test_moe.py::TestMoE::test[True-1-4]": "skip due to no support in Lazy mode for moe SW-220259",
    "unit/moe/test_moe.py::TestMoE::test[True-1-2]": "skip due to no support in Lazy mode for moe SW-220259",
    "unit/ops/fp_quantizer/test_fp_quant.py::test_fp_quant_selective[bf16]": "Skip, due to SW-209650",
}

g3_lazy_skip_tests = {
    "unit/inference/test_inference.py::TestLMCorrectness::test[lambada_standard-gpt2-openai-community/gpt2-xl-False]":
    "Skip workload takes longer time to run",
    "unit/inference/test_inference.py::TestLMCorrectness::test[lambada_standard-gpt2-EleutherAI/gpt-neo-2.7B-False]":
    "Skip workload takes longer time to run",
    "unit/runtime/zero/test_zeropp.py::TestZeroPPConfigSweep::test[4-9-1024]": "test hang patch:430071",
    "unit/runtime/zero/test_zeropp.py::TestZeroPPConfigSweep::test_gradient_accumulation[4-9-1024]":
    "test hang patch:430071",
    "unit/runtime/zero/test_zeropp.py::TestZeroPPConfigSweep::test_eval[4-9-1024]": "test hang patch:430071",
}
hpu_eager_skip_tests = {}

g1_eager_skip_tests = {
    "unit/inference/test_inference.py::TestMPSize::test[fp32-gpt-neo-True]":
    "Flaky Segfault. Stuck",
    "unit/inference/test_inference.py::TestMPSize::test[fp32-gpt-neo-False]":
    "Flaky Segfault. Stuck",
    "unit/inference/test_human_eval.py::test_human_eval[codellama/CodeLlama-7b-Python-hf]":
    "HPU is not supported on deepspeed-mii",
    "unit/inference/test_inference.py::TestLMCorrectness::test[lambada_standard-gpt2-openai-community/gpt2-xl-False]":
    "Skip workload takes longer time to run",
    "unit/inference/test_inference.py::TestLMCorrectness::test[lambada_standard-gpt2-EleutherAI/gpt-neo-2.7B-False]":
    "Skip workload takes longer time to run",
    "unit/inference/test_inference.py::TestLMCorrectness::test[lambada_standard-gpt2-openai-community/gpt2-xl-True]":
    "Skip workload takes longer time to run",
    "unit/inference/test_inference.py::TestLMCorrectness::test[lambada_standard-gpt2-EleutherAI/gpt-neo-2.7B-True]":
    "Skip workload takes longer time to run",
    "unit/inference/test_inference.py::TestAutoTensorParallelism::test_odd_world_size[bf16-marian-False-False]":
    "Struck observed",
    "unit/inference/test_inference.py::TestAutoTensorParallelism::test[bf16-marian-False-False]":
    "Flaky struck observed",
    "unit/inference/test_inference.py::TestModelTask::test[facebook/opt-125m-text-generation-fp16-noCG-noTriton-True-True]":
    "Skip due to SW-193097",
    "unit/inference/test_inference.py::TestModelTask::test[facebook/opt-125m-text-generation-fp32-noCG-noTriton-True-True]":
    "Skip due to SW-193097",
    "unit/inference/test_inference.py::TestModelTask::test[Norod78/hebrew-bad_wiki-gpt_neo-tiny-text-generation-fp32-noCG-noTriton-True-True]":
    "Skip due to SW-193097",
    "unit/inference/test_inference.py::TestModelTask::test[distilbert/distilgpt2-text-generation-fp32-noCG-noTriton-True-True]":
    "Skip due to SW-193097",
    "unit/inference/test_inference.py::TestModelTask::test[Norod78/hebrew-bad_wiki-gpt_neo-tiny-text-generation-fp16-noCG-noTriton-True-True]":
    "Skip due to SW-193097",
    "unit/inference/test_inference.py::TestModelTask::test[openai-community/gpt2-text-generation-fp32-noCG-noTriton-True-True]":
    "Skip due to SW-193097",
    "unit/inference/test_inference.py::TestLowCpuMemUsage::test[gpt2-True]":
    "Skip struck for longer duration",
    "unit/inference/test_inference.py::TestAutoTensorParallelism::test[fp16-marian-True-True]":
    "Skip struck and fp16 not supported",
    "unit/inference/test_inference.py::TestModelTask::test[openai-community/gpt2-text-generation-fp16-noCG-noTriton-True-True]":
    "Skip due to SW-193097",
    "unit/runtime/pipe/test_pipe.py::TestPipeCifar10::test_pipe_use_reentrant[topo_config1]":
    "Test Hang",
    "unit/runtime/pipe/test_pipe.py::TestPipeCifar10::test_pipe_use_reentrant[topo_config2]":
    "Test Hang",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp16-noCG-noTriton-True-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp16-noCG-noTriton-False-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp16-noCG-Triton-True-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp32-noCG-noTriton-True-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp32-noCG-Triton-False-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp16-CG-noTriton-True-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp32-CG-Triton-False-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp16-CG-Triton-True-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp16-noCG-Triton-False-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp32-CG-noTriton-False-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp32-CG-Triton-True-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp32-CG-noTriton-True-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp32-noCG-Triton-True-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp32-noCG-noTriton-False-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp16-CG-Triton-False-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp16-CG-noTriton-False-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp16-noCG-noTriton-False-True]":
    "Test Hang",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp16-noCG-noTriton-True-True]":
    "Test Hang",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp32-CG-Triton-True-True]":
    "Test Hang",
    "unit/inference/test_inference.py::TestAutoTensorParallelism::test_odd_world_size[bf16-marian-True-True]":
    "test Hang",
    "unit/inference/test_checkpoint_sharding.py::TestCheckpointShardinAutoTP::test[facebook/opt-350m-True]":
    "test Hang",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to4[False-True-False-1-dtype0]":
    "test Hang",
    "unit/inference/test_checkpoint_sharding.py::TestCheckpointShard::test[EleutherAI/gpt-j-6B-fp16-True]":
    "Skip due to SW-193097",
    "unit/inference/test_inference.py::TestAutoTensorParallelism::test[bf16-marian-True-True]":
    "Skip due to flaky hang",
    "unit/inference/test_checkpoint_sharding.py::TestCheckpointShardinAutoTP::test[EleutherAI/gpt-j-6B-True]":
    "test Hang",
    "unit/linear/test_ctx.py::TestEngine::test_model":
    "Skip on G1 due to SW-209651",
    "unit/linear/test_linear.py::TestOptimizedLinear::test[qbit6-bws2]":
    "Skip on G1 due to SW-209651",
    "unit/linear/test_linear.py::TestOptimizedLinear::test[qbit8-bws2]":
    "Skip on G1 due to SW-209651",
    "unit/linear/test_linear.py::TestLoRALinear::test[2]":
    "Skip on G1 due to SW-209651",
    "unit/linear/test_ctx.py::TestInitTransformers::test_pretrained_init":
    "Skip on G1 due to SW-209651",
    "unit/linear/test_linear.py::TestBasicLinear::test":
    "Skip on G1 due to SW-209651",
    "unit/linear/test_linear.py::TestOptimizedLinear::test[qbit8-bws1]":
    "Skip on G1 due to SW-209651",
    "unit/linear/test_ctx.py::TestInitTransformers::test_config_init":
    "Skip on G1 due to SW-209651",
    "unit/linear/test_linear.py::TestLoRALinear::test[1]":
    "Skip on G1 due to SW-209651",
    "unit/linear/test_linear.py::TestQuantLinear::test[8]":
    "Skip on G1 due to SW-209651",
    "unit/linear/test_quant_param.py::TestQuantParam::test_move_to_accelerator":
    "Skip on G1 due to SW-209651",
    "unit/linear/test_linear.py::TestQuantLinear::test[6]":
    "Skip on G1 due to SW-209651",
    "unit/linear/test_quant_param.py::TestQuantParam::test_unsupported_dtypes[dtype0]":
    "Skip on G1 due to SW-209651",
    "unit/linear/test_linear.py::TestOptimizedLinear::test[qbit6-bws1]":
    "Skip on G1 due to SW-209651",
    "unit/linear/test_quant_param.py::TestQuantParam::test_requires_grad":
    "Skip on G1 due to SW-209651",
    "unit/linear/test_quant_param.py::TestQuantParam::test_unsupported_dtypes[dtype1]":
    "Skip on G1 due to SW-209651",
    "unit/linear/test_quant_param.py::TestQuantParam::test_hf_clone":
    "Skip on G1 due to SW-209651",
    "unit/ops/fp_quantizer/test_fp8_gemm.py::test_fp_quant[2048-qbits8-bf16]":
    "Skip on G1 due to SW-209651",
    "unit/ops/fp_quantizer/test_fp8_gemm.py::test_fp_quant[64-qbits8-bf16]":
    "Skip on G1 due to SW-209651",
    "unit/ops/fp_quantizer/test_fp8_gemm.py::test_fp_quant[2-qbits8-bf16]":
    "Skip on G1 due to SW-209651",
    "unit/ops/fp_quantizer/test_fp8_gemm.py::test_fp_quant[256-qbits8-bf16]":
    "Skip on G1 due to SW-209651",
    "unit/ops/fp_quantizer/test_fp8_gemm.py::test_fp_quant[1-qbits8-bf16]":
    "Skip on G1 due to SW-209651",
    "unit/ops/fp_quantizer/test_fp8_gemm.py::test_fp_quant[128-qbits8-bf16]":
    "Skip on G1 due to SW-209651",
    "unit/ops/fp_quantizer/test_fp8_gemm.py::test_fp_quant[1024-qbits8-bf16]":
    "Skip on G1 due to SW-209651",
    "unit/ops/fp_quantizer/test_fp8_gemm.py::test_fp_quant[8-qbits8-bf16]":
    "Skip on G1 due to SW-209651",
    "unit/ops/fp_quantizer/test_fp8_gemm.py::test_fp_quant[32-qbits8-bf16]":
    "Skip on G1 due to SW-209651",
    "unit/ops/fp_quantizer/test_fp8_gemm.py::test_fp_quant[4-qbits8-bf16]":
    "Skip on G1 due to SW-209651",
    "unit/ops/fp_quantizer/test_fp8_gemm.py::test_fp_quant[512-qbits8-bf16]":
    "Skip on G1 due to SW-209651",
    "unit/ops/fp_quantizer/test_fp_quant.py::test_fp_quant_selective[bf16]":
    "Skip on G1 due to SW-209651",
    "unit/ops/fp_quantizer/test_fp_quant.py::test_fp_quant[qbits12-bf16]":
    "Skip on G1 due to SW-209651",
    "unit/ops/fp_quantizer/test_fp_quant.py::test_fp_quant_meta[bf16]":
    "Skip on G1 due to SW-209651",
    "unit/ops/fp_quantizer/test_fp_quant.py::test_fp_quant[qbits6-bf16]":
    "Skip on G1 due to SW-209651",
    "unit/ops/fp_quantizer/test_fp_quant.py::test_fp_quant[qbits8-bf16]":
    "Skip on G1 due to SW-209651",
}

g2_eager_skip_tests = {
    "unit/inference/test_human_eval.py::test_human_eval[codellama/CodeLlama-7b-Python-hf]":
    "HPU is not supported on deepspeed-mii",
    "unit/inference/test_inference.py::TestLMCorrectness::test[lambada_standard-gpt2-openai-community/gpt2-xl-True]":
    "Skip workload takes longer time to run",
    "unit/inference/test_inference.py::TestLMCorrectness::test[lambada_standard-gpt2-EleutherAI/gpt-neo-2.7B-True]":
    "Skip workload takes longer time to run",
    "unit/inference/test_inference.py::TestLowCpuMemUsage::test[gpt2-True]":
    "Skip struck for longer duration",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp16-noCG-noTriton-True-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp16-noCG-noTriton-False-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp16-noCG-Triton-True-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp32-noCG-noTriton-True-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp32-noCG-Triton-False-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp16-CG-noTriton-True-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp32-CG-Triton-False-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp16-CG-Triton-True-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp16-noCG-Triton-False-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp32-CG-noTriton-False-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp32-CG-Triton-True-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp32-CG-noTriton-True-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp32-noCG-Triton-True-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp32-noCG-noTriton-False-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp16-CG-Triton-False-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp16-CG-noTriton-False-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestMPSize::test[fp16-gpt-neo-True]":
    "GC failed so skip to check",
    "unit/inference/test_checkpoint_sharding.py::TestCheckpointShardinAutoTP::test[bigscience/bloom-560m-fp16-True]":
    "test Hang",
    "unit/inference/test_checkpoint_sharding.py::TestCheckpointShard::test[EleutherAI/gpt-j-6B-fp16-True]":
    "Skip due to SW-193097",
    "unit/inference/test_checkpoint_sharding.py::TestCheckpointShardinAutoTP::test[EleutherAI/gpt-j-6B-True]":
    "test Hang",
    "unit/ops/fp_quantizer/test_fp_quant.py::test_fp_quant[qbits12-bf16]":
    "Skip on G1 due to SW-209651",
    "unit/ops/fp_quantizer/test_fp_quant.py::test_fp_quant[qbits6-bf16]":
    "Skip on G1 due to SW-209651",
    "unit/linear/test_linear.py::TestOptimizedLinear::test[qbit6-bws2]":
    "Skip on G1 due to SW-209651",
    "unit/linear/test_linear.py::TestOptimizedLinear::test[qbit6-bws1]":
    "Skip on G1 due to SW-209651",
    "unit/inference/test_inference.py::TestAutoTensorParallelism::test[fp16-marian-True-True]":
    "Skip struck and fp16 not supported",
    "unit/runtime/pipe/test_pipe.py::TestPipeCifar10::test_pipe_use_reentrant[topo_config2]":
    "Test Hang",
    "unit/ops/fp_quantizer/test_fp_quant.py::test_fp_quant_selective[bf16]":
    "Skip, due to SW-209648",
}
g3_eager_skip_tests = {
    "unit/inference/test_inference.py::TestLMCorrectness::test[lambada_standard-gpt2-openai-community/gpt2-xl-False]":
    "Skip workload takes longer time to run",
    "unit/inference/test_inference.py::TestLMCorrectness::test[lambada_standard-gpt2-EleutherAI/gpt-neo-2.7B-False]":
    "Skip workload takes longer time to run",
    "unit/inference/test_inference.py::TestLMCorrectness::test[lambada_standard-gpt2-openai-community/gpt2-xl-True]":
    "Skip workload takes longer time to run",
    "unit/inference/test_inference.py::TestLMCorrectness::test[lambada_standard-gpt2-EleutherAI/gpt-neo-2.7B-True]":
    "Skip workload takes longer time to run",
    "unit/inference/test_inference.py::TestLowCpuMemUsage::test[gpt2-True]":
    "Skip struck for longer duration",
    "unit/inference/test_inference.py::TestModelTask::test[distilbert/distilgpt2-text-generation-fp16-noCG-noTriton-True-True]":
    "Skip struck for longer duration",
    "unit/runtime/zero/test_zeropp.py::TestZeroPPConfigSweep::test[4-9-1024]":
    "test hang patch:430071",
    "unit/runtime/zero/test_zeropp.py::TestZeroPPConfigSweep::test_gradient_accumulation[4-9-1024]":
    "test hang patch:430071",
    "unit/runtime/zero/test_zeropp.py::TestZeroPPConfigSweep::test_eval[4-9-1024]":
    "test hang patch:430071",
    "unit/inference/test_checkpoint_sharding.py::TestCheckpointShard::test[EleutherAI/gpt-j-6B-fp16-True]":
    "Skip due to SW-193097",
    "unit/inference/test_checkpoint_sharding.py::TestCheckpointShard::test[bigscience/bloom-560m-fp16-True]":
    "Skip due to SW-193097",
    "unit/inference/test_inference.py::TestModelTask::test[facebook/opt-125m-text-generation-fp16-noCG-noTriton-True-True]":
    "Skip due to SW-193097",
    "unit/inference/test_inference.py::TestModelTask::test[facebook/opt-125m-text-generation-fp32-noCG-noTriton-True-True]":
    "Skip due to SW-193097",
    "unit/inference/test_inference.py::TestModelTask::test[Norod78/hebrew-bad_wiki-gpt_neo-tiny-text-generation-fp16-noCG-noTriton-True-True]":
    "Skip due to SW-193097",
    "unit/inference/test_inference.py::TestModelTask::test[openai-community/gpt2-text-generation-fp32-noCG-noTriton-True-True]":
    "Skip due to SW-193097",
    "unit/inference/test_inference.py::TestModelTask::test[Norod78/hebrew-bad_wiki-gpt_neo-tiny-text-generation-fp32-noCG-noTriton-True-True]":
    "Skip due to SW-193097",
    "unit/inference/test_inference.py::TestModelTask::test[distilbert/distilgpt2-text-generation-fp32-noCG-noTriton-True-True]":
    "Skip due to SW-193097",
    "unit/inference/test_inference.py::TestModelTask::test[openai-community/gpt2-text-generation-fp16-noCG-noTriton-True-True]":
    "Skip due to SW-193097",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp16-noCG-noTriton-True-True]":
    "GC failed so skip to check",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp16-noCG-noTriton-False-True]":
    "GC failed so skip to check",
    "unit/inference/test_inference.py::TestMPSize::test[fp16-gpt-neo-True]":
    "GC failed so skip to check",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp16-noCG-noTriton-True-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp16-noCG-noTriton-False-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp16-noCG-Triton-True-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp32-noCG-noTriton-True-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp32-noCG-Triton-False-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp16-CG-noTriton-True-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp32-CG-Triton-False-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp16-CG-Triton-True-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp16-noCG-Triton-False-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp32-CG-noTriton-False-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp32-CG-Triton-True-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp32-CG-noTriton-True-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp32-noCG-Triton-True-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp32-noCG-noTriton-False-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp16-CG-Triton-False-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp16-CG-noTriton-False-False]":
    "Skip bloom due to process struck and also fail",
    "unit/inference/test_inference.py::TestMPSize::test[fp32-gpt-neo-True]":
    "Flaky Segfault. Stuck",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp32-CG-Triton-True-True]":
    "Test Hang",
    "unit/inference/test_checkpoint_sharding.py::TestCheckpointShardinAutoTP::test[facebook/opt-350m-True]":
    "test Hang",
    "unit/inference/test_checkpoint_sharding.py::TestCheckpointShardinAutoTP::test[EleutherAI/gpt-j-6B-True]":
    "test Hang",
    "unit/inference/test_checkpoint_sharding.py::TestCheckpointShardinAutoTP::test[bigscience/bloom-560m-True]":
    "test Hang",
}

gpu_skip_tests = {
    "unit/runtime/zero/test_zero.py::TestZeroOffloadOptim::test[True]":
    "Disabled as it is causing test to stuck. SW-163517.",
    "unit/inference/test_stable_diffusion.py::TestStableDiffusion::test":
    "Xfail not supported",
    "unit/inference/test_inference.py::TestLMCorrectness::test[lambada_standard-gpt2-openai-community/gpt2-xl-False]":
    "skip: timeout triggered",
    "unit/inference/test_inference.py::TestLMCorrectness::test[lambada_standard-gpt2-EleutherAI/gpt-neo-2.7B-False]":
    "skip: timeout triggered",
    "unit/runtime/zero/test_zero_tensor_fragment.py::TestTensorFragmentGet::test_bf16_fragments[False]":
    "Skip due to hang",
    "unit/runtime/zero/test_zero_tensor_fragment.py::TestTensorFragmentUpdate::test_zero_fragments[none-3-full-dtype0]":
    "skip due to hang",
    "unit/model_parallelism/test_autotp_training.py::TestParamsGather::test[linearallreduce]":
    "Skip due to SW-237966",
    "unit/model_parallelism/test_autotp_training.py::TestParamsGather::test[linear]":
    "Skip due to SW-237966",
    "unit/model_parallelism/test_autotp_training.py::TestTpGradNorm::test[4-2]":
    "Skip due to SW-237966",
    "unit/model_parallelism/test_autotp_training.py::TestTpGradNorm::test[2-0]":
    "Skip due to SW-237966",
    "unit/model_parallelism/test_autotp_training.py::TestTpGradNorm::test[2-2]":
    "Skip due to SW-237966",
    "unit/model_parallelism/test_autotp_training.py::TestTpGradNorm::test[2-1]":
    "Skip due to SW-237966",
    "unit/model_parallelism/test_autotp_training.py::TestTpGradNorm::test[4-1]":
    "Skip due to SW-237966",
    "unit/model_parallelism/test_autotp_training.py::TestTpGradNorm::test[4-0]":
    "Skip due to SW-237966",
    "unit/model_parallelism/test_autotp_training.py::TestSave::test_ckpt_save[4-1]":
    "Skip due to SW-237966",
    "unit/model_parallelism/test_autotp_training.py::TestSave::test_ckpt_save[2-0]":
    "Skip due to SW-237966",
    "unit/model_parallelism/test_autotp_training.py::TestSave::test_save_original_weight[2-1]":
    "Skip due to SW-237966",
    "unit/model_parallelism/test_autotp_training.py::TestSave::test_ckpt_save[2-2]":
    "Skip due to SW-237966",
    "unit/model_parallelism/test_autotp_training.py::TestSave::test_ckpt_save[2-1]":
    "Skip due to SW-237966",
    "unit/model_parallelism/test_autotp_training.py::TestSave::test_save_original_weight[4-0]":
    "Skip due to SW-237966",
    "unit/model_parallelism/test_autotp_training.py::TestSave::test_save_original_weight[2-0]":
    "Skip due to SW-237966",
    "unit/model_parallelism/test_autotp_training.py::TestSave::test_save_original_weight[2-2]":
    "Skip due to SW-237966",
    "unit/model_parallelism/test_autotp_training.py::TestSave::test_ckpt_save[4-0]":
    "Skip due to SW-237966",
    "unit/model_parallelism/test_autotp_training.py::TestSave::test_save_original_weight[4-1]":
    "Skip due to SW-237966",
    "unit/model_parallelism/test_autotp_training.py::TestSave::test_save_original_weight[4-2]":
    "Skip due to SW-237966",
    "unit/model_parallelism/test_autotp_training.py::TestSave::test_ckpt_save[4-2]":
    "Skip due to SW-237966",
    "unit/model_parallelism/test_autotp_training.py::TestTpDataloaderCorrectness::test[2]":
    "Skip due to SW-237966",
    "unit/model_parallelism/test_autotp_training.py::TestTpDataloaderCorrectness::test[4]":
    "Skip due to SW-237966",
    "unit/model_parallelism/test_autotp_training.py::TestTpParallelStates::test[2]":
    "Skip due to SW-237966",
    "unit/model_parallelism/test_autotp_training.py::TestTpParallelStates::test[4]":
    "Skip due to SW-237966",
    "unit/inference/quantization/test_intX_quantization.py::TestQuantizedInt::test_zero3_int4_quantized_initialization_cpu_offload[8bits]":
    "Skip due to SW-237966",
    "unit/inference/quantization/test_intX_quantization.py::TestQuantizedInt::test_zero3_int4_quantized_initialization[4bits]":
    "Skip due to SW-237966",
    "unit/inference/quantization/test_intX_quantization.py::TestQuantizedInt::test_zero3_int4_post_init_quant[4bits]":
    "Skip due to SW-237966",
    "unit/inference/quantization/test_intX_quantization.py::TestQuantizedInt::test_zero3_int4_post_init_quant_cpu_offload[4bits]":
    "Skip due to SW-237966",
    "unit/inference/quantization/test_intX_quantization.py::TestQuantizedInt::test_zero3_int4_post_init_quant[8bits]":
    "Skip due to SW-237966",
    "unit/inference/quantization/test_intX_quantization.py::TestQuantizedInt::test_zero3_int4_post_init_quant_cpu_offload[8bits]":
    "Skip due to SW-237966",
    "unit/inference/quantization/test_intX_quantization.py::TestQuantizedInt::test_zero3_int4_quantized_initialization[8bits]":
    "Skip due to SW-237966",
    "unit/inference/quantization/test_intX_quantization.py::TestQuantizedInt::test_zero3_int4_quantized_initialization_cpu_offload[4bits]":
    "Skip due to SW-237966",
    "unit/runtime/zero/test_zero_nesting_init.py::TestNestedParallelInit::test_nested_parallel_init":
    "Skip due to SW-237966",
    "unit/inference/test_inference.py::TestModelTask::test[bigscience/bloom-560m-text-generation-fp16-noCG-noTriton-True-False]":
    "Stuck",
    "unit/model_parallelism/test_autotp_training.py::TestTpLayerFwdBwd::testRowParallel[True-4]":
    "Skip due to SW-237966",
    "unit/model_parallelism/test_autotp_training.py::TestTpLayerFwdBwd::testRowParallel[False-4]":
    "Skip due to SW-237966",
    "unit/model_parallelism/test_autotp_training.py::TestTpLayerFwdBwd::testColumnParallel[False-4]":
    "Skip due to SW-237966",
    "unit/model_parallelism/test_autotp_training.py::TestTpLayerFwdBwd::testColumnParallel[True-2]":
    "Skip due to SW-237966",
    "unit/model_parallelism/test_autotp_training.py::TestTpLayerFwdBwd::testColumnParallel[False-2]":
    "Skip due to SW-237966",
    "unit/model_parallelism/test_autotp_training.py::TestTpLayerFwdBwd::testRowParallel[False-2]":
    "Skip due to SW-237966",
    "unit/model_parallelism/test_autotp_training.py::TestTpLayerFwdBwd::testColumnParallel[True-4]":
    "Skip due to SW-237966",
    "unit/model_parallelism/test_autotp_training.py::TestTpLayerFwdBwd::testRowParallel[True-2]":
    "Skip due to SW-237966",
    "unit/inference/test_inference.py::TestModelTask::test[j-hartmann/emotion-english-distilroberta-base-text-classification-fp32-CG-noTriton-True-False]":
    "Current PT version 2.2 does not supported. Require min 2.6",
    "unit/inference/test_inference.py::TestModelTask::test[j-hartmann/emotion-english-distilroberta-base-text-classification-fp32-CG-noTriton-False-False]":
    "Current PT version 2.2 does not supported. Require min 2.6",
    "unit/inference/test_inference.py::TestModelTask::test[j-hartmann/emotion-english-distilroberta-base-text-classification-fp16-CG-noTriton-True-False]":
    "Current PT version 2.2 does not supported. Require min 2.6",
    "unit/inference/test_inference.py::TestModelTask::test[j-hartmann/emotion-english-distilroberta-base-text-classification-fp16-noCG-noTriton-True-False]":
    "Current PT version 2.2 does not supported. Require min 2.6",
    "unit/inference/test_inference.py::TestModelTask::test[facebook/opt-125m-text-generation-fp32-noCG-noTriton-True-False]":
    "Current PT version 2.2 does not supported. Require min 2.6",
    "unit/inference/test_inference.py::TestModelTask::test[facebook/opt-350m-text-generation-fp32-noCG-noTriton-True-False]":
    "Current PT version 2.2 does not supported. Require min 2.6",
    "unit/inference/test_inference.py::TestModelTask::test[facebook/opt-350m-text-generation-fp32-noCG-noTriton-False-False]":
    "Current PT version 2.2 does not supported. Require min 2.6",
    "unit/inference/test_inference.py::TestModelTask::test[j-hartmann/emotion-english-distilroberta-base-text-classification-fp32-noCG-noTriton-True-False]":
    "Current PT version 2.2 does not supported. Require min 2.6",
    "unit/inference/test_inference.py::TestModelTask::test[j-hartmann/emotion-english-distilroberta-base-text-classification-fp16-noCG-noTriton-False-False]":
    "Current PT version 2.2 does not supported. Require min 2.6",
    "unit/inference/test_inference.py::TestModelTask::test[facebook/opt-350m-text-generation-fp16-noCG-noTriton-True-False]":
    "Current PT version 2.2 does not supported. Require min 2.6",
    "unit/inference/test_inference.py::TestModelTask::test[j-hartmann/emotion-english-distilroberta-base-text-classification-fp16-CG-noTriton-False-False]":
    "Current PT version 2.2 does not supported. Require min 2.6",
    "unit/inference/test_inference.py::TestModelTask::test[facebook/opt-125m-text-generation-fp16-noCG-noTriton-True-False]":
    "Current PT version 2.2 does not supported. Require min 2.6",
    "unit/inference/test_inference.py::TestModelTask::test[j-hartmann/emotion-english-distilroberta-base-text-classification-fp32-noCG-noTriton-False-False]":
    "Current PT version 2.2 does not supported. Require min 2.6",
    "unit/inference/test_inference.py::TestModelTask::test[facebook/opt-125m-text-generation-fp16-noCG-noTriton-False-False]":
    "Current PT version 2.2 does not supported. Require min 2.6",
    "unit/inference/test_inference.py::TestModelTask::test[facebook/opt-125m-text-generation-fp32-noCG-noTriton-False-False]":
    "Current PT version 2.2 does not supported. Require min 2.6",
    "unit/inference/test_inference.py::TestModelTask::test[facebook/opt-350m-text-generation-fp16-noCG-noTriton-False-False]":
    "Current PT version 2.2 does not supported. Require min 2.6",
    "unit/inference/test_inference.py::TestAutoTensorParallelism::test_odd_world_size[fp16-marian-False-False]":
    "Current PT version 2.2 does not supported. Require min 2.6",
    "unit/inference/test_inference.py::TestAutoTensorParallelism::test[fp16-marian-False-False]":
    "Current PT version 2.2 does not supported. Require min 2.6",
    "unit/inference/test_inference.py::TestInjectionPolicy::test[ws2-fp32-t5-False]":
    "Current PT version 2.2 does not supported. Require min 2.6",
    "unit/inference/test_inference.py::TestInjectionPolicy::test[ws1-fp32-t5-False]":
    "Current PT version 2.2 does not supported. Require min 2.6",
    "unit/inference/test_checkpoint_sharding.py::TestCheckpointShard::test[EleutherAI/gpt-j-6B-fp16-False]":
    "Big test",
    "unit/comm/test_dist.py::TestDistributedFixture::test[4-32]":
    "Skip due to SW-237969",
    "unit/comm/test_dist.py::TestDistributedFixture::test[4-16]":
    "Skip due to SW-237969",
    "unit/comm/test_dist.py::TestDistributedFixture::test[2-32]":
    "Skip due to SW-237969",
    "unit/comm/test_dist.py::TestDistributedFixture::test[2-16]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_4to2[False-True-True-3-dtype2]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to4[False-True-True-1-dtype2]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to2[False-True-True-3-dtype2]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_4to2[False-False-True-1-dtype1]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to2[False-False-False-3-dtype2]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to4[False-True-False-3-dtype2]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to4[False-False-False-3-dtype0]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_4to2[False-False-False-1-dtype2]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to4[False-False-False-1-dtype0]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to2[False-False-False-1-dtype0]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to4[False-True-False-1-dtype0]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_4to2[False-False-True-1-dtype2]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to2[False-False-True-3-dtype2]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to2[False-False-True-3-dtype1]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to2[False-True-True-3-dtype0]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to2[False-False-False-3-dtype0]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to2[False-False-False-1-dtype2]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_4to2[False-True-True-1-dtype2]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_4to2[False-True-True-1-dtype0]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to4[False-False-True-3-dtype0]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to4[False-False-False-1-dtype1]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to2[False-True-True-1-dtype2]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to4[False-False-False-1-dtype2]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to4[False-True-False-3-dtype1]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_4to2[False-False-True-3-dtype1]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to2[False-True-True-1-dtype1]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to4[False-False-True-1-dtype1]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_4to2[False-True-True-3-dtype0]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to4[False-True-True-3-dtype0]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to4[False-True-False-1-dtype1]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to2[False-True-False-1-dtype0]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_4to2[False-False-False-3-dtype2]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_4to2[False-False-True-3-dtype0]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to2[False-True-False-1-dtype2]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to4[False-True-True-3-dtype1]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to2[False-False-False-3-dtype1]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_4to2[False-False-False-1-dtype1]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_4to2[False-False-False-3-dtype0]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to4[False-False-True-1-dtype2]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to2[False-True-False-3-dtype2]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to2[False-True-False-3-dtype1]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to2[False-True-True-1-dtype0]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to4[False-True-False-1-dtype2]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_4to2[False-True-False-1-dtype2]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to2[False-True-False-1-dtype1]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to4[False-False-False-3-dtype1]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to2[False-False-True-1-dtype1]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to2[False-False-True-1-dtype2]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_4to2[False-False-True-1-dtype0]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_4to2[False-False-False-3-dtype1]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to4[False-False-False-3-dtype2]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to2[False-False-True-1-dtype0]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to2[False-True-False-3-dtype0]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to4[False-True-True-1-dtype0]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to4[False-False-True-3-dtype2]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_4to2[False-True-False-3-dtype1]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to2[False-True-True-3-dtype1]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_4to2[False-True-False-3-dtype0]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_4to2[False-True-True-3-dtype1]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to4[False-False-True-3-dtype1]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_4to2[False-True-False-1-dtype1]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to4[False-False-True-1-dtype0]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to4[False-True-True-1-dtype1]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_4to2[False-False-False-1-dtype0]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to4[False-True-False-3-dtype0]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_4to2[False-False-True-3-dtype2]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_4to2[False-True-False-3-dtype2]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to4[False-True-True-3-dtype2]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to2[False-False-False-1-dtype1]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_2to2[False-False-True-3-dtype0]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_4to2[False-True-True-1-dtype1]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_universal_checkpoint.py::TestZeROUniversalCheckpointDP::test_dp_world_size_4to2[False-True-False-1-dtype0]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROElasticCheckpoint::test_elastic_checkpoint_change_dp[True-True-True-False]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROElasticCheckpoint::test_elastic_checkpoint_change_dp[True-False-True-False]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROElasticCheckpoint::test_elastic_checkpoint_change_dp[False-True-True-False]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROElasticCheckpoint::test_elastic_checkpoint_change_dp[True-False-False-False]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROElasticCheckpoint::test_elastic_checkpoint_change_dp[False-False-True-False]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROElasticCheckpoint::test_elastic_checkpoint_change_dp[False-True-False-False]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROElasticCheckpoint::test_elastic_checkpoint_change_dp[True-True-False-False]":
    "Skip due to SW-237969",
    "unit/checkpoint/test_zero_optimizer.py::TestZeROElasticCheckpoint::test_elastic_checkpoint_change_dp[False-False-False-False]":
    "Skip due to SW-237969",
    "unit/inference/test_checkpoint_sharding.py::TestCheckpointShard::test[facebook/opt-350m-fp16-False]":
    "Skip due to SW-237969",
    "unit/inference/test_checkpoint_sharding.py::TestCheckpointShard::test[EleutherAI/gpt-neo-125M-fp16-False]":
    "Skip due to SW-237969",
    "unit/inference/test_checkpoint_sharding.py::TestCheckpointShard::test[bigscience/bloom-560m-fp16-False]":
    "Skip due to SW-237969",
    "unit/inference/test_checkpoint_sharding.py::TestCheckpointShard::test[facebook/opt-125m-fp16-False]":
    "Skip due to SW-237969",
    "unit/ulysses_plus/test_ulysses_sp_hf.py::TestUlyssesSPHF::test_ulysses_sp_hf[1]":
    "Skip due to SW-237969",
    "unit/ulysses_plus/test_ulysses_sp_hf.py::TestUlyssesSPHF::test_ulysses_sp_hf[3]":
    "Skip due to SW-237969",
}
