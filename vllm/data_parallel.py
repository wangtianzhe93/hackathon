# SPDX-License-Identifier: Apache-2.0
"""
Usage:
Single node:
    python examples/offline_inference/data_parallel.py \
            --model="ibm-research/PowerMoE-3b" \
            --dp-size=2 \
            --tp-size=2

Multi-node:
    Node 0 (assume the node has ip of 10.99.48.128):
            python examples/offline_inference/data_parallel.py \
                    --model="ibm-research/PowerMoE-3b" \
                    --dp-size=2 \
                    --tp-size=2 \
                    --node-size=2 \
                    --node-rank=0 \
                    --master-addr=10.99.48.128 \
                    --master-port=13345
    Node 1:
            python examples/offline_inference/data_parallel.py \
                    --model="ibm-research/PowerMoE-3b" \
                    --dp-size=2 \
                    --tp-size=2 \
                    --node-size=2 \
                    --node-rank=1 \
                    --master-addr=10.99.48.128 \
                    --master-port=13345
"""
import os
import json
from time import sleep
from typing import Tuple

from vllm import LLM, SamplingParams
from vllm.utils import get_open_port


def _get_data_num_and_offset(total_data_size: int, parallel_num: int, rank: int) -> Tuple[int, int]:
    """
    Get data size and offset for the current dp group.
    For example, total batch 11, parallel_num 4, result is [3, 3, 3, 2]
                 total batch  8, parallel_num 4, result is [2, 2, 2, 2]
    """
    base = total_data_size // parallel_num
    remainder = total_data_size % parallel_num
    if rank >= remainder:
        return base, total_data_size - (parallel_num - rank) * base
    else:
        return base + 1, (base + 1) * rank


def main(model, dp_size, local_dp_rank, global_dp_rank, dp_master_ip,
         dp_master_port, GPUs_per_dp_rank, enforce_eager, enable_expert_parallel,
         block_size, speculative_config, mlu_config):
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    kwargs = {}
    if speculative_config is not None:
        kwargs["speculative_config"] = speculative_config
    if mlu_config is not None:
        kwargs["mlu_config"] = mlu_config

    # Create an LLM.
    llm = LLM(model=model,
              trust_remote_code=True,
              tensor_parallel_size=GPUs_per_dp_rank,
              distributed_executor_backend="mp",
              gpu_memory_utilization=0.9,
              enforce_eager=enforce_eager,
              enable_expert_parallel=enable_expert_parallel,
              block_size=block_size,
              max_num_batched_tokens=2048,
              max_model_len=64,
              max_num_seqs=32,
              **kwargs)

    sampling_params = SamplingParams(temperature=0,
                                     top_k=1,
                                     max_tokens=20)

    # Sample prompts.
    prompt_examples = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    for prompt_num in range(1, 17):
        if local_dp_rank == 0:
            print("------------------------------------------")
            print(f"total prompt num: {prompt_num}")
            print("------------------------------------------")

        prompts = []
        example_len = len(prompt_examples)
        for num in range(prompt_num):
            prompts.append(prompt_examples[num % example_len])

        dp_prompt_num, dp_prompt_offset = _get_data_num_and_offset(
            prompt_num, dp_size, global_dp_rank)
        prompts = prompts[dp_prompt_offset:dp_prompt_offset+dp_prompt_num]

        if len(prompts) == 0:
            # if any rank has no prompts to process,
            # we need to set a placeholder prompt
            prompts = ["Placeholder"]
        print(f"DP rank {global_dp_rank} needs to process {len(prompts)} prompts")

        outputs = llm.generate(prompts, sampling_params)

        # Print the outputs.
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"DP rank {global_dp_rank}, Prompt: {prompt!r}, "
                  f"Generated text: {generated_text!r}")

        sleep(0.1)

    # Give engines time to pause their processing loops before exiting.
    sleep(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Data Parallel Inference")
    parser.add_argument("--model",
                        type=str,
                        default="ibm-research/PowerMoE-3b",
                        help="Model name or path")
    parser.add_argument('--enforce-eager',
                        action='store_true',
                        help='Always use eager-mode PyTorch. If False, '
                        'will use eager mode and CUDA graph in hybrid '
                        'for maximal performance and flexibility.')
    parser.add_argument('--enable-expert-parallel',
                        action='store_true',
                        help='Enable expert parallel')
    parser.add_argument('--block-size',
                        type=int,
                        default=16,
                        help='Block size')
    parser.add_argument('--mlu-config',
                        type=json.loads,
                        default=None,
                        help='mlu config')
    parser.add_argument('--speculative-config',                                
                        type=json.loads,                                          
                        default=None,                                                               
                        help='speculative config') 
    parser.add_argument("--dp-size",
                        type=int,
                        default=2,
                        help="Data parallel size")
    parser.add_argument("--tp-size",
                        type=int,
                        default=2,
                        help="Tensor parallel size")
    parser.add_argument("--node-size",
                        type=int,
                        default=1,
                        help="Total number of nodes")
    parser.add_argument("--node-rank",
                        type=int,
                        default=0,
                        help="Rank of the current node")
    parser.add_argument("--master-addr",
                        type=str,
                        default="",
                        help="Master node IP address")
    parser.add_argument("--master-port",
                        type=int,
                        default=0,
                        help="Master node port")
    args = parser.parse_args()

    dp_size = args.dp_size
    tp_size = args.tp_size
    node_size = args.node_size
    node_rank = args.node_rank
    enforce_eager = args.enforce_eager
    enable_expert_parallel = args.enable_expert_parallel

    if node_size == 1:
        dp_master_ip = "127.0.0.1"
        dp_master_port = get_open_port()
    else:
        dp_master_ip = args.master_addr
        dp_master_port = args.master_port

    assert dp_size % node_size == 0, "dp_size should be divisible by node_size"
    dp_per_node = dp_size // node_size

    from multiprocessing import Process

    procs = []
    for local_dp_rank, global_dp_rank in enumerate(
            range(node_rank * dp_per_node, (node_rank + 1) * dp_per_node)):
        proc = Process(target=main,
                       args=(args.model, dp_size, local_dp_rank,
                             global_dp_rank, dp_master_ip, dp_master_port,
                             tp_size, enforce_eager, enable_expert_parallel,
                             args.block_size, args.speculative_config, args.mlu_config))
        proc.start()
        procs.append(proc)
    exit_code = 0
    for proc in procs:
        proc.join(timeout=300)
        if proc.exitcode is None:
            print(f"Killing process {proc.pid} that "
                  f"didn't stop within 5 minutes.")
            proc.kill()
            exit_code = 1
        elif proc.exitcode:
            exit_code = proc.exitcode

    exit(exit_code)

