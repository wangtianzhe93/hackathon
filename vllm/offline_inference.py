import argparse
import dataclasses
import json

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from vllm.utils.argparse_utils import FlexibleArgumentParser


def main(args: argparse.Namespace):
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0, top_k=1, max_tokens=20)

    # Create an LLM.
    engine_args = EngineArgs.from_cli_args(args)

    # NOTE(woosuk): If the request cannot be processed in a single batch,
    # the engine will automatically process the request in multiple batches.
    engine_args_dict_org = dataclasses.asdict(engine_args)
    engine_args_dict = {
        **engine_args_dict_org,
        **{
            k: v
            for k, v in engine_args.__dict__.items() if k not in engine_args_dict_org
        }
    }
    llm = LLM(**engine_args_dict)


    for prompt_num in range(1, 13):
        print("-------------------------------------------")
        print("-------------------------------------------")

        prompts_list = []
        example_len = len(prompts)
        for num in range(prompt_num):
            prompts_list.append(prompts[num % example_len])

        # Generate texts from the prompts. The output is a list of RequestOutput objects
        # that contain the prompt, generated text, and other information.
        outputs = llm.generate(prompts_list, sampling_params)

        # Print the outputs.
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description='Offline inference test.')
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Path to save the results in JSON format.')
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
