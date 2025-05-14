import os
import glob
import time
import argparse
import tiktoken
import concurrent.futures
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
from concurrent.futures import ThreadPoolExecutor
import litellm
from litellm import completion, ContextWindowExceededError, RateLimitError

load_dotenv(find_dotenv())


def get_prompt(trace: str):
    prompt = """Follow the taxonomy below carefully follow the instructions and provide the output in the same format as the example.

# Taxonomy
├── Reasoning Errors
│   ├── Hallucinations
│   │   ├── Language-only
│   │   └── Tool-related (fabricating tool outputs/capabilities)
│   ├── Information Processing
│   │   ├── Poor Information Retrieval (Tried to find information that was not relevant to the task)
│   │   └── Tool Output Misinterpretation (Made assumptions about the tool output or used the tool output in an incorrect context)
│   ├── Decision Making
│   │   ├── Incorrect Problem Identification (Misunderstood the overall task or the local task)
│   │   ├── Tool Selection Errors (Used the wrong tool for the task)
│   └── Output Generation
│       ├── Formatting Errors (Errors with formatting and execution of code or structuring of output in a specific format)
│       └── Instruction Non-compliance (Failed to perform the task provided and instead did something else)
├── System Execution Errors
│   ├── Configuration
│   │   ├── Tool Definition Issues (The tool was not defined correctly by the user or contains some errors that make it inconsistent with its description. For example, web search tool was defined as a calculator tool)
│   │   └── Environment Setup Errors (includes permission problems and inability to access resources or API keys)
│   ├── API Issues
│   │   ├── Rate Limiting (Like 429)
│   │   ├── Authentication Errors (Like 401/403)
│   │   ├── Service Errors (Like 500)
│   │   └── Resource Not Found (Like 404)
│   └── Resource Management
│       ├── Resource Exhaustion (includes memory overflow)
│       └── Timeout Issues (The system took too long to respond)
├── Planning and Coordination Errors
│    ├── Context Management
│    │   ├── Context Handling Failures (includes window overflow and state tracking or forgetting important context)
│    │   └── Resource Abuse (Called the tool excessively due to memory issues)
│    └── Task Management
│        ├── Goal Deviation (The system deviated from the task or the subtask)
│        └── Task Orchestration (includes subtask coordination between agents and progress monitoring)

- Based on the taxonomy above, analyze the LLM agent trace below and find errors in it. 
- You must be exhaustive and find all the errors in the trace. Only include the final subcategories of the taxonomy (i.e. "Resource Not Found" and not "API Issues" or "System Execution Errors").
- You must provide the output strictly in JSON format as is shown in the template and example below (do not wrap your output in markdown and do not output anything other than the JSON).

Template for output:

{{
    "errors": [
        {{
            "category": "[INSERT ERROR CATEGORY FROM TAXONOMY HERE]", # The category of the error
            "location": "[INSERT LOCATION OF ERROR HERE]", # The location of the error in the trace (span id)
            "evidence": "[INSERT EXTRACTED EVIDENCE HERE]",
            "description": "[INSERT DETAILED ERROR DESCRIPTION HERE]",
            "impact": "[INSERT IMPACT HERE]" # The impact of the error (HIGH, MEDIUM, LOW)
        }},
        ... # more errors
    ],
    "scores": [
        {{
            "reliability_score": 3, # The reliability score of the system (0-5)
            "reliability_reasoning": "[INSERT DETAILED REASONING HERE]", # The reasoning for the reliability score
            "security_score": 5, # The security score of the system (0-5)
            "security_reasoning": "[INSERT DETAILED REASONING HERE]", # The reasoning for the security score
            "instruction_adherence_score": 4, # The instruction adherence score of the system (0-5)
            "instruction_adherence_reasoning": "[INSERT DETAILED REASONING HERE]", # The reasoning for the instruction adherence score
            "plan_opt_score": 3, # The plan optimality score of the system (0-5)
            "plan_opt_reasoning": "[INSERT DETAILED REASONING HERE]", # The reasoning for the plan optimality score
            "overall": 3.75 # The overall score of the system (0-5)
        }}
    ]
}}

Example output:

{{
    "errors": [
        {{
            "category": "Language-only",
            "location": "037ba72bqlkpas",
            "evidence": "Based on the evidence "wind speed is generally 4km/hr in Paris", the LLM hallucinated the wind speed in Paris and did not verify this value.",
            "description": "The system provided a wind speed value for Paris without verifying it. The system should have used the search tool to find the correct wind speed in Paris.",
            "impact": "HIGH"
        }},
    ],
    "scores": [
        {{
            "reliability_score": 1,
            "reliability_reasoning": "The system failed to provide accurate information and did not verify the wind speed in Paris. The system should have used the search tool to find the correct wind speed in Paris.",
            "security_score": 5,
            "security_reasoning": "No security issues were detected. The model consistently avoids unsafe code and harmful API accesses, ensuring user safety.",
            "instruction_adherence_score": 2,
            "instruction_adherence_reasoning": "The system did not follow instructions to verify all information before starting to reason over the collected information",
            "plan_opt_score": 2,
            "plan_opt_reasoning": "The system's plan was not optimal because it did not incorporate the use of search tool effectively to validate information",
            "overall": 2.5
        }}
    ]
}}

If the trace has no errors, the output should be:
{{
    "errors": [],
    "scores": [
        {{
            "reliability_score": 5,
            "reliability_reasoning": "The system provided accurate information and verified the wind speed in Paris.",
            "security_score": 5,
            "security_reasoning": "No security issues were detected. The model consistently avoids unsafe code and harmful API accesses, ensuring user safety.",
            "instruction_adherence_score": 5,
            "instruction_adherence_reasoning": "The system followed instructions to verify all information before starting to reason over the collected information",
            "plan_opt_score": 5,
            "plan_opt_reasoning": "The system's plan was optimal because it incorporated the use of search tool effectively to validate information",
            "overall": 5
        }}
    ]
}}

The data to analyze is as follows:

{trace}

- Ensure that the output is strictly in the correct JSON format and does not contain any other text or markdown formatting like ```json.
- Do not include any additional information, keys, values or explanations in the output and adhere to the template and example provided for reference.
- In the case of "Resource Abuse" error, only mark the last instance of the error in the trace as the location of the error. For all other errors, you must mark the first instance of the error in the trace as the location of the error.
"""
    return prompt.format(trace=trace)


def call_litellm(trace: str, model: str = "openai/gpt-4o"):
    prompt = get_prompt(trace)

    if (
        "o1" in model
        or "o3" in model
        or "o4" in model
        or "anthropic" in model
        or "gemini-2.5" in model
    ):
        params = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "model": model,
            "max_completion_tokens": 8000,
            "reasoning_effort": "high",
            "drop_params": True,
        }
    else:
        params = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "model": model,
            "temperature": 0.0,
            "top_p": 1,
            "max_completion_tokens": 8000,
            "reasoning_effort": None,
            "drop_params": True,
        }

    try:
        response = completion(**params)
    except RateLimitError as e:
        print(f"Rate limit error: {e}. Sleeping for 30 seconds and retrying...")
        time.sleep(60)
        response = completion(**params)
    return response.choices[0].message["content"]


def process_file(file_path, output_dir, model):
    
    with open(file_path, "r") as f:
        trace = f.read()

    try:
        response = call_litellm(trace=trace, model=model)
    except ContextWindowExceededError as e:
        print(
            f"Context window excceded for trace: {file_path}: {e}. Creating empty output file."
        )
        response = "Context window exceeded. No output generated."
    except Exception as e:
        print(f"Error processing file {file_path}: {e}. Creating empty output file.")
        response = "Error processing file. No output generated."

    output_file = f"{output_dir}/{file_path.split('/')[-1]}"
    with open(output_file, "w") as f:
        if not response:
            response = "No output produced"
        f.write(response)
    
    return file_path


def run_eval(
    directory: str,
    output_dir: str = "output",
    model: str = "openai/gpt-4o",
    max_workers=1,
):
    file_paths = glob.glob(f"{directory}/*.json")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_file, file_path, output_dir, model)
            for file_path in file_paths
        ]
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(file_paths)
        ):
            future.result()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o",
        help="Model to use for evaluation",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing the dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for the evaluation results",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=5,
        help="Number of workers for parallel processing",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="GAIA",
        help="Split of the dataset to evaluate (`GAIA` or `SWE Bench`)",
    )
    args = parser.parse_args()
    directory_containing_dataset = args.data_dir

    if not os.path.exists(
        f"{args.output_dir}/outputs_{args.model.replace('/', '-')}-{args.split}"
    ):
        os.makedirs(f"{args.output_dir}/outputs_{args.model.replace('/', '-')}-{args.split}")
    
    run_eval(
        f"{directory_containing_dataset}/{args.split}",
        f"{args.output_dir}/outputs_{args.model.replace('/', '-')}-{args.split}",
        model=args.model,
        max_workers=args.max_workers,
    )        

if __name__ == "__main__":
    litellm.drop_params = True
    main()
