# Benchmarking on TRAIL

TRAIL(Trace Reasoning and Agentic Issue Localization) is a benchmark dataset of 148 annotated AI agent execution traces containing 841 errors across reasoning, execution, and planning categories. Created from real-world software engineering and information retrieval tasks, it challenges even state-of-the-art LLMs, with the best model achieving only 11% accuracy, highlighting the difficulty of trace debugging for complex agent workflows.

<img src="https://i.imgur.com/QeHGLAj.png" alt="TRAIL Results" width="55%"/></br>


# Installation
Create a virtual environment and install the required packages as follows:
```bash
pip install -r requirements.txt
```

# Usage
```bash
python run_eval.py --model=[your_litellm_compatible_model_id] --data_dir="data/" --output_dir="results/" --max_workers=[integer_number_of_workers] --split=["GAIA"|"SWE Bench"]
``` 

This will produce a result evals in the `results/` directory. You can then run:

```bash
python calculate_scores.py --results_dir="results/"
```
This will create and store a `.txt` file in the same `results` directory with the calculated scores for each model.

# Citation
If you use this code or the dataset, please cite the following paper:

```bibtex
@misc{deshpande2025trail,
      title={TRAIL: Trace Reasoning and Agentic Issue Localization},
      author={Darshan Deshpande and Varun Gangal and Hersh Mehta and Jitin Krishnan and Anand Kannappan and Rebecca Qian},
      year={2025},
      eprint={2505.08638},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2505.08638},
}
```
