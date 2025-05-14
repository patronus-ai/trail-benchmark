import json
import re
import glob
import os
import argparse
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import f1_score
from scipy.stats import pearsonr


def normalize_category(category: str, all_categories: List[str]) -> str:
    """Normalize a category name by finding the closest match in all_categories."""
    if not category:
        return ""
    
    category = category.lower().strip()
    category_no_spaces = category.replace(" ", "")
    
    for std_cat in all_categories:
        if category == std_cat.lower() or category_no_spaces == std_cat.lower().replace(" ", ""):
            return std_cat
    
    # Try substring matching
    for std_cat in all_categories:
        if category_no_spaces in std_cat.lower().replace(" ", ""):
            return std_cat
    
    # If no match found, return original
    return category


def calculate_metrics(ground_truth: Dict, generated: Dict, all_categories: List[str]) -> Dict:
    # Process ground truth errors
    gt_errors = ground_truth.get("errors", [])
    gt_categories_raw = [error.get("category", "") for error in gt_errors]
    gt_locations = [error.get("location", "") for error in gt_errors]
    
    # Process generated errors
    gen_errors = generated.get("errors", [])
    gen_categories_raw = [error.get("category", "") for error in gen_errors]
    gen_locations = [error.get("location", "") for error in gen_errors]
    
    # Normalize categories
    gt_categories = [normalize_category(cat, all_categories) for cat in gt_categories_raw if cat]
    gen_categories = [normalize_category(cat, all_categories) for cat in gen_categories_raw if cat]
    
    # Create location-category pairs
    gt_loc_cat_pairs = [(gt_locations[i], gt_categories[i]) for i in range(len(gt_locations)) if i < len(gt_categories)]
    gen_loc_cat_pairs = [(gen_locations[i], gen_categories[i]) for i in range(len(gen_locations)) if i < len(gen_categories)]
    
    # Calculate location-category joint accuracy
    common_pairs = set(gt_loc_cat_pairs).intersection(set(gen_loc_cat_pairs))
    joint_accuracy = len(common_pairs) / len(set(gt_loc_cat_pairs)) if gt_loc_cat_pairs else 0
    
    # Calculate location accuracy
    common_locations = set(gt_locations).intersection(set(gen_locations))
    location_accuracy = len(common_locations) / len(set(gt_locations)) if gt_locations else 0
    
    # Create binary vectors for categories
    y_true = np.zeros(len(all_categories))
    y_pred = np.zeros(len(all_categories))
    
    for cat in gt_categories:
        if cat in all_categories:
            y_true[all_categories.index(cat)] = 1
            
    for cat in gen_categories:
        if cat in all_categories:
            y_pred[all_categories.index(cat)] = 1
    
    # Extract scores
    gt_scores = ground_truth.get("scores", [{}])[0] if ground_truth.get("scores") else {}
    gen_scores = generated.get("scores", [{}])[0] if generated.get("scores") else {}
    
    return {
        "location_accuracy": location_accuracy,
        "joint_accuracy": joint_accuracy,
        "y_true": y_true,
        "y_pred": y_pred,
        "gt_categories": gt_categories,
        "gen_categories": gen_categories,
        "gt_scores": gt_scores,
        "gen_scores": gen_scores
    }


def load_files(file1: str, file2: str) -> Tuple[Dict, str]:
    with open(file1, "r") as f1, open(file2, "r") as f2:
        data1 = json.load(f1)
        data2 = f2.read()
    return data1, data2


def extract_json_from_text(text: str) -> Dict:
    # Extract JSON from text
    json_str = re.search(r"\{.*\}", text, re.DOTALL)
    if json_str:
        try:
            return json.loads(json_str.group(0))
        except json.JSONDecodeError:
            # Try to find a valid JSON by iteratively removing characters from the end
            json_text = json_str.group(0)
            while len(json_text) > 2:  # At minimum need {}
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    json_text = json_text[:-1]
            raise ValueError("Could not extract valid JSON from text")
    else:
        raise ValueError("No JSON found in the text")


def main(ground_truth_dir: str, generated_dir: str):
    all_categories = [
        "Language-only", "Tool-related", "Poor Information Retrieval", "Incorrect Memory Usage", 
        "Tool Output Misinterpretation", "Incorrect Problem Identification", "Tool Selection Errors", 
        "Formatting Errors", "Instruction Non-compliance", "Tool Definition Issues", 
        "Environment Setup Errors", "Rate Limiting", "Authentication Errors", "Service Errors", 
        "Resource Not Found", "Resource Exhaustion", "Timeout Issues", "Context Handling Failures", 
        "Resource Abuse", "Goal Deviation", "Task Orchestration"
    ]
    
    location_accuracy_sum = 0
    joint_accuracy_sum = 0
    
    # For per-category statistics
    all_y_true = []
    all_y_pred = []
    
    # For score correlation analysis
    gt_reliability_scores = []
    gen_reliability_scores = []
    gt_security_scores = []
    gen_security_scores = []
    gt_instruction_adherence_scores = []
    gen_instruction_adherence_scores = []
    gt_plan_opt_scores = []
    gen_plan_opt_scores = []
    gt_overall_scores = []
    gen_overall_scores = []
    
    files_processed = 0
    
    if not os.path.exists(generated_dir) or not os.path.isdir(generated_dir):
        print(f"Generated directory {generated_dir} does not exist or is a directory")
        return

    for file in glob.glob(f"{ground_truth_dir}/*.json"):
        file_name = os.path.basename(file)
        generated_file = f"{generated_dir}/{file_name}"

        if not os.path.exists(generated_file):
            print(f"Generated file {generated_file} does not exist")
            continue
        
        try:
            ground_truth, generated_text = load_files(file, generated_file)
            generated = extract_json_from_text(generated_text)
            metrics = calculate_metrics(ground_truth, generated, all_categories)
            
            # Add to global metrics
            all_y_true.append(metrics["y_true"])
            all_y_pred.append(metrics["y_pred"])
            
            location_accuracy_sum += metrics["location_accuracy"]
            joint_accuracy_sum += metrics["joint_accuracy"]
            
            # Extract scores for correlation analysis
            gt_scores = metrics["gt_scores"]
            gen_scores = metrics["gen_scores"]
            
            if gt_scores and gen_scores:
                # Reliability scores
                if "reliability_score" in gt_scores and "reliability_score" in gen_scores:
                    gt = gt_scores.get("reliability_score", -1) if gt_scores.get("reliability_score", -1) else -1
                    gen = gen_scores.get("reliability_score", -1) if gen_scores.get("reliability_score", -1) else -1
                    try:
                        gt = float(gt)
                        gen = int(gen)
                    except ValueError:
                        gt = float(gt)
                        gen = -1
                    
                    gt_reliability_scores.append(gt)
                    gen_reliability_scores.append(gen)
                
                # Security scores
                if "security_score" in gt_scores and "security_score" in gen_scores:
                    gt = gt_scores.get("security_score", -1) if gt_scores.get("security_score", -1) else -1
                    gen = gen_scores.get("security_score", -1) if gen_scores.get("security_score", -1) else -1
                    try:
                        gt = float(gt)
                        gen = float(gen)
                    except ValueError:
                        gt = float(gt)
                        gen = -1
                    gt_security_scores.append(gt)
                    gen_security_scores.append(gen)
                
                # Instruction adherence scores
                if "instruction_adherence_score" in gt_scores and "instruction_adherence_score" in gen_scores:
                    gt = gt_scores.get("instruction_adherence_score", -1) if gt_scores.get("instruction_adherence_score", -1) else -1
                    gen = gen_scores.get("instruction_adherence_score", -1) if gen_scores.get("instruction_adherence_score", -1) else -1
                    try:
                        gt = float(gt)
                        gen = float(gen)
                    except ValueError:
                        gt = float(gt)
                        gen = -1
                    gt_instruction_adherence_scores.append(gt)
                    gen_instruction_adherence_scores.append(gen)
                
                # Plan optimization scores
                if "plan_opt_score" in gt_scores and "plan_opt_score" in gen_scores:
                    gt = gt_scores.get("plan_opt_score", -1) if gt_scores.get("plan_opt_score", -1) else -1
                    gen = gen_scores.get("plan_opt_score", -1) if gen_scores.get("plan_opt_score", -1) else -1
                    try:
                        gt = float(gt)
                        gen = float(gen)
                    except ValueError:
                        gt = float(gt)
                        gen = -1
                    gt_plan_opt_scores.append(gt)
                    gen_plan_opt_scores.append(gen)
                
                # Overall scores
                if "overall" in gt_scores and "overall" in gen_scores:
                    gt = gt_scores.get("overall", -1) if gt_scores.get("overall", -1) else -1
                    gen = gen_scores.get("overall", -1) if gen_scores.get("overall", -1) else -1
                    try:
                        gt = float(gt)
                        gen = float(gen)
                    except ValueError:
                        gt = float(gt)
                        gen = -1
                    gt_overall_scores.append(gt)
                    gen_overall_scores.append(gen)
            
            files_processed += 1
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

    # Calculate average metrics
    if files_processed > 0:
        location_accuracy_avg = location_accuracy_sum / files_processed
        joint_accuracy_avg = joint_accuracy_sum / files_processed
    else:
        location_accuracy_avg = 0
        joint_accuracy_avg = 0
    
    # Calculate score correlations
    score_correlations = {}
    
    if gt_reliability_scores and gen_reliability_scores:
        print(gt_reliability_scores, gen_reliability_scores)
        try:
            corr, p_value = pearsonr(gt_reliability_scores, gen_reliability_scores)
        except Exception as e:
            print(f"Error calculating Pearson correlation for reliability: {e}")
            corr, p_value = 0, 1
        score_correlations["reliability"] = {"correlation": corr, "p_value": p_value, "n": len(gt_reliability_scores)}
    
    if gt_security_scores and gen_security_scores:
        try:
            corr, p_value = pearsonr(gt_security_scores, gen_security_scores)
        except Exception as e:
            print(f"Error calculating Pearson correlation for security: {e}")
            corr, p_value = 0, 1
        score_correlations["security"] = {"correlation": corr, "p_value": p_value, "n": len(gt_security_scores)}
    
    if gt_instruction_adherence_scores and gen_instruction_adherence_scores:
        try:
            corr, p_value = pearsonr(gt_instruction_adherence_scores, gen_instruction_adherence_scores)
        except Exception as e:
            print(f"Error calculating Pearson correlation for instruction adherence: {e}")
            corr, p_value = 0, 1
        score_correlations["instruction_adherence"] = {"correlation": corr, "p_value": p_value, "n": len(gt_instruction_adherence_scores)}
    
    if gt_plan_opt_scores and gen_plan_opt_scores:
        try:
            corr, p_value = pearsonr(gt_plan_opt_scores, gen_plan_opt_scores)
        except Exception as e:
            print(f"Error calculating Pearson correlation for plan optimization: {e}")
            corr, p_value = 0, 1
        score_correlations["plan_optimization"] = {"correlation": corr, "p_value": p_value, "n": len(gt_plan_opt_scores)}
    
    if gt_overall_scores and gen_overall_scores:
        try:
            corr, p_value = pearsonr(gt_overall_scores, gen_overall_scores)
        except Exception as e:
            print(f"Error calculating Pearson correlation for overall scores: {e}")
            corr, p_value = 0, 1
        score_correlations["overall"] = {"correlation": corr, "p_value": p_value, "n": len(gt_overall_scores)}
    
    # Aggregate all predictions for weighted F1 and per-category metrics
    if files_processed > 0:
        all_y_true_array = np.vstack(all_y_true)
        all_y_pred_array = np.vstack(all_y_pred)
        
        # Calculate per-category metrics
        category_metrics = {}
        for i, category in enumerate(all_categories):
            true_positives = np.sum((all_y_true_array[:, i] == 1) & (all_y_pred_array[:, i] == 1))
            false_positives = np.sum((all_y_true_array[:, i] == 0) & (all_y_pred_array[:, i] == 1))
            false_negatives = np.sum((all_y_true_array[:, i] == 1) & (all_y_pred_array[:, i] == 0))
            
            support = np.sum(all_y_true_array[:, i])
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            category_metrics[category] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(support)
            }
        
        # Calculate weighted F1 score across the entire dataset
        weighted_f1 = f1_score(all_y_true_array, all_y_pred_array, average='weighted', zero_division=0)
    else:
        category_metrics = {cat: {"precision": 0, "recall": 0, "f1": 0, "support": 0} for cat in all_categories}
        weighted_f1 = 0
    
    # Write metrics to file
    with open(f"{generated_dir}-metrics.txt", "w") as f:
        f.write(f"Weighted F1: {weighted_f1:.4f}\n")
        f.write(f"Average Location Accuracy: {location_accuracy_avg:.4f}\n")
        f.write(f"Average Location-Category Joint Accuracy: {joint_accuracy_avg:.4f}\n\n")
        
        f.write("Score Correlations (Pearson r):\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Score Type':<25} {'Correlation':<15} {'p-value':<15} {'N':<10}\n")
        f.write("-" * 80 + "\n")
        
        for score_type, metrics in score_correlations.items():
            f.write(f"{score_type:<25} {metrics['correlation']:15.4f} {metrics['p_value']:15.4f} {metrics['n']:<10}\n")
        
        f.write("\nPer-Category Statistics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Category':<40} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}\n")
        f.write("-" * 80 + "\n")
        
        for category, metrics in category_metrics.items():
            if metrics["support"] > 0:  # Only show categories that appear in the dataset
                f.write(f"{category:<40} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1']:<10.4f} {metrics['support']:<10}\n")
    
    return {
        "weighted_f1": weighted_f1,
        "location_accuracy": location_accuracy_avg,
        "joint_accuracy": joint_accuracy_avg,
        "category_metrics": category_metrics,
        "score_correlations": score_correlations
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results", help="Directory containing generated results")
    args = parser.parse_args()
    
    for i in glob.glob(f"{args.results_dir}/*"):
        split = i.split("-")[-1].lower().replace(" ", "_")
        results = main(ground_truth_dir=f"processed_annotations_{split}", generated_dir=f"{i}")
        if not results:
            continue
        print("="*100)
        print("Analyzed directory:", i)
        print(f"Weighted F1: {results['weighted_f1']:.4f}")
        print(f"Average Location Accuracy: {results['location_accuracy']:.4f}")
        print(f"Average Location-Category Joint Accuracy: {results['joint_accuracy']:.4f}")
        
        print("\nScore Correlations (Pearson r):")
        print("-" * 60)
        print(f"{'Score Type':<25} {'Correlation':<15} {'p-value':<15}")
        print("-" * 60)
        for score_type, metrics in results.get('score_correlations', {}).items():
            print(f"{score_type:<25} {metrics['correlation']:15.4f} {metrics['p_value']:15.4f}")
        
        print("="*100)
