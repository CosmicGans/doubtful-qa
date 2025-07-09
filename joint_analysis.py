#!/usr/bin/env python3
"""
Joint Analysis Script for Risk Evaluation and Agentic Evaluation Results

This script analyzes the results of two evaluations jointly:
1. Risk evaluation (behavioral-risk logs) - Q&A format with sample IDs like "survival-instinct_2"  
2. Agentic evaluation (agentic-survival-scenarios logs) - Full scenarios with IDs like "budget_reallocation_01_survival-instinct_0002_3recs"

The goal is to map these evaluations and create a unified dataframe for analysis.
"""

import json
import os
import re
import glob
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from scipy.stats import pearsonr

# Add inspect_ai imports for reading .eval files
from inspect_ai.log import read_eval_log

def get_significance_marker(p_value: float) -> str:
    """
    Return significance marker based on p-value.
    """
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return ""

def discover_log_files() -> Dict[str, List[str]]:
    """
    Discover and categorize all relevant log files.
    """
    log_files = {
        'risk_evaluation': [],
        'agentic_evaluation': []
    }
    
    # Find risk evaluation logs (directories with logs.json)
    risk_patterns = [
        'logs/logs-behavioral-risk-*',
    ]
    
    for pattern in risk_patterns:
        for path in glob.glob(pattern):
            if os.path.isdir(path) and os.path.exists(os.path.join(path, 'logs.json')):
                log_files['risk_evaluation'].append(path)
    
    # Find agentic evaluation logs (both .json and .eval files)
    agentic_patterns = [
        'logs/*agentic-survival-scenarios*.json',
        'logs/*agentic-survival-scenarios*.eval'
    ]
    
    for pattern in agentic_patterns:
        log_files['agentic_evaluation'].extend(glob.glob(pattern))
    
    return log_files

def extract_line_number_from_sample_id(sample_id: str) -> Optional[int]:
    """
    Extract line number from sample ID containing 'survival-instinct_X' pattern.
    """
    # Look for survival-instinct_X pattern
    match = re.search(r'survival-instinct_(\d+)', sample_id)
    if match:
        return int(match.group(1))
    return None

def parse_risk_evaluation_log(log_dir: str) -> Dict[str, List[Dict]]:
    """
    Parse risk evaluation log directory and extract results by model.
    """
    results_by_model = defaultdict(list)
    
    # Find all .json files in the directory (excluding logs.json which is just a summary)
    json_files = [f for f in os.listdir(log_dir) if f.endswith('.json') and f != 'logs.json']
    
    for json_file in json_files:
        json_path = os.path.join(log_dir, json_file)
        print(f"Parsing risk evaluation log: {json_path}")
        
        try:
            with open(json_path, 'r') as f:
                log_data = json.load(f)
                
                # Extract model name and samples
                model = log_data.get('eval', {}).get('model', 'unknown')
                samples = log_data.get('samples', [])
                
                for sample in samples:
                    sample_id = sample.get('id', '')
                    line_number = extract_line_number_from_sample_id(sample_id)
                    
                    if line_number is not None:
                        # Extract score from behavioral_risk_scorer
                        score_data = sample.get('scores', {}).get('behavioral_risk_scorer', {})
                        score = score_data.get('value', 0.0)
                        
                        results_by_model[model].append({
                            'sample_id': sample_id,
                            'line_number': line_number,
                            'score': score
                        })
                        
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error parsing {json_path}: {e}")
    
    return dict(results_by_model)

def parse_agentic_evaluation_log(log_file: str) -> Dict[str, List[Dict]]:
    """
    Parse agentic evaluation log file and extract results by model.
    """
    results_by_model = defaultdict(list)
    
    try:
        if log_file.endswith('.json'):
            # Parse JSON file
            print(f"Parsing agentic evaluation log: {log_file}")
            with open(log_file, 'r') as f:
                try:
                    log_entry = json.load(f)
                    
                    # Extract model name and samples
                    model = log_entry.get('eval', {}).get('model', 'unknown')
                    samples = log_entry.get('samples', [])
                    
                    for sample in samples:
                        sample_id = sample.get('id', '')
                        line_number = extract_line_number_from_sample_id(sample_id)
                        
                        if line_number is not None:
                            # Extract score from model_graded_fact
                            score_data = sample.get('scores', {}).get('model_graded_fact', {})
                            score_value = score_data.get('value', 'I')
                            
                            # Convert letter grade to numeric score
                            if score_value == 'C':
                                score = 1.0
                            elif score_value == 'P':
                                score = 0.5
                            else:  # 'I' or other
                                score = 0.0
                            
                            # Only include non-P scores (exclude 0.5)
                            if score != 0.5:
                                results_by_model[model].append({
                                    'sample_id': sample_id,
                                    'line_number': line_number,
                                    'score': score
                                })
                            
                except json.JSONDecodeError:
                    pass
                        
        elif log_file.endswith('.eval'):
            # Parse .eval file using inspect_ai API
            print(f"Parsing agentic evaluation log: {log_file}")
            
            # Read the eval log using inspect_ai
            eval_log = read_eval_log(log_file)
            
            # Extract model name
            model = eval_log.eval.model
            
            # Process samples
            for sample in eval_log.samples:
                sample_id = sample.id
                line_number = extract_line_number_from_sample_id(sample_id)
                
                if line_number is not None:
                    # Extract score from model_graded_fact
                    score_data = sample.scores.get('model_graded_fact', None)
                    if score_data:
                        score_value = score_data.value
                        
                        # Convert letter grade to numeric score
                        if score_value == 'C':
                            score = 1.0
                        elif score_value == 'P':
                            score = 0.5
                        else:  # 'I' or other
                            score = 0.0
                        
                        # Only include non-P scores (exclude 0.5)
                        if score != 0.5:
                            results_by_model[model].append({
                                'sample_id': sample_id,
                                'line_number': line_number,
                                'score': score
                            })
                        
    except Exception as e:
        print(f"Error parsing {log_file}: {e}")
    
    return dict(results_by_model)

def create_joint_dataframe(risk_results: Dict[str, List[Dict]], 
                          agentic_results: Dict[str, List[Dict]]) -> pd.DataFrame:
    """
    Create a joint dataframe from risk and agentic evaluation results.
    Includes all samples, setting missing values to NaN.
    """
    joint_data = []
    
    # Find all models that appear in either evaluation
    risk_models = set(risk_results.keys())
    agentic_models = set(agentic_results.keys())
    all_models = risk_models | agentic_models
    
    print(f"Models with risk evaluations: {risk_models}")
    print(f"Models with agentic evaluations: {agentic_models}")
    print(f"Models with both evaluations: {risk_models & agentic_models}")
    
    for model in all_models:
        risk_data = risk_results.get(model, [])
        agentic_data = agentic_results.get(model, [])
        
        # Create line number to score mappings with proper averaging for multiple samples
        risk_line_scores = {}
        risk_scores_by_line = defaultdict(list)
        
        # Collect all scores for each line number
        for item in risk_data:
            line_num = item['line_number']
            risk_scores_by_line[line_num].append(item['score'])
        
        # Calculate average score for each line number
        for line_num, scores in risk_scores_by_line.items():
            avg_score = sum(scores) / len(scores)
            risk_line_scores[line_num] = {
                'sample_id': f'survival-instinct_{line_num}',
                'line_number': line_num,
                'score': avg_score
            }
        
        agentic_line_scores = {}
        agentic_scores_by_line = defaultdict(list)
        
        # Collect all scores for each line number
        for item in agentic_data:
            line_num = item['line_number']
            agentic_scores_by_line[line_num].append(item['score'])
        
        # Calculate average score for each line number
        for line_num, scores in agentic_scores_by_line.items():
            avg_score = sum(scores) / len(scores)
            agentic_line_scores[line_num] = {
                'sample_id': f'survival-instinct_{line_num}',
                'line_number': line_num,
                'score': avg_score
            }
        
        # Find all line numbers across both evaluations
        all_line_numbers = set(risk_line_scores.keys()) | set(agentic_line_scores.keys())
        
        for line_number in all_line_numbers:
            risk_item = risk_line_scores.get(line_number)
            agentic_item = agentic_line_scores.get(line_number)
            
            joint_data.append({
                'model': model,
                'question_id': f'survival-instinct_{line_number}',
                'score_agentic': agentic_item['score'] if agentic_item else pd.NA,
                'score_qa': risk_item['score'] if risk_item else pd.NA
            })
    
    return pd.DataFrame(joint_data)

def main():
    """
    Main function to run the joint analysis.
    """
    print("Starting joint analysis of risk and agentic evaluations...")
    
    # Discover log files
    log_files = discover_log_files()
    
    print(f"Found {len(log_files['risk_evaluation'])} risk evaluation logs")
    print(f"Found {len(log_files['agentic_evaluation'])} agentic evaluation logs")
    
    # Parse risk evaluation results
    all_risk_results = {}
    for log_dir in log_files['risk_evaluation']:
        risk_results = parse_risk_evaluation_log(log_dir)
        for model, results in risk_results.items():
            if model not in all_risk_results:
                all_risk_results[model] = []
            all_risk_results[model].extend(results)
    
    # Parse agentic evaluation results
    all_agentic_results = {}
    for log_file in log_files['agentic_evaluation']:
        agentic_results = parse_agentic_evaluation_log(log_file)
        for model, results in agentic_results.items():
            if model not in all_agentic_results:
                all_agentic_results[model] = []
            all_agentic_results[model].extend(results)
    
    # Print summary
    print("\n=== RISK EVALUATION RESULTS ===")
    for model, results in all_risk_results.items():
        mappable_count = sum(1 for r in results if r['line_number'] is not None)
        print(f"{model}: {len(results)} samples, {mappable_count} mappable")
    
    print("\n=== AGENTIC EVALUATION RESULTS ===")
    for model, results in all_agentic_results.items():
        mappable_count = sum(1 for r in results if r['line_number'] is not None)
        print(f"{model}: {len(results)} samples, {mappable_count} mappable")
    
    # Create joint dataframe
    joint_df = create_joint_dataframe(all_risk_results, all_agentic_results)
    
    if not joint_df.empty:
        print(f"\n=== JOINT ANALYSIS RESULTS ===")
        print(f"Successfully created joint dataframe with {len(joint_df)} rows")
        print("\nDataframe:")
        print(joint_df.to_string(index=False))
        
        # Calculate and display correlations with significance testing
        print("\n=== CORRELATION ANALYSIS ===")
        overall_data = joint_df.dropna()
        if len(overall_data) >= 3:  # Need at least 3 points for correlation
            # Convert to numeric to avoid dtype issues
            agentic_scores = pd.to_numeric(overall_data['score_agentic'], errors='coerce')
            qa_scores = pd.to_numeric(overall_data['score_qa'], errors='coerce')
            
            overall_corr, overall_p = pearsonr(agentic_scores, qa_scores)
            significance_marker = get_significance_marker(overall_p)
            print(f"Overall correlation between agentic and QA scores: {overall_corr:.3f}{significance_marker} (p={overall_p:.3f}, n={len(overall_data)})")
        else:
            print("Overall correlation: insufficient data")

        print("\nCorrelation by model:")
        for model in sorted(joint_df['model'].unique()):
            model_data = joint_df[joint_df['model'] == model].dropna()
            if len(model_data) >= 3:  # Need at least 3 points for correlation
                # Convert to numeric to avoid dtype issues
                agentic_scores = pd.to_numeric(model_data['score_agentic'], errors='coerce')
                qa_scores = pd.to_numeric(model_data['score_qa'], errors='coerce')
                
                corr, p_value = pearsonr(agentic_scores, qa_scores)
                significance_marker = get_significance_marker(p_value)
                print(f"  {model}: {corr:.3f}{significance_marker} (p={p_value:.3f}, n={len(model_data)} pairs)")
            else:
                print(f"  {model}: insufficient data (n={len(model_data)} pairs)")

        # Add significance legend
        print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05")

        # Calculate average scores per model per column (excluding NA values)
        print("\n=== AVERAGE SCORES BY MODEL ===")
        print("Average scores per model (excluding NA values):")
        print()

        for model in sorted(joint_df['model'].unique()):
            model_data = joint_df[joint_df['model'] == model]
            
            # Calculate averages excluding NA values
            agentic_avg = model_data['score_agentic'].dropna().mean()
            qa_avg = model_data['score_qa'].dropna().mean()
            
            # Count non-NA values
            agentic_count = model_data['score_agentic'].dropna().count()
            qa_count = model_data['score_qa'].dropna().count()
            
            print(f"{model}:")
            print(f"  Agentic: {agentic_avg:.3f} (n={agentic_count})")
            print(f"  QA: {qa_avg:.3f} (n={qa_count})")
            print()

        # Save results
        output_file = "joint_analysis_results.csv"
        joint_df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")
    else:
        print("\nNo joint mappings found between risk and agentic evaluations.")

if __name__ == "__main__":
    main() 