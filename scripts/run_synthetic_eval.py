
import json
import os
import sys
import time

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.brain import SmartBrain
from evaluation.evaluator import RAGEvaluator

def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_evaluation():
    dataset_path = "data/synthetic_dataset.json"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return

    print(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} items.")

    brain = SmartBrain()
    
    test_cases = []
    
    print("\nRunning queries through SmartBrain...")
    for i, item in enumerate(dataset):
        query = item["query"]
        reference_answer = item["reference_answer"]
        relevant_ids = item.get("relevant_ids", [])
        
        print(f"[{i+1}/{len(dataset)}] Query: {query}")
        
        # 1. Retrieval
        try:
            candidates = brain.retrieve(query, top_k=10)
            
            # 2. Reranking (if enabled internally)
            reranked = brain.rerank(query, candidates, top_n=5)
            
            # 3. Generation
            context_str = "\n\n".join(
                f"[ข้อมูลจาก: {c['metadata'].get('source', 'Unknown')}]\n{c['text']}"
                for c in reranked
            )
            
            response_gen = brain.think(query, context_str)
            full_response = "".join(list(response_gen))
            
            # Construct test case for evaluator
            retrieved_ids = []
            for c in reranked:
                # Reconstruct chunk ID from metadata if possible, or use filename_index
                # ingest.py stores chunk_index. 
                # ID format: filename___chunk_index (sanitized)
                filename = c['metadata'].get('source', '')
                chunk_index = c['metadata'].get('chunk_index', '')
                
                # Try to match the ID format used in generation script
                raw_id = f"{filename}___{chunk_index}"
                import re
                chunk_id = re.sub(r'[^a-zA-Z0-9_-]', '_', raw_id)
                retrieved_ids.append(chunk_id)

            test_cases.append({
                "query": query,
                "reference_answer": reference_answer,
                "relevant_ids": relevant_ids,
                "retrieved_ids": retrieved_ids,
                "context_chunks": [c["text"] for c in reranked],
                "generated_answer": full_response
            })
            
            time.sleep(2) # Avoid rate limits during evaluation
            
        except Exception as e:
            print(f"Error processing query: {e}")
            continue

    print(f"\nEvaluating {len(test_cases)} test cases...")
    
    evaluator = RAGEvaluator()
    report = evaluator.evaluate(
        test_cases,
        k=5,
        run_semantic=True
    )

    RAGEvaluator.print_report(report)
    RAGEvaluator.save_report(report, "evaluation/synthetic_report.json")
    print(f"\nReport saved to evaluation/synthetic_report.json")

if __name__ == "__main__":
    run_evaluation()
