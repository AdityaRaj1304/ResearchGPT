import os
import sys

# Ensure we can import from src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(project_root)

# We need to make sure the imports match the actual file structure
# evaluate_retrieval is in src.models
# statistical_tests is in src.evaluation
# plot_benchmark_results is in src.evaluation
# Note: we wrote them so they can be run independently, we can also just run them as scripts using os.system
# or import them since we designed them with run functions

def main():
    print("="*60)
    print("PHASE 5: AUTOMATED BENCHMARK & EVALUATION SUITE")
    print("="*60)
    
    print("\n[1/3] Running Retrieval & Generation Benchmarks...")
    ret1 = os.system(f'"{sys.executable}" ' + os.path.join('src', 'models', 'evaluate_retrieval.py'))
    if ret1 != 0:
        print("Benchmark evaluation failed!")
        sys.exit(1)
        
    print("\n[2/3] Executing Statistical Significance Tests...")
    ret2 = os.system(f'"{sys.executable}" ' + os.path.join('src', 'evaluation', 'statistical_tests.py'))
    if ret2 != 0:
        print("Statistical tests failed!")
        sys.exit(1)
        
    print("\n[3/3] Generating Visualizations...")
    ret3 = os.system(f'"{sys.executable}" ' + os.path.join('src', 'evaluation', 'plot_benchmark_results.py'))
    if ret3 != 0:
        print("Visualization generation failed!")
        sys.exit(1)
        
    print("\n" + "="*60)
    print("Phase 5 Evaluation Pipeline completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
