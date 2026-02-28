#!/usr/bin/env python3
"""
Example script for running attribution analysis.
Shows how to use both local and Modal-based attribution.
"""
import argparse
from pathlib import Path

# 延迟导入，只在需要时导入
# import torch
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from tone_classifier.attribution import sentence_level_attribution
# from tone_classifier.attention_attribution import aggregate_attention_across_layers


def main():
    parser = argparse.ArgumentParser(
        description="Run attribution analysis example"
    )
    parser.add_argument(
        "--hf_model_dir",
        type=str,
        required=True,
        help="Path to fine-tuned HuggingFace model directory",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Could you please help me with this issue when you have time?",
        help="Input text to analyze",
    )
    parser.add_argument(
        "--num_ablations",
        type=int,
        default=512,
        help="Number of ablation experiments (TA建议256或512)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["context-cite", "attention", "both"],
        default="both",
        help="Attribution method to use",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/attribution",
        help="Output directory for results",
    )
    parser.add_argument(
        "--use_modal",
        action="store_true",
        help="Use Modal for GPU resources",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.use_modal:
        print("Using Modal for GPU resources...")
        try:
            from modal_app import app, run_attribution_analysis, run_attention_attribution
        except ImportError as e:
            print(f"Error importing Modal: {e}")
            print("Make sure Modal is installed: pip install modal")
            return
        
        # 使用 with app.run() 来启动Modal app
        with app.run():
            if args.method in ["context-cite", "both"]:
                print("Running context-cite attribution on Modal...")
                try:
                    results = run_attribution_analysis.remote(
                        model_path=args.hf_model_dir,
                        text=args.text,
                        num_ablations=args.num_ablations,
                    )
                    output_file = output_dir / "context_cite_results.json"
                    import json
                    with output_file.open("w", encoding="utf-8") as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                    print(f"Results saved to: {output_file}")
                except Exception as e:
                    print(f"Error running context-cite attribution: {e}")
                    import traceback
                    traceback.print_exc()
            
            if args.method in ["attention", "both"]:
                print("Running attention attribution on Modal...")
                try:
                    results = run_attention_attribution.remote(
                        model_path=args.hf_model_dir,
                        text=args.text,
                    )
                    output_file = output_dir / "attention_results.json"
                    import json
                    with output_file.open("w", encoding="utf-8") as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                    print(f"Results saved to: {output_file}")
                except Exception as e:
                    print(f"Error running attention attribution: {e}")
                    import traceback
                    traceback.print_exc()
    
    else:
        # 只在本地运行时导入这些模块
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from tone_classifier.attribution import sentence_level_attribution
        from tone_classifier.attention_attribution import aggregate_attention_across_layers
        
        print("Running attribution locally...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Load model
        print(f"Loading model from {args.hf_model_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(args.hf_model_dir)
        model.to(device)
        model.eval()
        
        if args.method in ["context-cite", "both"]:
            print(f"\nRunning context-cite attribution with {args.num_ablations} ablations...")
            results = sentence_level_attribution(
                model=model,
                tokenizer=tokenizer,
                text=args.text,
                num_ablations=args.num_ablations,
                device=device,
            )
            
            output_file = output_dir / "context_cite_results.json"
            import json
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Context-cite results saved to: {output_file}")
            
            # Print summary
            print("\n" + "=" * 80)
            print("CONTEXT-CITE ATTRIBUTION RESULTS")
            print("=" * 80)
            print(f"Baseline: {results['baseline_prediction']['label']}")
            print("\nTop 3 sentences:")
            for i, sent in enumerate(results['sentences'][:3], 1):
                print(f"{i}. {sent['sentence'][:80]}...")
                print(f"   Score: {sent['attribution_score']:.4f}")
        
        if args.method in ["attention", "both"]:
            print("\nRunning attention-based attribution...")
            results = aggregate_attention_across_layers(
                model=model,
                tokenizer=tokenizer,
                text=args.text,
                device=device,
            )
            
            output_file = output_dir / "attention_results.json"
            import json
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Attention results saved to: {output_file}")
            
            # Print summary
            print("\n" + "=" * 80)
            print("ATTENTION-BASED ATTRIBUTION RESULTS")
            print("=" * 80)
            print("\nTop 10 tokens:")
            token_scores = list(zip(results['tokens'], results['token_importance']))
            token_scores.sort(key=lambda x: x[1], reverse=True)
            for i, (token, score) in enumerate(token_scores[:10], 1):
                print(f"{i}. {token}: {score:.4f}")


if __name__ == "__main__":
    main()