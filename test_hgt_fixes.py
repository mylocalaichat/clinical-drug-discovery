"""
Quick test script to verify HGT embedding fixes.

Tests with a small subset of nodes to quickly validate:
1. Full embedding dimensions (512) are preserved
2. Training converges properly
3. Embeddings show variance and separation
"""

import sys
from pathlib import Path
from clinical_drug_discovery.lib.gnn_hgt import generate_hgt_embeddings

def test_hgt_fixes():
    """Test HGT embeddings with fixes applied."""

    print("=" * 80)
    print("TESTING HGT EMBEDDING FIXES")
    print("=" * 80)

    # Configuration
    edges_csv = "data/01_raw/kg.csv"
    output_csv = "data/06_models/embeddings/test_hgt_embeddings.csv"

    # Small test: 5000 nodes, 20 epochs to validate quickly
    print("\n📊 Test Configuration:")
    print(f"  • Nodes: 5,000 (subset for quick validation)")
    print(f"  • Epochs: 20 (reduced for speed)")
    print(f"  • Embedding dim: 512 (FULL SIZE - not auto-scaled)")
    print(f"  • Device: cpu (to prevent auto-scaling)")
    print(f"  • Edge sample size: 5,000 (5x increase from 1,000)")

    print("\n🔬 Running training...")

    stats = generate_hgt_embeddings(
        edges_csv=edges_csv,
        output_csv=output_csv,
        embedding_dim=512,      # Full size
        hidden_dim=256,
        num_layers=2,
        num_heads=8,
        num_epochs=20,          # Quick test
        learning_rate=0.001,
        device="cpu",           # Force CPU to allow full dims
        limit_nodes=5000,       # Small subset for testing
        edge_sample_size=5000,  # Increased from 1000
        contrastive_weight=0.5,
        similarity_threshold=0.1,
        include_node_types=[
            'drug', 'disease', 'gene/protein', 'pathway', 'biological_process'
        ]
    )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Embedding dim: {stats['embedding_dim']}")
    print(f"Output file: {stats['output_file']}")

    # Quick validation
    print("\n🧪 Running quick validation...")
    import subprocess
    result = subprocess.run(
        ["uv", "run", "python", "validate_hgt_embeddings.py", output_csv],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("✓ Validation completed")

        # Parse output for key metrics
        output = result.stdout
        if "Mean: 0." in output:
            # Extract mean similarity
            for line in output.split('\n'):
                if 'Cosine Similarity Statistics:' in line:
                    print("\n📈 Key Metrics:")
                if '  Mean:' in line and 'Cosine' not in line:
                    print(line)
                if '  90% variance explained by:' in line:
                    print(line)

        # Check for issues
        if "CRITICAL" in output:
            print("\n❌ CRITICAL issues still detected - embeddings may still be collapsing")
            print("   Consider:")
            print("   • Increasing epochs further (50-100)")
            print("   • Checking loss convergence")
        elif "WARNING" in output:
            print("\n⚠️  Some warnings detected - review validation output")
        else:
            print("\n✅ No critical issues detected!")
    else:
        print(f"❌ Validation failed: {result.stderr}")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("If test looks good:")
    print("  1. Run full training: dagster dev (or use CLI)")
    print("  2. Monitor for ~100 epochs")
    print("  3. Validate final embeddings")
    print("\nIf issues persist:")
    print("  • Increase epochs to 100-200")
    print("  • Monitor loss convergence")
    print("  • Check node features")

if __name__ == "__main__":
    test_hgt_fixes()
