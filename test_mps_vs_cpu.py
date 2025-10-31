"""
Compare MPS (batched) vs CPU training performance.

Tests both approaches with a small subset to measure:
- Training speed
- Memory usage
- Embedding quality
"""

import time
import torch
from pathlib import Path
from clinical_drug_discovery.lib.gnn_hgt import generate_hgt_embeddings
from clinical_drug_discovery.lib.gnn_hgt_batched import generate_hgt_embeddings_batched


def test_cpu_training():
    """Test CPU training (original fix)."""
    print("\n" + "=" * 80)
    print("TEST 1: CPU TRAINING")
    print("=" * 80)

    start = time.time()

    stats = generate_hgt_embeddings(
        edges_csv="data/01_raw/kg.csv",
        output_csv="data/06_models/embeddings/test_cpu_hgt_embeddings.csv",
        embedding_dim=512,
        hidden_dim=256,
        num_layers=2,
        num_heads=8,
        num_epochs=20,      # Quick test
        learning_rate=0.001,
        device="cpu",
        limit_nodes=5000,
        edge_sample_size=5000,
        contrastive_weight=0.5,
        similarity_threshold=0.1,
        include_node_types=['drug', 'disease', 'gene/protein', 'pathway', 'biological_process']
    )

    elapsed = time.time() - start

    print("\n" + "=" * 80)
    print("CPU TRAINING RESULTS")
    print("=" * 80)
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Time per epoch: {elapsed/20:.1f}s")
    print(f"Embedding dim: {stats['embedding_dim']}")
    print(f"Total nodes: {stats['total_nodes']}")

    return {
        'method': 'CPU',
        'time': elapsed,
        'time_per_epoch': elapsed / 20,
        'embedding_dim': stats['embedding_dim'],
        'output_file': stats['output_file']
    }


def test_mps_batched_training():
    """Test MPS with batched training."""
    print("\n" + "=" * 80)
    print("TEST 2: MPS BATCHED TRAINING")
    print("=" * 80)

    if not torch.backends.mps.is_available():
        print("❌ MPS not available on this system")
        return None

    start = time.time()

    stats = generate_hgt_embeddings_batched(
        edges_csv="data/01_raw/kg.csv",
        output_csv="data/06_models/embeddings/test_mps_hgt_embeddings.csv",
        embedding_dim=512,
        hidden_dim=256,
        num_layers=2,
        num_heads=8,
        num_epochs=20,      # Quick test
        learning_rate=0.001,
        device=None,        # Auto-detect MPS
        limit_nodes=5000,
        edge_sample_size=5000,
        node_batch_size=2048,
        accumulation_steps=4,
        contrastive_weight=0.5,
        similarity_threshold=0.1,
        include_node_types=['drug', 'disease', 'gene/protein', 'pathway', 'biological_process']
    )

    elapsed = time.time() - start

    print("\n" + "=" * 80)
    print("MPS BATCHED TRAINING RESULTS")
    print("=" * 80)
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Time per epoch: {elapsed/20:.1f}s")
    print(f"Embedding dim: {stats['embedding_dim']}")
    print(f"Total nodes: {stats['total_nodes']}")

    return {
        'method': 'MPS (batched)',
        'time': elapsed,
        'time_per_epoch': elapsed / 20,
        'embedding_dim': stats['embedding_dim'],
        'output_file': stats['output_file']
    }


def validate_embeddings(output_file: str, method: str):
    """Quick validation of embeddings."""
    print(f"\n🧪 Validating {method} embeddings...")

    import subprocess
    result = subprocess.run(
        ["uv", "run", "python", "validate_hgt_embeddings.py", output_file],
        capture_output=True,
        text=True,
        timeout=60
    )

    if result.returncode == 0:
        output = result.stdout

        # Extract key metrics
        mean_sim = None
        pca_components = None

        for line in output.split('\n'):
            if '  Mean:' in line and mean_sim is None:
                try:
                    mean_sim = float(line.split(':')[1].strip())
                except:
                    pass
            if '  90% variance explained by:' in line:
                try:
                    pca_components = int(line.split(':')[1].split()[0])
                except:
                    pass

        return {
            'mean_similarity': mean_sim,
            'pca_components_90': pca_components,
            'has_issues': 'CRITICAL' in output or 'WARNING' in output
        }
    else:
        print(f"❌ Validation failed: {result.stderr}")
        return None


def compare_results(cpu_results, mps_results):
    """Compare CPU vs MPS results."""
    print("\n" + "=" * 80)
    print("COMPARISON: CPU vs MPS BATCHED")
    print("=" * 80)

    if mps_results is None:
        print("MPS not available for comparison")
        return

    print("\n📊 Training Time:")
    print(f"  CPU:         {cpu_results['time']:.1f}s ({cpu_results['time']/60:.1f} min)")
    print(f"  MPS:         {mps_results['time']:.1f}s ({mps_results['time']/60:.1f} min)")

    speedup = cpu_results['time'] / mps_results['time']
    if speedup > 1:
        print(f"  Speedup:     {speedup:.2f}x faster on MPS ✨")
    else:
        print(f"  Speedup:     {1/speedup:.2f}x slower on MPS")

    print("\n📐 Embedding Dimensions:")
    print(f"  CPU:         {cpu_results['embedding_dim']}")
    print(f"  MPS:         {mps_results['embedding_dim']}")

    # Validate both
    cpu_val = validate_embeddings(cpu_results['output_file'], "CPU")
    mps_val = validate_embeddings(mps_results['output_file'], "MPS")

    if cpu_val and mps_val:
        print("\n🎯 Embedding Quality:")
        print(f"  CPU mean similarity:  {cpu_val['mean_similarity']:.4f}")
        print(f"  MPS mean similarity:  {mps_val['mean_similarity']:.4f}")
        print(f"  CPU PCA components:   {cpu_val['pca_components_90']}")
        print(f"  MPS PCA components:   {mps_val['pca_components_90']}")

        if cpu_val['has_issues'] and mps_val['has_issues']:
            print("\n  ⚠️  Both have issues - may need more epochs")
        elif cpu_val['has_issues']:
            print("\n  ⚠️  CPU has issues, MPS looks better")
        elif mps_val['has_issues']:
            print("\n  ⚠️  MPS has issues, CPU looks better")
        else:
            print("\n  ✅ Both look good!")

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if mps_results is None:
        print("Use CPU training (MPS not available)")
    elif speedup > 1.5:
        print(f"✅ Use MPS batched training ({speedup:.1f}x faster!)")
        print("   Set use_batched_mps = True in embeddings.py")
    elif speedup > 1.1:
        print(f"✅ Use MPS batched training ({speedup:.1f}x faster)")
        print("   Moderate speedup, but worthwhile")
    else:
        print("⚠️  CPU is competitive or faster")
        print("   Consider CPU for stability, or tune MPS batching")

    print("\n💡 For full training (100 epochs):")
    if mps_results:
        full_mps = mps_results['time'] * 5  # 20 epochs -> 100 epochs
        full_cpu = cpu_results['time'] * 5
        print(f"   MPS: ~{full_mps/60:.1f} min ({full_mps/3600:.1f} hours)")
        print(f"   CPU: ~{full_cpu/60:.1f} min ({full_cpu/3600:.1f} hours)")


def main():
    """Run comparison tests."""
    print("=" * 80)
    print("MPS vs CPU TRAINING COMPARISON")
    print("=" * 80)
    print("\nThis will test both approaches with:")
    print("  • 5,000 nodes (subset)")
    print("  • 20 epochs (quick test)")
    print("  • Full 512 dimensions")
    print("\nExpected time: 10-20 minutes total\n")

    # Test CPU
    cpu_results = test_cpu_training()

    # Test MPS
    mps_results = test_mps_batched_training()

    # Compare
    compare_results(cpu_results, mps_results)

    print("\n✅ Comparison complete!")


if __name__ == "__main__":
    main()
