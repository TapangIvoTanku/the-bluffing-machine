"""Quick test of the simulation engine — 5 games per model."""
import sys
sys.path.insert(0, '/home/ubuntu/bluffing_machine')
from simulation_engine import run_single_game, MODELS

print("=== QUICK ENGINE TEST ===\n")

for model_key, model_name in MODELS.items():
    print(f"Testing {model_name}...")
    for i in range(3):
        result = run_single_game(model_key, model_key, treatment="zero_shot")
        print(f"  Game {i+1}: Type={result['sender_type']} | Signal={result['signal']} | "
              f"Action={result['action']} | Bluff={result['is_bluff']} | "
              f"Success={result['bluff_success']} | Posterior={result['posterior_belief']:.2f}")
        print(f"    Sender reasoning: {result['sender_reasoning'][:80]}...")
    print()

print("=== TEST PASSED ===")
