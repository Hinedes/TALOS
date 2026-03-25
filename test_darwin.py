import numpy as np
from pathlib import Path
from darwin import DarwinEngine

def test_diagnostician():
    print("--- Testing Diagnostician ---")
    darwin = DarwinEngine()
    
    # Fake history: high slap rate (too tight)
    history_slap = [{'slap_rate_pct': 50.0}] * 5
    diag = darwin.diagnose(history_slap)
    print("High Slap Rate Diagnosis:", diag)
    assert 'SLAP_THRESHOLD' in diag and diag['SLAP_THRESHOLD'][0] == 1, "Should widen slap threshold"

    # Fake history: high cage rate (too much drift)
    history_cage = [{'cage_clamp_rate_pct': 40.0, 'pred_gt_speed_ratio': 1.0}] * 5
    diag = darwin.diagnose(history_cage)
    print("High Cage Rate Diagnosis:", diag)
    assert 'R_OBS_FIXED_DIAG' in diag and diag['R_OBS_FIXED_DIAG'][0] == 1, "Should lose trust in neural"

    # Fake history: scale collapse
    history_collapse = [{'pred_gt_speed_ratio': 0.3}] * 5
    diag = darwin.diagnose(history_collapse)
    print("Scale Collapse Diagnosis:", diag)
    assert 'PRED_VEL_GAIN' in diag and diag['PRED_VEL_GAIN'][0] == 1, "Should boost vel gain"

    print("Diagnostician tests passed!\n")

def test_mutation():
    print("--- Testing Mutation ---")
    darwin = DarwinEngine(population_size=10, seed=123)
    parent = darwin._get_defaults()
    
    # Diagnose scale collapse to force PRED_VEL_GAIN up
    diagnosis = {'PRED_VEL_GAIN': (1, 0.9)}
    mutants = darwin.spawn_mutants(parent, diagnosis)
    
    gains = [m['PRED_VEL_GAIN'] for m in mutants]
    print(f"Parent PRED_VEL_GAIN: {parent['PRED_VEL_GAIN']}")
    print(f"Mutant PRED_VEL_GAINs: {gains}")
    
    # With a strong positive diagnostic weight, most mutants should have higher gain
    higher_count = sum(1 for g in gains if g > parent['PRED_VEL_GAIN'])
    print(f"Mutants with higher gain: {higher_count}/{len(gains)}")
    assert higher_count >= len(gains) / 2, "Diagnostic bias failed to push values up"
    
    print("Mutation tests passed!\n")

def test_evolution_cycle():
    print("--- Testing Evolution Cycle ---")
    darwin = DarwinEngine(population_size=5, seed=456)
    
    # Dummy fitness function: wants PRED_VEL_GAIN to equal 1.5
    def mock_eval(params):
        return abs(params['PRED_VEL_GAIN'] - 1.5)
        
    history = [{'pred_gt_speed_ratio': 0.5}] * 5 # encourages higher gain
    
    parent = darwin._get_defaults()
    run_dir = Path('./')
    
    print("Starting Evolution...")
    winner = darwin.evolve(mock_eval, parent, history, run_dir)
    
    print(f"Parent Gain: {parent['PRED_VEL_GAIN']:.3f} | Error: {abs(parent['PRED_VEL_GAIN'] - 1.5):.3f}")
    print(f"Winner Gain: {winner['PRED_VEL_GAIN']:.3f} | Error: {abs(winner['PRED_VEL_GAIN'] - 1.5):.3f}")
    
    # Winner should be closer to 1.5 than parent (1.0)
    assert abs(winner['PRED_VEL_GAIN'] - 1.5) < abs(parent['PRED_VEL_GAIN'] - 1.5)
    
    # Clean up logs
    log_file = run_dir / 'darwin_log.json'
    cfg_file = run_dir / 'darwin_config.json'
    if log_file.exists(): log_file.unlink()
    if cfg_file.exists(): cfg_file.unlink()

    print("Evolution cycle tests passed!\n")

if __name__ == '__main__':
    try:
        test_diagnostician()
        test_mutation()
        test_evolution_cycle()
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"Test failed: {e}")
