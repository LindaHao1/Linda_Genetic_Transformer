"""
Cost function for filter optimization based on S-parameters.
"""

import numpy as np
import skrf as rf
from typing import Tuple


def calculate_filter_cost(
    s2p_path: str,
    center_freq_ghz: float = 35.0,
    bandwidth_percent: float = 0.25,
    passband_loss_target_db: float = 5.0,
    stopband_loss_target_db: float = 10.0,
    stopband_offset_ghz: float = 4.0,
    stopband_width_ghz: float = 10.0
) -> float:
    """
    Calculate cost for a bandpass filter based on S21/S12 parameters.
    
    The cost function rewards:
    - Low insertion loss (< target dB) in the passband
    - High isolation (> target dB) in the stopband
    
    Args:
        s2p_path: Path to the .s2p file
        center_freq_ghz: Center frequency in GHz (default: 35 GHz)
        bandwidth_percent: Fractional bandwidth (default: 0.25 = 25%)
        passband_loss_target_db: Target max loss in passband (default: 5 dB)
        stopband_loss_target_db: Target min loss in stopband (default: 10 dB)
        stopband_offset_ghz: Gap between passband edge and stopband (default: 4 GHz)
        stopband_width_ghz: Width of each stopband region (default: 10 GHz)
        
    Returns:
        Cost value (lower is better, 0 is perfect)
    """
    try:
        # Load the 2-port network
        net = rf.Network(s2p_path)
        
        # Get frequencies in GHz
        freqs_ghz = net.frequency.f / 1e9
        
        # Calculate passband boundaries
        bw_ghz = center_freq_ghz * bandwidth_percent
        passband_min = center_freq_ghz - bw_ghz / 2
        passband_max = center_freq_ghz + bw_ghz / 2
        
        # Calculate stopband regions dynamically
        # Lower stopband: offset below passband, with specified width
        lower_stopband_max = passband_min - stopband_offset_ghz
        lower_stopband_min = lower_stopband_max - stopband_width_ghz
        # Upper stopband: offset above passband, with specified width
        upper_stopband_min = passband_max + stopband_offset_ghz
        upper_stopband_max = upper_stopband_min + stopband_width_ghz
        
        # Get S21 magnitude in dB (insertion loss)
        s21_db = 20 * np.log10(np.abs(net.s[:, 1, 0]))
        
        # Create masks for passband and stopband
        passband_mask = (freqs_ghz >= passband_min) & (freqs_ghz <= passband_max)
        lower_stopband_mask = (freqs_ghz >= lower_stopband_min) & (freqs_ghz <= lower_stopband_max)
        upper_stopband_mask = (freqs_ghz >= upper_stopband_min) & (freqs_ghz <= upper_stopband_max)
        stopband_mask = lower_stopband_mask | upper_stopband_mask
        
        if not np.any(passband_mask):
            return 1e6  # Huge penalty if no passband points
        
        # Insertion loss in passband (should be close to 0 dB, ideally < target)
        passband_loss = -s21_db[passband_mask]  # Convert to positive loss
        
        # Stopband rejection (should be large negative, ideally < -target dB)
        stopband_loss = -s21_db[stopband_mask]  # Convert to positive loss
        lower_stopband_loss = -s21_db[lower_stopband_mask] if np.any(lower_stopband_mask) else np.array([0])
        upper_stopband_loss = -s21_db[upper_stopband_mask] if np.any(upper_stopband_mask) else np.array([0])
        
        # Cost components
        # 1. Passband penalty: penalize loss > target
        passband_violations = np.maximum(0, passband_loss - passband_loss_target_db)
        passband_cost = np.mean(passband_violations) ** 2
        
        # Also penalize average loss even if below target (want it close to 0)
        passband_avg_cost = np.mean(passband_loss) ** 2
        
        # 2. Stopband penalty: penalize rejection < target (separate lower/upper)
        lower_violations = np.maximum(0, stopband_loss_target_db - lower_stopband_loss)
        upper_violations = np.maximum(0, stopband_loss_target_db - upper_stopband_loss)
        lower_stopband_cost = np.mean(lower_violations) ** 2 if len(lower_violations) > 0 else 0
        upper_stopband_cost = np.mean(upper_violations) ** 2 if len(upper_violations) > 0 else 0
        
        # Also minimize average stopband loss even when meeting threshold
        lower_avg = np.mean(lower_stopband_loss) if len(lower_stopband_loss) > 0 else 0
        upper_avg = np.mean(upper_stopband_loss) if len(upper_stopband_loss) > 0 else 0
        # Inverse penalty: reward higher stopband loss (better rejection)
        # Using dB directly - normalize by typical target (10-30 dB range)
        lower_improvement = np.maximum(0, 30.0 - lower_avg) / 30.0 if lower_avg > 0 else 1.0
        upper_improvement = np.maximum(0, 30.0 - upper_avg) / 30.0 if upper_avg > 0 else 1.0
        
        # 3. Flatness penalty in passband (reward flat response)
        passband_std = np.std(passband_loss)
        flatness_cost = passband_std ** 2
        
        # Weighted total cost
        # Upper stopband gets 2x weight since it's performing worse
        total_cost = (
            1.0 * passband_cost +           # Primary: meet passband target
            0.3 * passband_avg_cost +       # Secondary: minimize passband loss
            1.5 * lower_stopband_cost +     # Lower stopband violations
            # 1.5 * upper_stopband_cost +     # Upper stopband violations 
            1.0 * lower_improvement       # Bonus: maximize lower stopband
            # 0.5 * upper_improvement +        # Bonus: maximize upper stopband (2x)
            # 0.2 * flatness_cost             # Bonus: flat passband
        )
        
        return total_cost
        
    except Exception as e:
        print(f"Error calculating cost from {s2p_path}: {e}")
        return 1e6  # Return huge penalty for failed evaluations


def analyze_filter_performance(
    s2p_path: str,
    center_freq_ghz: float = 35.0,
    bandwidth_percent: float = 0.25,
    stopband_offset_ghz: float = 4.0,
    stopband_width_ghz: float = 10.0
) -> dict:
    """
    Detailed analysis of filter performance.
    
    Args:
        s2p_path: Path to the .s2p file
        center_freq_ghz: Center frequency in GHz
        bandwidth_percent: Fractional bandwidth
        
    Returns:
        Dictionary with performance metrics
    """
    try:
        net = rf.Network(s2p_path)
        freqs_ghz = net.frequency.f / 1e9
        
        # Calculate passband
        bw_ghz = center_freq_ghz * bandwidth_percent
        passband_min = center_freq_ghz - bw_ghz / 2
        passband_max = center_freq_ghz + bw_ghz / 2
        
        # S-parameters in dB
        s21_db = 20 * np.log10(np.abs(net.s[:, 1, 0]))
        s11_db = 20 * np.log10(np.abs(net.s[:, 0, 0]))
        
        # Masks
        passband_mask = (freqs_ghz >= passband_min) & (freqs_ghz <= passband_max)
        
        # Passband metrics
        passband_loss = -s21_db[passband_mask]
        passband_loss_avg = np.mean(passband_loss)
        passband_loss_max = np.max(passband_loss)
        passband_loss_min = np.min(passband_loss)
        passband_ripple = passband_loss_max - passband_loss_min
        
        # Return loss in passband
        passband_return_loss = -s11_db[passband_mask]
        passband_return_loss_min = np.min(passband_return_loss)
        
        # Find 3dB bandwidth
        idx_center = np.argmin(np.abs(freqs_ghz - center_freq_ghz))
        loss_at_center = -s21_db[idx_center]
        target_3db = loss_at_center + 3
        
        bw_3db_mask = -s21_db <= target_3db
        if np.any(bw_3db_mask):
            bw_3db_freqs = freqs_ghz[bw_3db_mask]
            bw_3db = bw_3db_freqs[-1] - bw_3db_freqs[0]
        else:
            bw_3db = 0.0
        
        return {
            'cost': calculate_filter_cost(s2p_path, center_freq_ghz, bandwidth_percent,
                                        stopband_offset_ghz=stopband_offset_ghz,
                                        stopband_width_ghz=stopband_width_ghz),
            'passband_loss_avg_db': passband_loss_avg,
            'passband_loss_max_db': passband_loss_max,
            'passband_loss_min_db': passband_loss_min,
            'passband_ripple_db': passband_ripple,
            'passband_return_loss_min_db': passband_return_loss_min,
            'center_freq_ghz': center_freq_ghz,
            'target_bandwidth_ghz': bw_ghz,
            'bw_3db_ghz': bw_3db,
            'loss_at_center_db': loss_at_center,
        }
        
    except Exception as e:
        print(f"Error analyzing {s2p_path}: {e}")
        return {'cost': 1e6, 'error': str(e)}


if __name__ == "__main__":
    # Test with an example file
    import sys
    
    if len(sys.argv) > 1:
        s2p_file = sys.argv[1]
        print(f"Analyzing: {s2p_file}")
        
        cost = calculate_filter_cost(s2p_file)
        print(f"\nCost: {cost:.4f}")
        
        analysis = analyze_filter_performance(s2p_file)
        print("\nDetailed Analysis:")
        for key, value in analysis.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
    else:
        print("Usage: python cost_function.py <path_to_s2p_file>")
