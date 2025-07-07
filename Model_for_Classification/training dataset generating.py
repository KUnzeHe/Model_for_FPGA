# 禁用警告
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.interpolate import interp2d
from tqdm import tqdm  # <-- 1. 导入tqdm库

def generate_lp_mode(m, n, r, phi, a=25e-6, NA=0.1, wavelength=1550e-9):
    """
    Generate LP mode field distribution for given mode numbers (m,n)
    
    Parameters:
        m, n : LP mode numbers (azimuthal, radial)
        r : radial coordinates array
        phi : angular coordinates array
        a : fiber core radius (25 μm as in paper)
        NA : numerical aperture (0.1 as in paper)
        wavelength : light wavelength (1550 nm as in paper)
    
    Returns:
        LP mode field distribution
    """
    # Normalized frequency V = (2πa/λ) * NA
    V = (2 * np.pi * a / wavelength) * NA
    
    if (m + 2*n) >= V:
        # 模式不可行，直接返回全零
        return np.zeros_like(r)
    
    # Find u (core parameter) by solving eigenvalue equation for LP modes
    # This is simplified - in practice would need to solve transcendental equation
    u = np.sqrt(V**2 - (m + 2*n)**2) if (m + 2*n) < V else V
    
    # LP mode field in core (r <= a)
    R = np.zeros_like(r)
    core_mask = r <= a
    R[core_mask] = jv(m, u*r[core_mask]/a) * np.cos(m*phi[core_mask])
    
    # Field in cladding (r > a) decays exponentially (approximation)
    w = np.sqrt(V**2 - u**2)  # cladding parameter
    cladding_mask = r > a
    R[cladding_mask] = (jv(m, u) * np.exp(-w*(r[cladding_mask]-a)/a) * 
                        np.cos(m*phi[cladding_mask]))
    
    return R

def generate_speckle_pattern(amplitudes, phases, modes, resolution=16):
    """
    Generate a speckle pattern from superposition of LP modes
    
    Parameters:
        amplitudes : array of amplitude weights for each mode
        phases : array of phase weights for each mode
        modes : list of (m,n) tuples representing LP modes
        resolution : output image resolution (16x16 as in paper)
    
    Returns:
        intensity image (speckle pattern)
    """
    # Create coordinate grid
    x = np.linspace(-30e-6, 30e-6, resolution*4)  # oversample for better accuracy
    y = np.linspace(-30e-6, 30e-6, resolution*4)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)
    phi = np.arctan2(yy, xx)
    
    # Initialize field
    E_field = np.zeros_like(r, dtype=complex)
    
    # Superpose all modes
    for i, (m, n) in enumerate(modes):
        lp_mode = generate_lp_mode(m, n, r, phi)
        E_field += amplitudes[i] * np.exp(1j * phases[i]) * lp_mode
    
    # Calculate intensity
    intensity = np.abs(E_field)**2
    
    # Downsample to target resolution
    if resolution != resolution*4:
        f = interp2d(x, y, intensity, kind='cubic')
        x_new = np.linspace(-30e-6, 30e-6, resolution)
        y_new = np.linspace(-30e-6, 30e-6, resolution)
        intensity = f(x_new, y_new)
    
    # Normalize intensity
    if np.max(intensity) > 0:
        intensity = intensity / np.max(intensity)
    else:
        intensity[:] = 0
    
    return intensity

def generate_training_data_for_regression(num_modes, num_samples, resolution=16, noise_level=0.05):
    """
    Generate training dataset with random mode combinations
    
    Parameters:
        num_modes : number of modes to consider
        num_samples : number of samples to generate
        resolution : image resolution (default 16x16)
        noise_level : standard deviation of Gaussian noise
    
    Returns:
        X : intensity images (num_samples, resolution, resolution)
        y_amp : amplitude weights (num_samples, num_modes)
        y_phase : phase weights (num_samples, num_modes-1)
    """
    # Define supported LP modes (simplified - actual modes depend on fiber parameters)
    # For a real implementation, would need to calculate which LP modes are supported
    modes = [(0,1)]  # LP01 mode (fundamental)
    if num_modes >= 2:
        modes.append((1,1))  # LP11 mode
    if num_modes >= 3:
        modes.append((2,1))  # LP21 mode
    if num_modes >= 4:
        modes.append((0,2))  # LP02 mode
    # Continue adding more modes as needed
    
    # Initialize arrays
    X = np.zeros((num_samples, resolution, resolution))
    y_amp = np.zeros((num_samples, num_modes))
    y_phase = np.zeros((num_samples, num_modes-1))  # phase relative to fundamental
    
    # <-- 2. 在循环中加入tqdm，并添加一个描述性的标题 (desc)
    for i in tqdm(range(num_samples), desc=f"Generating {num_modes}-modes regression data"):
        # Generate random amplitudes (normalized to sum of squares = 1)
        amplitudes = np.random.rand(num_modes)
        amplitudes = amplitudes / np.sqrt(np.sum(amplitudes**2))
        
        # Generate random phases with special handling to avoid ambiguity
        # Fundamental mode phase is 0 (reference)
        phases = np.zeros(num_modes)
        
        # First higher-order mode phase in [0, π] (as in paper)
        if num_modes >= 2:
            phases[1] = np.random.uniform(0, np.pi)
        
        # Other higher-order mode phases in [-π, π]
        if num_modes >= 3:
            phases[2:] = np.random.uniform(-np.pi, np.pi, num_modes-2)
        
        # Generate speckle pattern
        intensity = generate_speckle_pattern(amplitudes, phases, modes, resolution)
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_level, intensity.shape)
        intensity = np.clip(intensity + noise, 0, 1)
        
        # Store data
        X[i] = intensity
        y_amp[i] = amplitudes
        y_phase[i] = phases[1:]  # only store relative phases (fundamental is 0)
    
    return X, y_amp, y_phase

def generate_training_data_for_classification(num_modes, num_samples, resolution=16, noise_level=0.05):
    """
    生成分类任务所需的数据集，每张图片只激活一个模式
    返回：
      X: (num_samples, resolution, resolution)
      y: (num_samples, )  # 每个元素是int，类别编号
    """
    # 支持的模式列表

    '''modes = [(0,1), (1,1), (2,1), (0,2)][:num_modes] '''

    V = 2 * np.pi * 25e-6 / 1550e-9 * 0.1  # Normalized frequency for LP modes

    modes = []
    max_n = int(V // 2) + 1
    for m in range(0, int(V) + 1):
        for n in range(1, max_n):
            if (m + 2 * n) < V:
                modes.append((m, n))          
    modes = modes[:num_modes]  # 只取前num_modes个

    X = np.zeros((num_samples, resolution, resolution))
    y = np.zeros((num_samples,), dtype=np.int64)
    
    # <-- 2. 在循环中加入tqdm
    for i in tqdm(range(num_samples), desc=f"Generating {num_modes}-modes classification data"):
        mode_idx = np.random.randint(num_modes)  # 随机选一个模式
        amplitudes = np.zeros(num_modes)
        amplitudes[mode_idx] = 1.0  # 只激活一个模式
        phases = np.zeros(num_modes) # 分类任务下无须相位
        intensity = generate_speckle_pattern(amplitudes, phases, modes, resolution)
        noise = np.random.normal(0, noise_level, intensity.shape)
        intensity = np.clip(intensity + noise, 0, 1)
        X[i] = intensity
        y[i] = mode_idx
    return X, y
# Example usage:
if __name__ == "__main__":
    
    # 没有模式串扰下的分类任务数据集生成
    for num_modes in range(3, 41):

        num_samples_train = 5000
        num_samples_test = 1000
        resolution = 16
        
        # 生成训练集
        X_train, y_train = generate_training_data_for_classification(num_modes, num_samples_train, resolution)
        np.savez(f'datasets\{num_modes}modes_mode_decomposition_dataset_train_classification.npz', X=X_train, y=y_train)
        # 生成测试集
        X_test, y_test = generate_training_data_for_classification(num_modes, num_samples_test, resolution)
        np.savez(f'datasets\{num_modes}modes_mode_decomposition_dataset_test_classification.npz', X=X_test, y=y_test)
    
    print("All training and test datasets generated successfully.")
