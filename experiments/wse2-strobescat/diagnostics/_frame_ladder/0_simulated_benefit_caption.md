# Caption: simulated recovery from an unusable-looking frame

A single representative low-CNR frame (CNR = 1.5) from simulation, where the true
width is known. The data (gray) look qualitatively unusable -- scatter as large
as the peak, points swinging negative -- yet the bounded-Gaussian *naive* fit is
biased ~1.7x too wide (sigma^2: true 9.0 -> naive 16.0), and subtracting the
calibrated mean bias recovers the true width (corrected ~9.3). All three curves
are drawn at one amplitude so only the width differs.

This is a *representative* frame (naive near the ensemble mean), not a
cherry-picked tail; the rigorous statement is the companion distribution panel
(`0b_simulated_distribution.png`). Simulation carries this "recovers the true
width" claim because real data has no ground truth.

Rendered by `render_recovery_demo.py`.
