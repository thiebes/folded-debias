# Caption: distribution of fitted width (benefit and limitation)

The ensemble picture over 5000 simulated CNR = 1.5 frames of a known
Gaussian (true sigma^2 = 9). The naive fitted-width distribution (red) is
right-skewed and biased high *in the mean* (mean 15.4 = 1.7x true; the median is
actually below true -- the bias lives in a heavy right tail plus a pile-up at the
fit's upper bound). Subtracting the single calibrated mean bias recenters the
corrected distribution (blue) mean on truth (8.7 ~ 9.0).

It also shows the limitation: at CNR 1.5 about 44% of individual frames
over-subtract below zero. So the correction debiases the *mean* (and hence D, a
mean-trend slope over many frames), not every individual near-floor frame -- which
is exactly why the per-frame CNR floor (~1.3) exists.

Rendered by `render_recovery_demo.py`.
