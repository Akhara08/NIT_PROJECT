VAE Curve Generation and Evaluation
This project uses a Variational Autoencoder (VAE) to learn the features of 2D time-series curves (called phrases) from a dataset.
It can generate new synthetic curves and evaluate their authenticity using:

Dynamic Time Warping (DTW)

Automated curve segmentation via Change Point Detection and Knee Locator.

📦 Requirements
Install the required Python packages:

bash
Copy
Edit
pip install torch numpy scikit-learn ruptures kneed fastdtw matplotlib
📂 Data Setup
Place your data file (e.g., 1R.txt) in the same directory as the script.

▶️ How to Run
bash
Copy
Edit
python evaluate_curves.py
🧠 Evaluation Methodology
The evaluation checks if a generated curve’s segments are statistically similar to reference phrases from the training data.

1. Curve Segmentation
We automatically segment curves using:

Change Point Detection (CPD)

Uses ruptures.KernelCPD with an RBF kernel for detecting points where statistical properties change significantly.

Captures complex, non-linear shifts in the curve’s behavior.

Knee Locator (Elbow Method)

Finds the optimal number of breakpoints to avoid over/under-segmentation.

Prevents meaningless tiny segments.

Function:

python
Copy
Edit
segments = self._segment_curve_with_change_points(curve, max_bkps=10)
2. Building the Benchmark
We create a statistical benchmark from reference phrases:

Segment Reference Curves using the same CPD + Knee Locator method.

Compute DTW Distances for each segment index across all reference phrases.

Example: For segment #1, compute DTW between all phrase pairs’ segment #1.

Calculate Statistics: Mean and variance of DTW distances per segment.

Function:

python
Copy
Edit
evaluator.calculate_reference_stats()
3. Evaluating a Generated Phrase
Given a generated phrase G:

Segment G with the same CPD + Knee Locator method.

Compute Mean DTW for each segment against all corresponding reference segments.

Check Allowed Range:

A segment is valid if mean_distance is within mean ± 2 × std from the reference benchmark.

Final Verdict:

If all segments pass → G is considered authentic.

If any segment fails → G is flagged as not authentic.

Function:

python
Copy
Edit
is_good = evaluator.evaluate_curve_authenticity(generated_curve)
📊 Summary
✅ Automated Segmentation – No manual breakpoints needed.
✅ Statistical DTW Benchmark – Measures similarity with variance tolerance.
✅ One-Call Evaluation – Minimal code to check curve authenticity.
