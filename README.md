# VAE Curve Generation and Evaluation

This project uses a Variational Autoencoder (VAE) to learn the features of 2D time-series curves (also referred to as "phrases") from a dataset. It can then generate new, synthetic curves and evaluate their authenticity using Dynamic Time Warping (DTW) and an automatic segmentation strategy.

---

## How to Run

1. Ensure you have the required Python dependencies:
   ```bash
   pip install torch numpy scikit-learn ruptures kneed fastdtw matplotlib
   ```
2. Place your data file (e.g., `1R.txt`) in the same directory as the script.
3. Run the script from your terminal:
   ```bash
   python evaluate_curves.py
   ```

---

## Evaluation Methodology

The authenticity of a generated curve is determined by comparing it to the ground truth "reference phrases" from the training data. The core idea is to check if the generated curve's segments are statistically similar to the corresponding segments of the reference curves.

### How is the curve segmented?

Instead of using manual breakpoints, the script employs a two-part automated approach to find the most meaningful segments in a curve.

#### 1. Change Point Detection (CPD)
This technique automatically detects points in a sequence where the statistical properties change significantly.
- The script uses `ruptures.KernelCPD` with a Radial Basis Function (`rbf`) kernel.
- The RBF kernel is powerful because it can capture complex, **nonlinear changes** in the curve's behavior, making it ideal for more than just simple linear shifts.

#### 2. Knee Locator (Elbow Method)
While CPD can find any number of change points, we need to select the *optimal* number. Adding too many breakpoints leads to "over-segmentation" with meaningless tiny segments.
- The Knee Locator algorithm is used to find the "knee" or "elbow point" in the cost function of the change point detection.
- This point represents the sweet spot where adding more breakpoints provides diminishing returns, thus preventing over- and under-segmentation.

**Function Call:**
```python
segments = self._segment_curve_with_change_points(curve, max_bkps=10)
```

---

## Computing Allowed Variations (The Benchmark)

Before we can evaluate a new curve, we first need to build a statistical benchmark from the reference phrases.

- **Segment Reference Curves:** Every curve in the reference dataset is broken into segments using the Change Point Detection and Knee Locator method described above. This ensures all reference curves are segmented consistently.
- **Calculate Segment-wise DTW Distances:** For each segment index (e.g., "segment 1", "segment 2", etc.), we calculate the DTW distance between every possible pair of corresponding segments from the reference phrases.

Example:  
If there are 10 reference phrases, for segment #1, we compute the DTW distance between phrase 1's segment #1 and phrase 2's segment #1, phrase 1 and phrase 3, and so on, for a total of `(10 * 9) / 2 = 45` unique distances.

- **Compute Statistics:** From these distances, we calculate the mean distance and the variance for each segment group. This gives us a statistical profile of what a "normal" segment looks like.

**Function Call:**
```python
evaluator.calculate_reference_stats()
```

---

## Evaluating a Generated Phrase

Once the benchmark is established, evaluating a new generated phrase (let's call it `G`) is straightforward.

1. **Segment the Generated Phrase:** `G` is segmented using the exact same CPD and Knee Locator approach.
2. **Compute Mean DTW to References:** For each segment of `G`, we compute its DTW distance to the corresponding segment of all the reference phrases. We then take the average of these distances to get a single mean distance (`md`).
3. **Check Against Benchmark:** For each segment, we check if its `md` falls within the allowed statistical range (e.g., within 2 standard deviations of the reference mean) calculated earlier.
4. **Final Verdict:**  
   - If the `md` for all segments of `G` falls within their allowed ranges, the phrase `G` is considered a good candidate.  
   - If even one segment falls outside the range, it is flagged as not a good candidate.

**Function Call:**
```python
is_good = evaluator.evaluate_curve_authenticity(generated_curve)
