import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean
from kneed import KneeLocator
from fastdtw import fastdtw
import ruptures as rpt
import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================================================
# VAE Model Definition
# =============================================================================
class VAE(nn.Module):
    """
    A Variational Autoencoder for 2D curve data.

    Args:
        input_dim (int): The flattened dimension of the input curve (e.g., num_points * 2).
        latent_dim (int): The dimension of the latent space.
    """
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_log_var = nn.Linear(64, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # Use Sigmoid as data is scaled to [0, 1]
        )

    def reparameterize(self, mu, log_var):
        """Performs the reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """Forward pass through the VAE."""
        h = self.encoder(x)
        mu, log_var = self.fc_mu(h), self.fc_log_var(h)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

# =============================================================================
# Curve Authenticity Evaluator
# =============================================================================
class CurveEvaluator:
    """
    A class to handle VAE training, curve generation, and evaluation.
    """
    def __init__(self, latent_dim=2, epochs=100, batch_size=32, min_len=500):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.min_len = min_len
        self.model = None
        self.scaler = None
        self.ref_curves = None
        self.ref_segments_by_index = None
        self.segment_stats = None
        self.n_segments = None

    def _load_and_preprocess_data(self, file_path):
        """Loads and preprocesses curve data from a text file."""
        print(f"Loading data from {file_path}...")
        raw_curves = []
        with open(file_path, 'r') as f:
            # Assuming a single curve in the file for this example
            # In a real scenario, you might have delimiters for multiple curves
            curve = np.loadtxt(f)
            raw_curves.append(curve)

        # Pad or truncate curves to a fixed length
        processed_curves = []
        for curve in raw_curves:
            if len(curve) > self.min_len:
                processed_curves.append(curve[:self.min_len])
            else:
                pad_width = self.min_len - len(curve)
                padded = np.pad(curve, ((0, pad_width), (0, 0)), 'edge')
                processed_curves.append(padded)
        self.ref_curves = np.array(processed_curves)

        # Scale and flatten the data for the VAE
        num_samples, num_points, num_dims = self.ref_curves.shape
        data_flat = self.ref_curves.reshape(num_samples, -1)
        self.scaler = MinMaxScaler()
        data_scaled = self.scaler.fit_transform(data_flat)
        return torch.tensor(data_scaled, dtype=torch.float32)

    def _vae_loss_function(self, recon_x, x, mu, log_var):
        """Calculates the VAE loss (Reconstruction + KL Divergence)."""
        BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, self.min_len * 2), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD

    def train_vae(self, data_tensor):
        """Initializes and trains the VAE model."""
        print("\n--- Training VAE ---")
        input_dim = data_tensor.shape[1]
        self.model = VAE(input_dim, self.latent_dim)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        train_loader = DataLoader(TensorDataset(data_tensor), batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            for batch_idx, (data,) in enumerate(train_loader):
                optimizer.zero_grad()
                recon_batch, mu, log_var = self.model(data)
                loss = self._vae_loss_function(recon_batch, data, mu, log_var)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()

            avg_loss = train_loss / len(train_loader.dataset)
            if (epoch + 1) % 10 == 0:
                print(f'Epoch: {epoch + 1}/{self.epochs}, Average Loss: {avg_loss:.4f}')
        print("--- VAE Training Complete ---")

    def _segment_curve_with_change_points(self, curve, max_bkps=10):
        """Segments a single curve using RBF kernel change point detection."""
        model = rpt.KernelCPD(kernel="rbf", min_size=5).fit(curve)
        costs = [model.cost.sum_of_costs(model.predict(n_bkps=i)) for i in range(1, max_bkps + 1)]

        # Use KneeLocator to find the optimal number of breakpoints
        kneedle = KneeLocator(range(1, max_bkps + 1), costs, curve="convex", direction="decreasing")
        # Fallback to a default if no knee is found
        optimal_bkps = kneedle.knee if kneedle.knee else 3

        breakpoints = model.predict(n_bkps=optimal_bkps)
        segments = []
        prev = 0
        for bp in breakpoints:
            segments.append(curve[prev:bp])
            prev = bp
        return segments

    def calculate_reference_stats(self, max_bkps=10):
        """Segments all reference curves and computes DTW statistics."""
        print("\n--- Calculating Statistics from Reference Curves ---")
        if self.ref_curves is None:
            raise ValueError("Reference curves not loaded. Run train_vae first.")

        # Use the first curve to establish a standard number of segments
        base_segments = self._segment_curve_with_change_points(self.ref_curves[0], max_bkps)
        self.n_segments = len(base_segments)
        print(f"Established baseline of {self.n_segments} segments per curve.")

        # Segment all reference curves
        ref_segments_list = []
        for curve in self.ref_curves:
            segments = self._segment_curve_with_change_points(curve, max_bkps)
            if len(segments) == self.n_segments:
                ref_segments_list.append(segments)
            else:
                print(f"Skipping a reference curve due to inconsistent segment count: {len(segments)} found.")

        # Transpose the list to group segments by their index
        self.ref_segments_by_index = [[] for _ in range(self.n_segments)]
        for segs in ref_segments_list:
            for i in range(self.n_segments):
                self.ref_segments_by_index[i].append(segs[i])

        # Compute DTW distances and statistics for each segment group
        self.segment_stats = []
        for i in range(self.n_segments):
            segment_group = self.ref_segments_by_index[i]
            dtw_distances = []
            # Compare each segment with every other in the group
            for j in range(len(segment_group)):
                for k in range(j + 1, len(segment_group)):
                    d, _ = fastdtw(segment_group[j], segment_group[k], dist=euclidean)
                    dtw_distances.append(d)

            if not dtw_distances: # Handle case with only one reference curve
                mean_dtw, var_dtw = 0.0, 0.0
            else:
                mean_dtw = np.mean(dtw_distances)
                var_dtw = np.var(dtw_distances)

            self.segment_stats.append((mean_dtw, var_dtw))
            print(f"Segment {i + 1}: Mean DTW = {mean_dtw:.2f}, Variance = {var_dtw:.2f}")
        print("--- Reference Statistics Calculated ---")

    def generate_curve_from_latent_space(self, z_vector):
        """Generates a curve from a given latent space vector."""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained or scaler not fitted.")

        with torch.no_grad():
            self.model.eval()
            z = torch.tensor(z_vector, dtype=torch.float32)
            generated_scaled = self.model.decoder(z).numpy()
            generated_inverse = self.scaler.inverse_transform(generated_scaled)
            return generated_inverse.reshape(self.min_len, 2)

    def evaluate_curve_authenticity(self, generated_curve, max_bkps=10, threshold=2.0):
        """
        Evaluates a generated curve against the reference statistics.

        Args:
            generated_curve (np.array): The 2D curve to evaluate.
            max_bkps (int): Max breakpoints to search for during segmentation.
            threshold (float): Std deviation multiplier for defining the allowed range.

        Returns:
            bool: True if the curve is "good", False otherwise.
        """
        print(f"\n--- Evaluating Generated Curve (Threshold: {threshold} std dev) ---")
        if self.segment_stats is None:
            raise ValueError("Reference statistics not calculated. Run calculate_reference_stats first.")

        # Segment the generated curve
        generated_segments = self._segment_curve_with_change_points(generated_curve, max_bkps)

        if len(generated_segments) != self.n_segments:
            print(f"❌ Evaluation Failed: Generated curve has {len(generated_segments)} segments, expected {self.n_segments}.")
            return False

        # Evaluate each segment
        is_good_candidate = True
        for i in range(self.n_segments):
            g_seg = generated_segments[i]
            ref_segs_group = self.ref_segments_by_index[i]

            # Calculate mean DTW distance from this segment to all reference segments in the group
            dists = [fastdtw(g_seg, ref_seg, dist=euclidean)[0] for ref_seg in ref_segs_group]
            mean_dist_to_ref = np.mean(dists)

            # Compare with reference stats
            mean_ref, var_ref = self.segment_stats[i]
            std_ref = np.sqrt(var_ref)
            lower_bound = mean_ref - threshold * std_ref
            upper_bound = mean_ref + threshold * std_ref

            print(f"Segment {i + 1}: Mean DTW to refs = {mean_dist_to_ref:.2f}. Allowed range = [{lower_bound:.2f}, {upper_bound:.2f}]")

            if not (lower_bound <= mean_dist_to_ref <= upper_bound):
                is_good_candidate = False

        # Final verdict
        if is_good_candidate:
            print("\n✅ Verdict: Generated curve is a GOOD candidate.")
        else:
            print("\n❌ Verdict: Generated curve is NOT a good candidate.")

        return is_good_candidate

def plot_curve(curve, title="Generated Curve", color='blue'):
    """Utility function to plot a 2D curve."""
    plt.figure(figsize=(10, 5))
    plt.plot(curve[:, 0], curve[:, 1], label=title, color=color)
    plt.title(title)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# =============================================================================
# Main Execution Block
# =============================================================================
if __name__ == '__main__':
    # --- Configuration ---
    DATA_FILE_PATH = "1R.txt" # Path to your data file
    LATENT_DIM = 2
    EPOCHS = 50 # Reduced for quick demonstration
    MIN_CURVE_LENGTH = 500

    # Check if data file exists
    if not os.path.exists(DATA_FILE_PATH):
        print(f"Error: Data file not found at '{DATA_FILE_PATH}'")
        print("Please create a file named '1R.txt' with your curve data.")
        # As a fallback, create a dummy sine wave file
        print("Creating a dummy sine wave file for demonstration purposes...")
        t = np.linspace(0, 22, MIN_CURVE_LENGTH)
        x = t
        y = np.sin(t)
        dummy_data = np.vstack((x, y)).T
        np.savetxt(DATA_FILE_PATH, dummy_data)


    # --- Step 1: Initialize and Train ---
    evaluator = CurveEvaluator(latent_dim=LATENT_DIM, epochs=EPOCHS, min_len=MIN_CURVE_LENGTH)
    curve_data_tensor = evaluator._load_and_preprocess_data(DATA_FILE_PATH)
    evaluator.train_vae(curve_data_tensor)

    # --- Step 2: Calculate Reference Statistics ---
    evaluator.calculate_reference_stats()

    # --- Step 3: Generate and Evaluate a "Good" Curve ---
    # Sample from the center of the latent space (mean=0, std=1)
    print("\n\n" + "="*50)
    print("Generating a 'GOOD' curve from the latent space center...")
    good_z_vector = [[0.1, -0.1]]
    good_generated_curve = evaluator.generate_curve_from_latent_space(good_z_vector)
    plot_curve(good_generated_curve, title="Good Generated Curve (from Latent Center)", color='green')
    evaluator.evaluate_curve_authenticity(good_generated_curve)
    print("="*50)


    # --- Step 4: Generate and Evaluate a "Bad" Curve ---
    # Use an outlier latent vector, far from the learned distribution
    print("\n\n" + "="*50)
    print("Generating a 'BAD' curve from an outlier latent vector...")
    bad_z_vector = [[-10.0, 20.0]]
    bad_generated_curve = evaluator.generate_curve_from_latent_space(bad_z_vector)
    plot_curve(bad_generated_curve, title="Bad Generated Curve (from Outlier Vector)", color='red')
    evaluator.evaluate_curve_authenticity(bad_generated_curve)
    print("="*50)
