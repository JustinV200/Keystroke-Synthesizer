import torch
def compute_empirical_variance(train_dataset, device, max_samples=2000):
    # Compute empirical variance of continuous features from a subset of the training data
    all_cont_features = []
    for idx in range(min(len(train_dataset), 2000)):  # Sample 2000
        sample = train_dataset[idx]
        all_cont_features.append(sample["target"][:, [0, 1, 2]])  # DwellTime, FlightTime, typing_speed
    all_cont = torch.cat(all_cont_features, dim=0)
    # Use NaN-aware variance computation (same approach as dataLoader)
    empirical_var = torch.tensor([all_cont[:, i][~torch.isnan(all_cont[:, i])].var(unbiased=True) for i in range(all_cont.shape[1])])
    # Ensure empirical variance is never too small to prevent numerical issues
    empirical_var = torch.clamp(empirical_var, min=1e-4)
    print(f"Empirical variance (standardized space): {empirical_var.tolist()}")
    empirical_var = empirical_var.to(device)
    return empirical_var


def checkforNans(mean, logvar, logits, i, input_ids=None, attention_m=None, targets=None):
    # Debug: Check for NaN in inputs
    if torch.isnan(input_ids).any():
        print(f"  WARNING: NaN in input_ids for batch {i}")
    if torch.isnan(attention_m).any():
        print(f"  WARNING: NaN in attention_mask for batch {i}")
    for j, target in enumerate(targets):
        if torch.isnan(target).any():
            print(f"  WARNING: NaN in target {j} for batch {i}")
            print(f"    Target shape: {target.shape}")
            print(f"    NaN locations: {torch.isnan(target).sum(dim=0)}")
        # Debug: Check for NaN in model outputs
    if torch.isnan(mean).any():
        print(f"  WARNING: NaN in mean output for batch {i}")
    if torch.isnan(logvar).any():
        print(f"  WARNING: NaN in logvar output for batch {i}")
        print(f"    logvar range: min={logvar.min():.3f}, max={logvar.max():.3f}")
    if torch.isnan(logits).any():
        print(f"  WARNING: NaN in logits output for batch {i}")
