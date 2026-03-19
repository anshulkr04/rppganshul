import matplotlib.pyplot as plt

# ------------------------------------------------------------
# VALIDATION LOSS DATA (reconstructed from your logs)
# ------------------------------------------------------------

val_losses = {
    "Initial PhysMamba (bad)": [
        0.99, 0.99, 0.99, 0.99, 1.00, 0.98, 0.96, 0.98, 0.89, 0.84,
        0.81, 0.84, 0.89, 0.98, 0.78, 0.53, 0.42, 0.38, 0.33, 0.34
    ],

    "Broken Temporal-only": [
        1.00, 1.00, 1.00, 1.00, 1.00, 1.00
    ],

    "Intermediate (improving)": [
        0.98, 0.44, 0.88, 0.50, 0.35, 0.45, 0.71, 0.30, 0.18, 0.28,
        0.17, 0.19, 0.22, 0.17, 0.20, 0.18, 0.17, 0.19, 0.17
    ],

    "Final Balanced Mamba (BEST)": [
        0.98, 0.62, 0.90, 0.83, 0.18, 0.30, 0.65, 0.33, 0.23, 0.23,
        0.17, 0.15, 0.17, 0.15, 0.14, 0.15, 0.13, 0.14, 0.13, 0.13
    ]
}

# ------------------------------------------------------------
# PLOT 1: VALIDATION LOSS
# ------------------------------------------------------------

plt.figure(figsize=(10,6))

for name, losses in val_losses.items():
    epochs = list(range(len(losses)))
    plt.plot(epochs, losses, label=name)

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss Across Model Iterations")
plt.legend()
plt.grid()

plt.savefig("validation_loss_evolution.png")
plt.show()

# ------------------------------------------------------------
# FINAL METRICS
# ------------------------------------------------------------

models = ["Initial", "Intermediate", "Final"]

mae = [36, 12, 5.7]
pearson = [0.0, 0.66, 0.875]
snr = [-5.9, 0.3, 0.6]

x = range(len(models))

plt.figure(figsize=(10,6))

plt.plot(x, mae, marker='o', label="MAE")
plt.plot(x, pearson, marker='o', label="Pearson")
plt.plot(x, snr, marker='o', label="SNR")

plt.xticks(x, models)
plt.title("Performance Improvement Across Iterations")
plt.legend()
plt.grid()

plt.savefig("performance_evolution.png")
plt.show()