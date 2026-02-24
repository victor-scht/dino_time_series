import matplotlib.pyplot as plt
from tslearn.datasets import UCR_UEA_datasets

print("hello")
# Load dataset
ucr = UCR_UEA_datasets()
X_train, y_train, X_test, y_test = ucr.load_dataset("ECG200")

# X shape: (n_samples, time_steps, 1)
print("Train shape:", X_train.shape)
print("Labels shape:", y_train.shape)

# Plot a few time series
plt.figure(figsize=(10, 5))
color_palette = ["blue", "red", "violet", "lightgreen"]

plt.title("Sample Time Series from UCR ECG200 Dataset")
plt.xlabel("Time")
plt.ylabel("Value")
for i in range(4):  # plot first 5 samples
    plt.plot(
        X_train[i].ravel(),
        label=f"class {y_train[i]}",
        c=color_palette[i],
        lw=2,
        alpha=0.7,
    )

plt.legend()
plt.grid(lw=1, alpha=0.3)
plt.savefig("ucr_ecg.png", dpi=300)

plt.show()
