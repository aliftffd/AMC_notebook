# def test_model(model, device='cuda'):
def test_model_with_improved_plots(model, device="cuda"):
    model.eval()
    Y_pred_ = []  # Predictions
    Y_true_ = []  # Ground truth
    Z_snr_ = []  # SNR values

    target_classes = test_dl.dataset.target_modulations
    target_snrs = test_dl.dataset.target_snrs
    modulation_classes = test_dl.dataset.modulation_classes
    target_modulations_indices = [
        modulation_classes.index(mod) for mod in target_classes
    ]

    # add debug
    print(f"target modulation:{target_classes}")
    print(f"target SNR: {target_snrs}")

    # Initialize accuracy stats DataFrame
    accuracy_stats = pd.DataFrame(
        0.0, index=target_classes, columns=target_snrs.astype("str")
    )

    # Get predictions
    with torch.no_grad():
        for x, y, z in test_dl:
            # Move tensors to specified device
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)

            # Get model predictions on device
            logits = model(x)
            y_pred = torch.argmax(logits, dim=-1)

            # Store results
            Y_pred_.append(y_pred.cpu())  # Move back to CPU for storage
            Y_true_.append(y.cpu())
            Z_snr_.append(z.cpu())

    # Convert to numpy for easier processing
    Y_pred = torch.cat(Y_pred_).numpy()
    Y_true = torch.cat(Y_true_).numpy()
    Z_snr = torch.cat(Z_snr_).numpy()

    # Calculate overall accuracy
    correct_preds = (Y_pred == Y_true).sum()
    total_samples = len(Y_true)
    total_accuracy = round(correct_preds * 100 / total_samples, 2)
    print(f"Accuracy on test dataset: {total_accuracy}%")

    # Count sampels for each modulation type
    mode_counts = {}
    for mod_idx, mod_name in enumerate(target_classes):
        count = np.sum(Y_true == mod_idx)
        mod_counts[mod_name] = count
        print(f"Modulation {mod_name}: {count} test samples")

    # Calculate accuracy per modulation and SNR
    for mod_idx, mod_name in enumerate(target_classes):
        for snr_idx, snr in enumerate(targer_snrs):
            snr_str = str(snr)

            mask = (Y_true == mod_idx) & (Z_snr == snr)
            total_samples = mask.sum()
        if total_samples > 0:
            correct_samples = ((Y_pred == Y_true) & mask).sum()

            accuracy = correct_sampels * 100 / total_samples
            accuracy_stats.loc[mod_name, snr_str] = round(accuracy, 2)
        else:
            accuracy_stats.loc[mod_name, snr_str] = np.nan
            pritn(f"warning: no sampel for {mod_name} at SNR = {snr}")

    return accuracy_stats

    # # Map indices back to original modulation classes if needed
    # for index, value in enumerate(target_modulations_indices):
    #     Y_pred[Y_pred == index] = value
    #     Y_true[Y_true == index] = value

    # # Calculate accuracy per modulation and SNR
    # for modu in target_modulations_indices:
    #     mod_class = modulation_classes[modu]
    #     for snr in target_snrs:
    #         snr_str = str(snr)

    #         # Find samples for this modulation and SNR
    #         mask = (Y_true == modu) & (Z_snr == snr)
    #         total_samples = mask.sum()

    #         if total_samples > 0:
    #             # Count correct predictions
    #             correct_samples = ((Y_pred == Y_true) & mask).sum()

    #             # Calculate and store accuracy percentage
    #             accuracy = (correct_samples * 100 / total_samples)
    #             accuracy_stats.loc[mod_class, snr_str] = round(accuracy, 2)

    # return accuracy_stats


def plot_improved_test_accuracy(model, device="cuda"):
    """
    Improved plotting function that shows all modulations properly
    """
    accuracy_df = test_model_with_improved_plots(model, device)

    plt.figure(figsize=(14, 8))

    accuracy_long = accuracy_df.reset_index().mel(
        id_vars=["index"], var_name="SNR", value_name="Accuracy"
    )
    accuracy_long.columns = ["Modulation", "SNR", "Accuracy"]

    sns.lineplot(
        data=accuracy_long,
        x="SNR",
        y="Accuracy",
        hue="Modulation",
        marker="o",
        markersize=8,
        linewidth=2,
    )

    # Specifically highlights PSK modulations
    psk_mods = [mod for mod in accuracy_df.index if "PSK" in mod]
    if psk_mods:
        print(f"Highlighting PSK modulation : {psk_mods}")
        psk_df = accuracy_long[accuracy_long["Modulation"].isin(psk_mods)]

        for mod in psk_mods:
            mod_data = psk_df[psk_df["Modulation"] == mod]
            plt.plot(
                mod_data["SNR"],
                mod_data["Accuracy"],
                linewidth=3.5,
                linestyle="--",
                marker="*",
                markersize=12,
            )

    plt.title(
        "Classification Accuracy vs SNR for Different Modulation Types", fontsize=16
    )
    plt.xlabel("Signal-to-Noise Ratio (dB)", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("all_modulations_accuracy.png", dpi=300)
    plt.show()

    n_rows = (len(accuracy_df.index) + 2) // 3

    fig, axes = plt.subplots(n_rows, 3, figsize=(18, n_rows * 4), sharey=True)
    axes = axes.flatten()

    for i in range(len(accuracy_df.index), len(axes)):
        axes[i].set_visible(False)

    for i, mod in enumerate(accuracy_df.index):
        ax = axes[i]

        mod_data = accuracy_df.loc[mod].astype[float]

        ax.bar(
            mod_data.index,
            mode_data.values,
            color="skyblue" if "PSK" not in mod else "red",
        )

        ax.plot(mod_data.index, mod_data.values, "k--", linewidth=2)

        ax.set_title(f"{mod}", fontsize=14)
        ax.set_xlabel("SNR (dB)" if i >= len(accuracy_df.index) - 3 else "")
        ax.set_ylabel("Accuracy (%)" if i % 3 == 0 else "")
        ax.set_ylim(0, 105)  # Set y-axis from 0 to 100% with a bit of margin
        ax.grid(True, alpha=0.3)

        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45)

    plt.suptitle("Classification Accuracy by Modulation Type", fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Make room for suptitle
    plt.savefig("modulation_accuracy_subplots.png", dpi=300)
    plt.show()

    # 3. Also create a heatmap visualization
    plt.figure(figsize=(14, 8))
    sns.heatmap(
        accuracy_df.astype(float),
        annot=True,
        cmap="viridis",
        fmt=".1f",
        cbar_kws={"label": "Accuracy (%)"},
    )
    plt.title("Classification Accuracy Heatmap by Modulation and SNR", fontsize=16)
    plt.xlabel("Signal-to-Noise Ratio (dB)", fontsize=14)
    plt.ylabel("Modulation Type", fontsize=14)
    plt.tight_layout()
    plt.savefig("modulation_accuracy_heatmap.png", dpi=300)
    plt.show()

    return accuracy_df


def plot_training_history(model_name: str, history: list):
    plt.figure(figsize=(10, 6))
    plt.title(f"Training of {model_name} model on radioml2018")
    plt.xlabel("Epochs")
    plt.plot(history[0], label="train_loss")
    plt.plot(history[1], label="train_accuracy")
    plt.plot(history[2], label="valid_loss")
    plt.plot(history[3], label="valid_accuracy")
    plt.legend(loc="upper left")
    plt.show()


# def plot_test_accuracy(model, device='cuda'):
#     accuracy_df = test_model(model, device)

#     fig, axes = plt.subplots(len(test_dl.dataset.target_modulations), 1, figsize=(12, 8), sharex=True, sharey=True)
#     fig.subplots_adjust(hspace=0.4)
#     fig.supylabel('Accuracy (%)')
#     fig.supxlabel('Signal to noise ratios (dB)')

#     # Handle the case where there's only one modulation
#     if len(test_dl.dataset.target_modulations) == 1:
#         axes = [axes]

#     for index, ax in enumerate(axes):
#         ax.set_title(accuracy_df.index[index])
#         ax.bar(accuracy_df.iloc[index].index, accuracy_df.iloc[index].values)
#         ax.set_ylim(0, 100)  # Set y-axis from 0 to 100%

#     plt.tight_layout()
#     plt.show()


#     return accuracy_df  # Return the DataFrame for potential further analysis
def check_dataset_distribution(test_dl):
    """
    Analyze the distribution of modulations in the dataset
    """
    # Get dataset
    dataset = test_dl.dataset

    # Analyze distribution
    mod_counts = {}
    snr_mod_counts = {}

    # Initialize counts for all modulations
    for mod in dataset.target_modulations:
        mod_counts[mod] = 0

    # Count occurrences of each modulation
    for i in range(len(dataset)):
        _, mod_idx, snr = dataset[i]
        mod = dataset.target_modulations[mod_idx]

        # Count by modulation
        mod_counts[mod] += 1

        # Count by modulation and SNR
        if snr not in snr_mod_counts:
            snr_mod_counts[snr] = {}
        if mod not in snr_mod_counts[snr]:
            snr_mod_counts[snr][mod] = 0
        snr_mod_counts[snr][mod] += 1

    print("Modulation distribution in dataset:")
    for mod, count in mod_counts.items():
        print(f"  {mod}: {count} samples")

    # Special check for PSK modulations
    psk_mods = [mod for mod in dataset.target_modulations if "PSK" in mod]
    print("\nPSK modulation distribution:")
    for mod in psk_mods:
        print(f"\n{mod} distribution across SNRs:")
        for snr in sorted(snr_mod_counts.keys()):
            count = snr_mod_counts[snr].get(mod, 0)
            print(f"  SNR {snr}dB: {count} samples")

    return mod_counts, snr_mod_counts


# def train_test_plots(model, model_name, verbose=False, device='cuda', num_epoch=30):
#     model, train_history = train_model(model, verbose=verbose, device=device, num_epoch=num_epoch)
#     torch.save(model, f'{model_name}.pth')
#     plot_training_history(model_name, train_history)
#     accuracy_results = plot_test_accuracy(model, device)
#     del model
#     return train_history, accuracy_results


def improved_train_test_plots(
    model, model_name, verbose=False, device="cuda", num_epoch=30
):
    # First check the dataset distribution
    print("Analyzing test dataset distribution...")
    mod_counts, snr_mod_counts = check_dataset_distribution(test_dl)

    # Train the model
    print(f"\nTraining {model_name}...")
    model, train_history = train_model(
        model, verbose=verbose, device=device, num_epoch=num_epoch
    )
    torch.save(model, f"{model_name}.pth")

    # Plot training history
    plot_training_history(model, train_history)

    # Plot test accuracy with improved visualization
    print("\nEvaluating and plotting test accuracy...")
    accuracy_results = plot_improved_test_accuracy(model, device)

    return model, train_history, accuracy_results
