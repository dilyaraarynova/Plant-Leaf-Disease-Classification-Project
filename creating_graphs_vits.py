"""
Visualization Script
Input data(requirement):
- experiments/comprehensive_comparison.json
- experiments/[experiment_folders]/history.json
- experiments/[experiment_folders]/test_results.json

Generates:
1. Model comparison bar charts
2. Augmentation effectiveness analysis
3. Training curves (loss & accuracy)
4. Per-class performance heatmaps
5. Confusion matrices
6. Domain generalization improvement analysis
7. Strategy comparison charts
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List
import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

OUTPUT_DIR = Path('./presentation_graphs')
OUTPUT_DIR.mkdir(exist_ok=True)

#input data directory
DATA_DIR = Path('./experiments')


def load_comprehensive_results():
    """Load the comprehensive comparison results"""
    results_file = DATA_DIR / 'comprehensive_comparison.json'
    with open(results_file, 'r') as f:
        return json.load(f)


def load_experiment_history(exp_dir):
    """Load training history for a specific experiment"""
    history_file = Path(exp_dir) / 'history.json'
    if history_file.exists():
        with open(history_file, 'r') as f:
            return json.load(f)
    return None


def load_test_results(exp_dir):
    """Load test results for a specific experiment"""
    results_file = Path(exp_dir) / 'test_results.json'
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return None


# GRAPH 1: MODEL COMPARISON - OVERALL PERFORMANCE
def plot_model_comparison_overall(results):
    """
    Compare dinov2_vits14 vs vit_base across all augmentations
    Shows average and best performance
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    models = ['dinov2_vits14', 'vit_base']
    strategies = ['feature_extraction', 'partial_finetuning']

    # Prepare data
    data = []
    for model in models:
        for strategy in strategies:
            filtered = [r for r in results if r['model'] == model and r['strategy'] == strategy]
            if filtered:
                avg_acc = np.mean([r['test_accuracy'] for r in filtered]) * 100
                best_acc = np.max([r['test_accuracy'] for r in filtered]) * 100
                avg_f1 = np.mean([r['macro_f1'] for r in filtered])
                best_f1 = np.max([r['macro_f1'] for r in filtered])

                data.append({
                    'model': model,
                    'strategy': strategy,
                    'avg_acc': avg_acc,
                    'best_acc': best_acc,
                    'avg_f1': avg_f1,
                    'best_f1': best_f1
                })

    df = pd.DataFrame(data)

    # Plot 1: Accuracy
    x = np.arange(len(strategies))
    width = 0.35

    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        axes[0].bar(x + i * width, model_data['avg_acc'], width,
                    label=f'{model} (avg)', alpha=0.7)
        axes[0].bar(x + i * width, model_data['best_acc'] - model_data['avg_acc'],
                    width, bottom=model_data['avg_acc'],
                    label=f'{model} (best)', alpha=0.4, hatch='//')

    axes[0].set_xlabel('Training Strategy')
    axes[0].set_ylabel('Test Accuracy (%)')
    axes[0].set_title('Model Comparison: Test Accuracy')
    axes[0].set_xticks(x + width / 2)
    axes[0].set_xticklabels([s.replace('_', ' ').title() for s in strategies])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: F1 Score
    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        axes[1].bar(x + i * width, model_data['avg_f1'], width,
                    label=f'{model} (avg)', alpha=0.7)
        axes[1].bar(x + i * width, model_data['best_f1'] - model_data['avg_f1'],
                    width, bottom=model_data['avg_f1'],
                    label=f'{model} (best)', alpha=0.4, hatch='//')

    axes[1].set_xlabel('Training Strategy')
    axes[1].set_ylabel('Macro F1 Score')
    axes[1].set_title('Model Comparison: Macro F1 Score')
    axes[1].set_xticks(x + width / 2)
    axes[1].set_xticklabels([s.replace('_', ' ').title() for s in strategies])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_model_comparison_overall.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: 01_model_comparison_overall.png")


# GRAPH 2: AUGMENTATION EFFECTIVENESS RANKING
def plot_augmentation_effectiveness(results):
    """
    Rank all 7 augmentation strategies by their effectiveness
    Shows average performance across all models and strategies
    """
    # Get unique augmentations
    augmentations = []
    aug_seen = set()
    for r in results:
        if r['augmentation'] not in aug_seen:
            augmentations.append((r['augmentation'], r['description']))
            aug_seen.add(r['augmentation'])

    # Calculate average performance
    aug_performance = []
    for aug_name, aug_desc in augmentations:
        filtered = [r for r in results if r['augmentation'] == aug_name]
        avg_acc = np.mean([r['test_accuracy'] for r in filtered]) * 100
        avg_f1 = np.mean([r['macro_f1'] for r in filtered])
        std_acc = np.std([r['test_accuracy'] for r in filtered]) * 100
        std_f1 = np.std([r['macro_f1'] for r in filtered])

        aug_performance.append({
            'name': aug_name,
            'description': aug_desc,
            'avg_acc': avg_acc,
            'avg_f1': avg_f1,
            'std_acc': std_acc,
            'std_f1': std_f1
        })

    # Sort by F1 score
    aug_performance.sort(key=lambda x: x['avg_f1'], reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    names = [a['name'] for a in aug_performance]
    acc_vals = [a['avg_acc'] for a in aug_performance]
    acc_errs = [a['std_acc'] for a in aug_performance]
    f1_vals = [a['avg_f1'] for a in aug_performance]
    f1_errs = [a['std_f1'] for a in aug_performance]

    y_pos = np.arange(len(names))

    # Plot 1: Accuracy
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
    axes[0].barh(y_pos, acc_vals, xerr=acc_errs, color=colors, alpha=0.8)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(names)
    axes[0].set_xlabel('Average Test Accuracy (%) ± Std Dev')
    axes[0].set_title('Augmentation Strategy Ranking by Accuracy')
    axes[0].grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (val, err) in enumerate(zip(acc_vals, acc_errs)):
        axes[0].text(val + err + 0.5, i, f'{val:.2f}%', va='center')

    # Plot 2: F1 Score
    axes[1].barh(y_pos, f1_vals, xerr=f1_errs, color=colors, alpha=0.8)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(names)
    axes[1].set_xlabel('Average Macro F1 Score ± Std Dev')
    axes[1].set_title('Augmentation Strategy Ranking by F1 Score')
    axes[1].grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (val, err) in enumerate(zip(f1_vals, f1_errs)):
        axes[1].text(val + err + 0.01, i, f'{val:.4f}', va='center')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_augmentation_effectiveness.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: 02_augmentation_effectiveness.png")


# GRAPH 3: DOMAIN GENERALIZATION IMPROVEMENT
def plot_domain_gen_improvement(results):
    """
    Show improvement over baseline for each model-strategy combination
    Demonstrates the effectiveness of domain generalization techniques
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    models = ['dinov2_vits14', 'vit_base']
    strategies = ['feature_extraction', 'partial_finetuning']

    for i, model in enumerate(models):
        for j, strategy in enumerate(strategies):
            ax = axes[i, j]

            # Filter results
            filtered = [r for r in results if r['model'] == model and r['strategy'] == strategy]

            # Get baseline
            baseline = next((r for r in filtered if 'baseline' in r['augmentation']), None)
            if not baseline:
                continue

            baseline_acc = baseline['test_accuracy'] * 100

            # Calculate improvements
            improvements = []
            for r in filtered:
                if 'baseline' not in r['augmentation']:
                    improvement = (r['test_accuracy'] * 100) - baseline_acc
                    improvements.append({
                        'name': r['augmentation'],
                        'improvement': improvement,
                        'absolute_acc': r['test_accuracy'] * 100
                    })

            # Sort by improvement
            improvements.sort(key=lambda x: x['improvement'], reverse=True)

            names = [imp['name'] for imp in improvements]
            vals = [imp['improvement'] for imp in improvements]
            colors = ['green' if v > 0 else 'red' for v in vals]

            y_pos = np.arange(len(names))
            ax.barh(y_pos, vals, color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=10)
            ax.set_xlabel('Improvement over Baseline (%)')
            ax.set_title(f'{model} - {strategy.replace("_", " ").title()}\n(Baseline: {baseline_acc:.2f}%)')
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
            ax.grid(True, alpha=0.3, axis='x')

            # Add value labels
            for k, (val, imp) in enumerate(zip(vals, improvements)):
                ax.text(val + 0.1 if val > 0 else val - 0.1, k,
                        f'{val:+.2f}% ({imp["absolute_acc"]:.2f}%)',
                        va='center', ha='left' if val > 0 else 'right', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_domain_gen_improvement.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: 03_domain_gen_improvement.png")


# GRAPH 4: TRAINING CURVES
def plot_training_curves(results):
    """
    Plot training and validation curves for best performing configurations
    Shows learning dynamics
    """
    # Find best configuration for each model-strategy pair
    models = ['dinov2_vits14', 'vit_base']
    strategies = ['feature_extraction', 'partial_finetuning']

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    for i, model in enumerate(models):
        for j, strategy in enumerate(strategies):
            ax = axes[i, j]

            # Filter results
            filtered = [r for r in results if r['model'] == model and r['strategy'] == strategy]

            # Get best (by F1) and baseline
            best = max(filtered, key=lambda x: x['macro_f1'])
            baseline = next((r for r in filtered if 'baseline' in r['augmentation']), None)

            # Load histories
            best_history = load_experiment_history(best['experiment_dir'])
            baseline_history = load_experiment_history(baseline['experiment_dir']) if baseline else None

            if best_history:
                epochs_best = range(1, len(best_history['train_loss']) + 1)

                # Plot best configuration
                ax.plot(epochs_best, best_history['train_loss'], 'b-', label='Train Loss (Best)', linewidth=2)
                ax.plot(epochs_best, best_history['val_loss'], 'b--', label='Val Loss (Best)', linewidth=2)

                # Plot baseline for comparison
                if baseline_history:
                    epochs_baseline = range(1, len(baseline_history['train_loss']) + 1)
                    ax.plot(epochs_baseline, baseline_history['train_loss'], 'r-', alpha=0.5,
                            label='Train Loss (Baseline)', linewidth=1.5)
                    ax.plot(epochs_baseline, baseline_history['val_loss'], 'r--', alpha=0.5,
                            label='Val Loss (Baseline)', linewidth=1.5)

            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'{model} - {strategy.replace("_", " ").title()}\nBest: {best["augmentation"]}')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_training_curves_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: 04_training_curves_loss.png")

    # Plot accuracy curves
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    for i, model in enumerate(models):
        for j, strategy in enumerate(strategies):
            ax = axes[i, j]

            # Filter results
            filtered = [r for r in results if r['model'] == model and r['strategy'] == strategy]

            # Get best and baseline
            best = max(filtered, key=lambda x: x['macro_f1'])
            baseline = next((r for r in filtered if 'baseline' in r['augmentation']), None)

            # Load histories
            best_history = load_experiment_history(best['experiment_dir'])
            baseline_history = load_experiment_history(baseline['experiment_dir']) if baseline else None

            if best_history:
                epochs_best = range(1, len(best_history['train_acc']) + 1)

                # Convert to percentage
                train_acc_best = [acc * 100 for acc in best_history['train_acc']]
                val_acc_best = [acc * 100 for acc in best_history['val_acc']]

                ax.plot(epochs_best, train_acc_best, 'b-', label='Train Acc (Best)', linewidth=2)
                ax.plot(epochs_best, val_acc_best, 'b--', label='Val Acc (Best)', linewidth=2)

                # Plot baseline
                if baseline_history:
                    epochs_baseline = range(1, len(baseline_history['train_acc']) + 1)
                    train_acc_base = [acc * 100 for acc in baseline_history['train_acc']]
                    val_acc_base = [acc * 100 for acc in baseline_history['val_acc']]

                    ax.plot(epochs_baseline, train_acc_base, 'r-', alpha=0.5,
                            label='Train Acc (Baseline)', linewidth=1.5)
                    ax.plot(epochs_baseline, val_acc_base, 'r--', alpha=0.5,
                            label='Val Acc (Baseline)', linewidth=1.5)

            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title(f'{model} - {strategy.replace("_", " ").title()}\nBest: {best["augmentation"]}')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_training_curves_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: 05_training_curves_accuracy.png")


# GRAPH 5: STRATEGY COMPARISON
def plot_strategy_comparison(results):
    """
    Compare feature extraction vs partial finetuning
    Shows which strategy benefits more from augmentation
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    models = ['dinov2_vits14', 'vit_base']
    augmentations = list(set(r['augmentation'] for r in results))

    for i, model in enumerate(models):
        ax = axes[i]

        strategies = ['feature_extraction', 'partial_finetuning']

        # Prepare data
        strategy_data = {s: [] for s in strategies}

        for aug in augmentations:
            for strategy in strategies:
                result = next((r for r in results
                               if r['model'] == model
                               and r['strategy'] == strategy
                               and r['augmentation'] == aug), None)
                if result:
                    strategy_data[strategy].append(result['test_accuracy'] * 100)

        # Create box plot
        data_to_plot = [strategy_data[s] for s in strategies]
        bp = ax.boxplot(data_to_plot, labels=[s.replace('_', ' ').title() for s in strategies],
                        patch_artist=True, showmeans=True)

        # Color boxes
        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title(f'{model}\nStrategy Performance across All Augmentations')
        ax.grid(True, alpha=0.3, axis='y')

        # Add statistical annotation
        fe_mean = np.mean(strategy_data['feature_extraction'])
        pf_mean = np.mean(strategy_data['partial_finetuning'])
        diff = pf_mean - fe_mean

        ax.text(0.5, 0.95, f'Partial FT advantage: {diff:+.2f}%',
                transform=ax.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '06_strategy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: 06_strategy_comparison.png")


# GRAPH 6: PER-CLASS PERFORMANCE HEATMAP
def plot_per_class_heatmap(results):
    """
    Heatmap showing per-class F1 scores for different configurations
    Useful for identifying which classes benefit from augmentation
    """
    # Get best configuration for each model
    models = ['dinov2_vits14', 'vit_base']

    for model in models:
        # Get baseline and best for this model
        model_results = [r for r in results if r['model'] == model]
        baseline = next((r for r in model_results if 'baseline' in r['augmentation']), None)
        best = max(model_results, key=lambda x: x['macro_f1'])

        if not baseline or not best:
            continue

        # Load test results
        baseline_test = load_test_results(baseline['experiment_dir'])
        best_test = load_test_results(best['experiment_dir'])

        if not baseline_test or not best_test:
            continue

        # Extract per-class F1 scores
        classes = list(baseline_test['per_class_metrics'].keys())
        baseline_f1 = [baseline_test['per_class_metrics'][c]['f1-score'] for c in classes]
        best_f1 = [best_test['per_class_metrics'][c]['f1-score'] for c in classes]

        # Create comparison
        data = np.array([baseline_f1, best_f1])

        fig, ax = plt.subplots(figsize=(14, 8))

        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        # Set ticks
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks([0, 1])
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_yticklabels(['Baseline', f'Best ({best["augmentation"]})'])

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('F1 Score', rotation=270, labelpad=20)

        # Add text annotations
        for i in range(2):
            for j in range(len(classes)):
                text = ax.text(j, i, f'{data[i, j]:.3f}',
                               ha="center", va="center", color="black", fontsize=9)

        ax.set_title(f'{model}: Per-Class F1 Score Comparison\nBaseline vs Best Augmentation')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'07_per_class_heatmap_{model}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Created: 07_per_class_heatmap_{model}.png")


# GRAPH 7: AUGMENTATION TECHNIQUE BREAKDOWN
def plot_augmentation_breakdown(results):
    """
    Shows the incremental benefit of adding each augmentation technique
    Compares: baseline → aggressive → +cutmix → +label smoothing → +strong reg
    """
    models = ['dinov2_vits14', 'vit_base']

    # Define the progression of techniques
    progression = [
        'baseline',
        'aggressive',
        'aggressive_cutmix',
        'aggressive_cutmix_ls',
        'aggressive_full'
    ]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for i, model in enumerate(models):
        ax = axes[i]

        # Get results for this model (average across strategies)
        model_results = [r for r in results if r['model'] == model]

        accuracies = []
        f1_scores = []
        labels = []

        for aug in progression:
            aug_results = [r for r in model_results if r['augmentation'] == aug]
            if aug_results:
                avg_acc = np.mean([r['test_accuracy'] for r in aug_results]) * 100
                avg_f1 = np.mean([r['macro_f1'] for r in aug_results])
                accuracies.append(avg_acc)
                f1_scores.append(avg_f1)

                # Create label
                if aug == 'baseline':
                    labels.append('Baseline')
                elif aug == 'aggressive':
                    labels.append('+ Aggressive\nAug')
                elif aug == 'aggressive_cutmix':
                    labels.append('+ CutMix')
                elif aug == 'aggressive_cutmix_ls':
                    labels.append('+ Label\nSmoothing')
                elif aug == 'aggressive_full':
                    labels.append('+ Strong\nReg')

        x = np.arange(len(labels))
        width = 0.35

        ax.bar(x - width / 2, accuracies, width, label='Test Accuracy (%)', alpha=0.8)
        ax2 = ax.twinx()
        ax2.bar(x + width / 2, [f * 100 for f in f1_scores], width,
                label='Macro F1 (×100)', alpha=0.8, color='orange')

        ax.set_xlabel('Augmentation Progression')
        ax.set_ylabel('Test Accuracy (%)', color='C0')
        ax2.set_ylabel('Macro F1 (×100)', color='orange')
        ax.set_title(f'{model}\nIncremental Augmentation Benefits')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.tick_params(axis='y', labelcolor='C0')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax.grid(True, alpha=0.3)

        # Add improvement annotations
        for j in range(1, len(accuracies)):
            improvement = accuracies[j] - accuracies[j - 1]
            ax.annotate(f'+{improvement:.2f}%',
                        xy=(j, accuracies[j]),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '08_augmentation_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: 08_augmentation_breakdown.png")


# GRAPH 8: CONFUSION MATRIX FOR BEST MODEL
def plot_best_confusion_matrix(results):
    """
    Plot confusion matrix for the overall best performing model
    """
    # Find overall best model
    best = max(results, key=lambda x: x['macro_f1'])

    # Load test results
    test_results = load_test_results(best['experiment_dir'])

    if not test_results or 'confusion_matrix' not in test_results:
        return

    cm = np.array(test_results['confusion_matrix'])
    classes = list(test_results['per_class_metrics'].keys())

    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Plot 1: Raw counts
    im1 = axes[0].imshow(cm, interpolation='nearest', cmap='Blues')
    axes[0].set_title(f'Confusion Matrix (Counts)\n{best["model"]} - {best["strategy"]} - {best["augmentation"]}')

    # Plot 2: Normalized
    im2 = axes[1].imshow(cm_normalized, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    axes[1].set_title(
        f'Confusion Matrix (Normalized)\nTest Accuracy: {best["test_accuracy"] * 100:.2f}%, F1: {best["macro_f1"]:.4f}')

    for ax, cm_data in zip(axes, [cm, cm_normalized]):
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_yticklabels(classes)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

        # Add text annotations
        fmt = 'd' if ax == axes[0] else '.2f'
        thresh = cm_data.max() / 2.
        for i in range(cm_data.shape[0]):
            for j in range(cm_data.shape[1]):
                ax.text(j, i, format(cm_data[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm_data[i, j] > thresh else "black",
                        fontsize=8)

    plt.colorbar(im1, ax=axes[0])
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '09_best_model_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: 09_best_model_confusion_matrix.png")


# GRAPH 9: SUMMARY COMPARISON TABLE
def create_summary_table(results):
    """
    Create a beautiful summary table as an image
    """
    # Find best configurations
    models = ['dinov2_vits14', 'vit_base']
    strategies = ['feature_extraction', 'partial_finetuning']

    table_data = []
    for model in models:
        for strategy in strategies:
            filtered = [r for r in results if r['model'] == model and r['strategy'] == strategy]
            if not filtered:
                continue

            baseline = next((r for r in filtered if 'baseline' in r['augmentation']), None)
            best = max(filtered, key=lambda x: x['macro_f1'])

            if baseline:
                improvement = (best['test_accuracy'] - baseline['test_accuracy']) * 100
            else:
                improvement = 0

            table_data.append([
                model,
                strategy.replace('_', ' ').title(),
                f"{baseline['test_accuracy'] * 100:.2f}%" if baseline else "N/A",
                best['augmentation'],
                f"{best['test_accuracy'] * 100:.2f}%",
                f"{best['macro_f1']:.4f}",
                f"+{improvement:.2f}%"
            ])

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('tight')
    ax.axis('off')

    columns = ['Model', 'Strategy', 'Baseline Acc', 'Best Augmentation',
               'Best Acc', 'Best F1', 'Improvement']

    table = ax.table(cellText=table_data, colLabels=columns,
                     cellLoc='center', loc='center',
                     colWidths=[0.15, 0.15, 0.12, 0.18, 0.12, 0.12, 0.12])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style rows
    for i in range(1, len(table_data) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    plt.title('Summary: Best Configurations for Each Model-Strategy Combination',
              fontsize=16, fontweight='bold', pad=20)

    plt.savefig(OUTPUT_DIR / '10_summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: 10_summary_table.png")


def main():
    """Generate all presentation graphs"""
    print("\n" + "=" * 80)
    print("GENERATING PRESENTATION GRAPHS")
    print("=" * 80 + "\n")

    # Load results
    print("Loading comprehensive results...")
    results = load_comprehensive_results()
    print(f"Loaded {len(results)} experiment results\n")

    print("Creating visualizations...\n")

    # Generate all graphs
    plot_model_comparison_overall(results)
    plot_augmentation_effectiveness(results)
    plot_domain_gen_improvement(results)
    plot_training_curves(results)
    plot_strategy_comparison(results)
    plot_per_class_heatmap(results)
    plot_augmentation_breakdown(results)
    plot_best_confusion_matrix(results)
    create_summary_table(results)

    print("\n" + "=" * 80)
    print(f"ALL GRAPHS CREATED SUCCESSFULLY!")
    print(f"Saved to: {OUTPUT_DIR.absolute()}")
    print("=" * 80 + "\n")

    print("Graph descriptions:")
    print("1. 01_model_comparison_overall.png - Overall model performance comparison")
    print("2. 02_augmentation_effectiveness.png - Ranking of augmentation strategies")
    print("3. 03_domain_gen_improvement.png - Improvement over baseline per model")
    print("4. 04_training_curves_loss.png - Training loss curves")
    print("5. 05_training_curves_accuracy.png - Training accuracy curves")
    print("6. 06_strategy_comparison.png - Feature extraction vs partial finetuning")
    print("7. 07_per_class_heatmap_*.png - Per-class performance comparison")
    print("8. 08_augmentation_breakdown.png - Incremental augmentation benefits")
    print("9. 09_best_model_confusion_matrix.png - Confusion matrix of best model")
    print("10. 10_summary_table.png - Summary comparison table")


if __name__ == "__main__":
    main()