import matplotlib.pyplot as plt
import numpy as np

# Data from best model evaluation
tp = 6749830
tn = 2905365
fp = 6251050
fn = 1003067

total = tp + tn + fp + fn
accuracy = (tp + tn) / total
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1_score]

plt.figure(figsize=(10, 6))
bars = plt.bar(metrics, values, color=['#4285F4', '#EA4335', '#FBBC05', '#34A853'])

# Add labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.ylim(0, 1.1)
plt.ylabel('Score', fontsize=12)
plt.title('Performance Metrics for Best Trained Model\n(Epoch 2, IoU: 0.4716)', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('performance_metrics.png', dpi=300)
print("Performance metrics plot saved to performance_metrics.png")
print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}")
