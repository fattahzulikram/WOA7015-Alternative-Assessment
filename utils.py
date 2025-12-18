import os
import numpy as np
from matplotlib import pyplot as plt

def plot_training_curves(history, results_dir, model_name='baseline'):
        """
        Plots training curves for loss, accuracy, learning rate and validation accuracy with the best validation marked
        
        :param history: MUST have the properties: train_loss, val_loss, train_acc, val_acc, learning_rates
        :param results_dir: The path where the plots will be saved
        :param model_name: The baseline or the challenger
        """

        _, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 0].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Validation accuracy with best marker
        axes[1, 1].plot(epochs, history['val_acc'], 'r-', linewidth=2)
        best_epoch = np.argmax(history['val_acc']) + 1
        best_acc = max(history['val_acc'])
        axes[1, 1].scatter([best_epoch], [best_acc], color='gold', s=200, 
                          marker='*', edgecolors='black', linewidths=2, 
                          label=f'Best: {best_acc:.2f}% (Epoch {best_epoch})', zorder=5)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Validation Accuracy (%)')
        axes[1, 1].set_title('Validation Accuracy Progress')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'training_curves_{model_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_ngram_analysis(bleu_metrics, results_dir, model_name='baseline'):
        """
        Docstring for plot_ngram_analysis
        
        :param bleu_metrics: The BLEU scores for BLEU-1 to BLEU-4
        :param results_dir: The path where the plots will be saved
        :param model_name: The baseline or the challenger
        """

        _, ax = plt.subplots(figsize=(10, 6))
        
        ngrams = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']
        scores = [bleu_metrics['bleu1'], bleu_metrics['bleu2'], 
                 bleu_metrics['bleu3'], bleu_metrics['bleu4']]
        
        bars = ax.bar(ngrams, scores, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'], 
                     alpha=0.8, edgecolor='black', linewidth=2)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.1f}%',
                   ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax.set_ylabel('BLEU Score (%)', fontsize=12, fontweight='bold')
        ax.set_title('N-gram Analysis (BLEU Scores)', fontsize=14, fontweight='bold', pad=15)
        ax.set_ylim([0, max(scores) * 1.2])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'ngram_analysis_{model_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"N-gram analysis plot saved for {model_name} model.")

def print_all_metrics(metrics, model_name='baseline'):
        """
        Prints all metrics
        
        :param metrics: The dictionary with the metrics. Expects classification (accuracy, macro_f1, weighted_f1), bleu (bleu1, bleu2, bleu3, bleu4), meteor (meteor), 
        rouge(rouge1, rouge2, rougeL), bertscore (bertscore_precision, bertscore_recall, bertscore_f1), exact_match (exact_match), by_type(ans_type (accuracy, f1, exact_match)) metrics.
        :param model_name: The baseline or the challenger
        """
        print(f"COMPREHENSIVE METRICS RESULTS FOR {model_name.upper()} MODEL")
        
        # Classification Metrics
        print("\nCLASSIFICATION METRICS")
        cls_metrics = metrics['classification']
        print(f"  Accuracy:           {cls_metrics['accuracy']:.2f}%")
        print(f"  Macro F1 Score:     {cls_metrics['macro_f1']:.2f}%")
        print(f"  Weighted F1 Score:  {cls_metrics['weighted_f1']:.2f}%")
        
        # Exact Match
        print("\nEXACT MATCH")
        print(f"  Exact Match:        {metrics['exact_match']['exact_match']:.2f}%")
        
        # BLEU Scores
        print("\nBLEU SCORES")
        bleu_metrics = metrics['bleu']
        print(f"  BLEU-1:             {bleu_metrics['bleu1']:.2f}%")
        print(f"  BLEU-2:             {bleu_metrics['bleu2']:.2f}%")
        print(f"  BLEU-3:             {bleu_metrics['bleu3']:.2f}%")
        print(f"  BLEU-4:             {bleu_metrics['bleu4']:.2f}%")

        # ROUGE Scores
        print("\nROUGE SCORES")
        rouge_metrics = metrics['rouge']
        print(f"  ROUGE-1:            {rouge_metrics['rouge1']:.2f}%")
        print(f"  ROUGE-2:            {rouge_metrics['rouge2']:.2f}%")
        print(f"  ROUGE-L:            {rouge_metrics['rougeL']:.2f}%")
        
        # METEOR
        print("\nMETEOR SCORE")
        print(f"  METEOR:             {metrics['meteor']['meteor']:.2f}%")
        
        # BERTScore
        print("\nBERTSCORE")
        bert_metrics = metrics['bertscore']
        print(f"  BERTScore Precision: {bert_metrics['bertscore_precision']:.2f}%")
        print(f"  BERTScore Recall:    {bert_metrics['bertscore_recall']:.2f}%")
        print(f"  BERTScore F1:        {bert_metrics['bertscore_f1']:.2f}%")
        
        # Type-specific
        print("\nTYPE-SPECIFIC METRICS")
        for q_type in sorted(metrics['by_type'].keys()):
            type_data = metrics['by_type'][q_type]
            print(f"\n  {q_type} ({type_data['count']} samples):")
            print(f"    Accuracy:     {type_data['accuracy']:.2f}%")
            print(f"    F1 Score:     {type_data['f1']:.2f}%")
            print(f"    Exact Match:  {type_data['exact_match']:.2f}%")

def plot_type_specific_comparison(type_metrics, results_dir, model_name='baseline'):
        """
        Plots answer-type specific comparison plots.
        
        :param type_metrics: Expects a dictionary with question types as key and accuracy, f1 and exact_match as values to each key
        :param results_dir: The path where the plots will be saved
        :param model_name: The baseline or the challenger
        """
        if not type_metrics:
            return
        
        _, ax = plt.subplots(figsize=(12, 6))
        
        question_types = sorted(type_metrics.keys())
        x = np.arange(len(question_types))
        width = 0.25
        
        accuracies = [type_metrics[t]['accuracy'] for t in question_types]
        f1_scores = [type_metrics[t]['f1'] for t in question_types]
        exact_matches = [type_metrics[t]['exact_match'] for t in question_types]
        
        _ = ax.bar(x - width, accuracies, width, label='Accuracy', alpha=0.8)
        _ = ax.bar(x, f1_scores, width, label='F1 Score', alpha=0.8)
        _ = ax.bar(x + width, exact_matches, width, label='Exact Match', alpha=0.8)
        
        ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Question Type', fontsize=12, fontweight='bold')
        ax.set_title('Metrics by Question Type', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(question_types)
        ax.legend(fontsize=11)
        ax.set_ylim([0, 105])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'type_specific_comparison_{model_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Type-specific comparison plot saved")