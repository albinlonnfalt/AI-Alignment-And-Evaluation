import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

class Visualizer:
    def __init__(self, df_rows: pd.DataFrame, output_folder: str):
        self.df_rows = df_rows
        self.output_folder = output_folder
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)

    def visualize(self):
        """Create simple visualizations of numerical data."""
        if self.df_rows is None or self.df_rows.empty:
            print("No data available for visualization.")
            return
        numerical_cols = self.df_rows.select_dtypes(include=['number']).columns.tolist()
        
        if not numerical_cols:
            print("No numerical columns found.")
            return
        
        # Set clean style
        sns.set_style("whitegrid")
        
        # Create histograms
        self._create_histograms(numerical_cols)
        
        # Create box plots if multiple columns
        if len(numerical_cols) > 1:
            self._create_boxplots(numerical_cols)
        
        # Create visualizations for boolean data only (no categorical)
        boolean_cols = self._identify_boolean_columns()
        if boolean_cols:
            self._create_boolean_charts(boolean_cols)

    def _identify_boolean_columns(self):
        """Identify columns that contain boolean-like data."""
        boolean_cols = []
        
        for col in self.df_rows.columns:
            try:
                # Skip columns with complex data types (dicts, lists, etc.)
                if self.df_rows[col].dtype == 'object':
                    # Check if any values are unhashable (dict, list, etc.)
                    sample_data = self.df_rows[col].dropna().head(10)
                    if any(isinstance(val, (dict, list)) for val in sample_data):
                        continue
                
                unique_vals = self.df_rows[col].dropna().unique()
                
                # Check for explicit boolean columns
                if self.df_rows[col].dtype == bool:
                    boolean_cols.append(col)
                # Check for True/False strings
                elif set(str(v).lower() for v in unique_vals if pd.notna(v)).issubset({'true', 'false'}):
                    boolean_cols.append(col)
                # Check for actual boolean values mixed with strings
                elif set(unique_vals).issubset({True, False}):
                    boolean_cols.append(col)
                # Check for pass/fail pattern
                elif set(str(v).lower() for v in unique_vals if pd.notna(v)).issubset({'pass', 'fail'}):
                    boolean_cols.append(col)
                    
            except (TypeError, ValueError) as e:
                # Skip columns that cause errors (likely contain unhashable types)
                print(f"Skipping column '{col}' due to data type issues: {str(e)[:100]}")
                continue
        
        return boolean_cols

    def _create_histograms(self, numerical_cols):
        """Create simple histograms for each numerical column."""
        n_cols = len(numerical_cols)
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
        
        if n_cols == 1:
            axes = [axes]
        
        for i, col in enumerate(numerical_cols):
            data = self.df_rows[col].dropna()
            axes[i].hist(data, bins=15, alpha=0.7, color='skyblue')
            axes[i].set_title(f'{col}')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, 'histograms.png'), dpi=150)
        plt.show()

    def _create_boxplots(self, numerical_cols):
        """Create box plots for comparison."""
        plt.figure(figsize=(8, 6))
        self.df_rows[numerical_cols].boxplot()
        plt.title('Score Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, 'boxplots.png'), dpi=150)
        plt.show()

    def _create_boolean_charts(self, boolean_cols):
        """Create bar charts for boolean columns."""
        n_cols = len(boolean_cols)
        if n_cols == 0:
            return
            
        fig, axes = plt.subplots(1, min(n_cols, 4), figsize=(4 * min(n_cols, 4), 4))
        
        if n_cols == 1:
            axes = [axes]
        elif n_cols > 4:
            # If more than 4 columns, create multiple figures
            self._create_boolean_charts_multiple_figures(boolean_cols)
            return
        
        for i, col in enumerate(boolean_cols[:4]):
            # Convert to standardized format for counting
            data = self.df_rows[col].dropna()
            
            # Normalize boolean-like values
            normalized_data = data.apply(self._normalize_boolean_value)
            value_counts = normalized_data.value_counts()
            
            # Create bar chart
            colors = ['lightcoral' if 'False' in str(k) or 'Fail' in str(k) else 'lightgreen' 
                     for k in value_counts.index]
            
            bars = axes[i].bar(value_counts.index, value_counts.values, color=colors, alpha=0.7)
            axes[i].set_title(f'{col}\n(n={len(data)})')
            axes[i].set_ylabel('Count')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, 'boolean_metrics.png'), dpi=150, bbox_inches='tight')
        plt.show()

    def _create_boolean_charts_multiple_figures(self, boolean_cols):
        """Create multiple figures if there are many boolean columns."""
        for i in range(0, len(boolean_cols), 4):
            chunk = boolean_cols[i:i+4]
            self._create_boolean_charts(chunk)

    def _normalize_boolean_value(self, value):
        """Normalize different boolean representations to standard format."""
        if pd.isna(value):
            return 'Unknown'
        
        str_val = str(value).lower()
        
        if str_val in ['true', '1', 'yes', 'pass']:
            return 'True/Pass'
        elif str_val in ['false', '0', 'no', 'fail']:
            return 'False/Fail'
        else:
            return str(value)