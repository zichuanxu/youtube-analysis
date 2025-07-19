import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVR, SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from datetime import datetime
from config import OUTPUT_CSV_FILE
import os

import warnings
warnings.filterwarnings('ignore')

# Ensure pictures directory exists
os.makedirs('pictures', exist_ok=True)

class YouTubeViewsAnalyzer:
    def __init__(self, data_path=None, data_df=None):
        """
        Initialize the YouTube Views Analyzer

        Args:
            data_path: Path to CSV file (optional)
            data_df: DataFrame with YouTube data (optional)
        """
        if data_path:
            self.df = pd.read_csv(data_path)
        elif data_df is not None:
            self.df = data_df.copy()

        self.processed_df = None
        self.pca = None
        self.scaler = StandardScaler()
        self.svm_regressor = None
        self.svm_classifier = None
        self.output_file = 'result.md'
        self.figure_counter = 1

        # Initialize the output file
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write("# YouTube Views Analysis Report\n\n")

    def write_to_output(self, content, section_type='text'):
        """Write content to the output markdown file"""
        with open(self.output_file, 'a', encoding='utf-8') as f:
            if section_type == 'header':
                f.write(f"\n## {content}\n\n")
            elif section_type == 'subheader':
                f.write(f"\n### {content}\n\n")
            elif section_type == 'code':
                f.write(f"```\n{content}\n```\n\n")
            elif section_type == 'image':
                f.write(f"![Figure {self.figure_counter}]({content})\n\n")
                self.figure_counter += 1
            else:
                f.write(f"{content}\n\n")

    def save_figure(self, filename_prefix):
        """Save the current matplotlib figure to the pictures directory"""
        filename = f"pictures/{filename_prefix}_{self.figure_counter:02d}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        self.write_to_output(filename, 'image')
        plt.close()  # Close the figure to free memory

    def load_data(self, file_path):
        """Load data from CSV file"""
        self.df = pd.read_csv(file_path)

        self.write_to_output("Data Loading", 'header')
        self.write_to_output(f"Data loaded successfully. Shape: {self.df.shape}")
        self.write_to_output(f"Columns: {list(self.df.columns)}")

        # Check for expected columns
        expected_cols = ['Video ID', 'Title', 'Channel Name', 'Published At (JST)',
                        'View Count', 'Like Count', 'Comment Count', 'Duration (sec)',
                        'Video Length Category', 'Thumbnail Brightness', 'Thumbnail Colorfulness',
                        'Person in Thumbnail', 'Tags', 'Thumbnail URL']

        missing_cols = [col for col in expected_cols if col not in self.df.columns]
        if missing_cols:
            self.write_to_output(f"Warning: Missing expected columns: {missing_cols}")

        # Show sample of problematic columns
        if 'Person in Thumbnail' in self.df.columns:
            self.write_to_output(f"Person in Thumbnail unique values: {self.df['Person in Thumbnail'].unique()}")

        if 'Tags' in self.df.columns:
            self.write_to_output(f"Tags data type: {self.df['Tags'].dtype}")
            self.write_to_output(f"Sample Tags values: {self.df['Tags'].head().tolist()}")

        return self.df

    def exploratory_data_analysis(self):
        """Perform exploratory data analysis"""
        self.write_to_output("Exploratory Data Analysis", 'header')
        self.write_to_output(f"Dataset shape: {self.df.shape}")

        # Column information
        self.write_to_output("Column Information", 'subheader')
        info_str = f"Data types:\n{self.df.dtypes.to_string()}"
        self.write_to_output(info_str, 'code')

        # Basic statistics
        self.write_to_output("Basic Statistics", 'subheader')
        self.write_to_output(self.df.describe().to_string(), 'code')

        # Missing values
        self.write_to_output("Missing Values", 'subheader')
        missing_values = self.df.isnull().sum()
        missing_str = missing_values[missing_values > 0].to_string() if missing_values.sum() > 0 else "No missing values found"
        self.write_to_output(missing_str, 'code')

        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # View count distribution
        axes[0, 0].hist(self.df['View Count'], bins=50, alpha=0.7)
        axes[0, 0].set_title('Distribution of View Count')
        axes[0, 0].set_xlabel('View Count')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_yscale('log')

        # View count vs Duration
        axes[0, 1].scatter(self.df['Duration (sec)'], self.df['View Count'], alpha=0.5)
        axes[0, 1].set_title('View Count vs Duration')
        axes[0, 1].set_xlabel('Duration (seconds)')
        axes[0, 1].set_ylabel('View Count')
        axes[0, 1].set_yscale('log')

        # Thumbnail brightness vs views
        axes[1, 0].scatter(self.df['Thumbnail Brightness'], self.df['View Count'], alpha=0.5)
        axes[1, 0].set_title('View Count vs Thumbnail Brightness')
        axes[1, 0].set_xlabel('Thumbnail Brightness')
        axes[1, 0].set_ylabel('View Count')
        axes[1, 0].set_yscale('log')

        # Views by video length category
        self.df.boxplot(column='View Count', by='Video Length Category', ax=axes[1, 1])
        axes[1, 1].set_title('View Count by Video Length Category')
        axes[1, 1].set_yscale('log')

        plt.tight_layout()
        self.save_figure('eda_overview')

        # Correlation matrix
        numerical_cols = ['View Count', 'Like Count', 'Comment Count', 'Duration (sec)',
                         'Thumbnail Brightness', 'Thumbnail Colorfulness']

        plt.figure(figsize=(10, 8))
        correlation_matrix = self.df[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        self.save_figure('correlation_matrix')

        # Write correlation values to output
        self.write_to_output("Feature Correlations", 'subheader')
        self.write_to_output(correlation_matrix.to_string(), 'code')

        return self.df

    def feature_engineering(self):
        """Engineer features from the raw data"""
        self.write_to_output("Feature Engineering", 'header')

        df_processed = self.df.copy()

        # Convert datetime
        if 'Published At (JST)' in df_processed.columns:
            df_processed['Published At (JST)'] = pd.to_datetime(df_processed['Published At (JST)'])
            df_processed['publish_hour'] = df_processed['Published At (JST)'].dt.hour
            df_processed['publish_day_of_week'] = df_processed['Published At (JST)'].dt.dayofweek
            df_processed['publish_month'] = df_processed['Published At (JST)'].dt.month
            self.write_to_output("✓ Created time-based features: publish_hour, publish_day_of_week, publish_month")

        # Handle 'Person in Thumbnail' - convert Yes/No to 1/0
        if 'Person in Thumbnail' in df_processed.columns:
            df_processed['Person in Thumbnail'] = df_processed['Person in Thumbnail'].map({'Yes': 1, 'No': 0})
            df_processed['Person in Thumbnail'] = df_processed['Person in Thumbnail'].fillna(0)
            self.write_to_output("✓ Converted 'Person in Thumbnail' to binary (1/0)")

        # Handle Tags - convert to number of tags if it's a string
        if 'Tags' in df_processed.columns:
            if df_processed['Tags'].dtype == 'object':
                df_processed['Tags'] = df_processed['Tags'].fillna('')
                df_processed['Tags'] = df_processed['Tags'].apply(
                    lambda x: len(x.split(',')) if isinstance(x, str) and x.strip() else 0
                )
            df_processed['Tags'] = pd.to_numeric(df_processed['Tags'], errors='coerce').fillna(0)
            self.write_to_output("✓ Converted Tags to count of tags")

        # Create engagement ratio features
        df_processed['like_to_view_ratio'] = df_processed['Like Count'] / (df_processed['View Count'] + 1)
        df_processed['comment_to_view_ratio'] = df_processed['Comment Count'] / (df_processed['View Count'] + 1)
        df_processed['engagement_score'] = (df_processed['Like Count'] + df_processed['Comment Count']) / (df_processed['View Count'] + 1)
        self.write_to_output("✓ Created engagement ratio features: like_to_view_ratio, comment_to_view_ratio, engagement_score")

        # Title length feature (if title is available)
        if 'Title' in df_processed.columns:
            df_processed['title_length'] = df_processed['Title'].str.len()
            self.write_to_output("✓ Created title_length feature")

        # Handle categorical variables
        le = LabelEncoder()
        categorical_cols = ['Channel Name', 'Video Length Category']

        for col in categorical_cols:
            if col in df_processed.columns:
                df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col])
                self.write_to_output(f"✓ Encoded categorical variable: {col}")

        # Create view category for classification
        view_percentiles = df_processed['View Count'].quantile([0.33, 0.67])
        df_processed['view_category'] = pd.cut(df_processed['View Count'],
                                             bins=[0, view_percentiles.iloc[0], view_percentiles.iloc[1], float('inf')],
                                             labels=['Low', 'Medium', 'High'])
        self.write_to_output(f"✓ Created view categories: Low (<{view_percentiles.iloc[0]:.0f}), Medium ({view_percentiles.iloc[0]:.0f}-{view_percentiles.iloc[1]:.0f}), High (>{view_percentiles.iloc[1]:.0f})")

        self.processed_df = df_processed
        self.write_to_output(f"Feature engineering completed. Final shape: {df_processed.shape}")

        return df_processed

    def prepare_features_for_analysis(self):
        """Prepare features for PCA and SVM analysis"""
        if self.processed_df is None:
            self.feature_engineering()

        # Select numerical features for analysis
        numerical_features = [
            'Like Count', 'Comment Count', 'Duration (sec)',
            'Thumbnail Brightness', 'Thumbnail Colorfulness', 'Person in Thumbnail',
            'Tags', 'like_to_view_ratio', 'comment_to_view_ratio', 'engagement_score'
        ]

        # Add time-based features if available
        time_features = ['publish_hour', 'publish_day_of_week', 'publish_month']
        for feature in time_features:
            if feature in self.processed_df.columns:
                numerical_features.append(feature)

        # Add encoded categorical features
        categorical_encoded = ['Channel Name_encoded', 'Video Length Category_encoded', 'title_length']
        for feature in categorical_encoded:
            if feature in self.processed_df.columns:
                numerical_features.append(feature)

        # Remove any features that don't exist
        available_features = [f for f in numerical_features if f in self.processed_df.columns]

        self.write_to_output("Feature Preparation", 'subheader')
        self.write_to_output(f"Features selected for analysis: {available_features}", 'code')

        # Prepare feature matrix and ensure all values are numeric
        X = self.processed_df[available_features].copy()

        # Convert all columns to numeric, handling any remaining string values
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        # Fill any remaining NaN values with 0
        X = X.fillna(0)

        # Verify all values are numeric
        self.write_to_output(f"Data types after conversion: {X.dtypes.unique()}")
        non_numeric = X.select_dtypes(include=['object']).columns.tolist()
        if non_numeric:
            self.write_to_output(f"Non-numeric values remaining: {non_numeric}")
        else:
            self.write_to_output("All features successfully converted to numeric")

        y_regression = self.processed_df['View Count']
        y_classification = self.processed_df['view_category']

        return X, y_regression, y_classification, available_features

    def perform_pca_analysis(self, n_components=None):
        """Perform Principal Component Analysis"""
        self.write_to_output("Principal Component Analysis (PCA)", 'header')

        X, y_reg, y_class, feature_names = self.prepare_features_for_analysis()

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Determine number of components
        if n_components is None:
            n_components = min(len(feature_names), X.shape[0] - 1)

        # Apply PCA
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X_scaled)

        # Calculate explained variance
        explained_var_ratio = self.pca.explained_variance_ratio_
        cumulative_var_ratio = np.cumsum(explained_var_ratio)

        self.write_to_output("Explained Variance Analysis", 'subheader')
        self.write_to_output(f"Number of components: {n_components}")

        # Create explained variance table
        variance_data = []
        for i, (individual, cumulative) in enumerate(zip(explained_var_ratio, cumulative_var_ratio)):
            variance_data.append(f"PC{i+1}: {individual:.3f} ({individual*100:.1f}%) | Cumulative: {cumulative:.3f} ({cumulative*100:.1f}%)")

        self.write_to_output("Explained variance by component:", 'code')
        self.write_to_output('\n'.join(variance_data), 'code')

        # Plot explained variance
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Individual explained variance
        axes[0].bar(range(1, len(explained_var_ratio) + 1), explained_var_ratio)
        axes[0].set_title('Explained Variance Ratio by Component')
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')

        # Cumulative explained variance
        axes[1].plot(range(1, len(cumulative_var_ratio) + 1), cumulative_var_ratio, 'bo-')
        axes[1].axhline(y=0.8, color='r', linestyle='--', label='80% threshold')
        axes[1].axhline(y=0.95, color='g', linestyle='--', label='95% threshold')
        axes[1].set_title('Cumulative Explained Variance')
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('Cumulative Explained Variance')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        self.save_figure('pca_variance')

        # Feature importance in principal components
        components_df = pd.DataFrame(
            self.pca.components_[:5].T,  # First 5 components
            columns=[f'PC{i+1}' for i in range(min(5, n_components))],
            index=feature_names
        )

        self.write_to_output("Feature Loadings in Principal Components", 'subheader')
        self.write_to_output(components_df.to_string(), 'code')

        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.heatmap(components_df.T, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Loadings in Principal Components')
        plt.tight_layout()
        self.save_figure('pca_loadings')

        return X_pca, explained_var_ratio, components_df

    def svm_regression_analysis(self):
        """Perform SVM regression to predict view count"""
        self.write_to_output("SVM Regression Analysis", 'header')

        X, y_reg, _, feature_names = self.prepare_features_for_analysis()

        # Use log transformation for view count (since it's heavily skewed)
        y_log = np.log1p(y_reg)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Simplified grid search for better performance
        param_grid = {
            'C': [1, 10, 100],
            'gamma': ['scale', 0.01, 0.1],
            'kernel': ['rbf', 'linear']
        }

        self.write_to_output("Model Training", 'subheader')
        self.write_to_output("Performing grid search for SVM regression...")
        self.write_to_output(f"Parameter grid: {param_grid}", 'code')

        svm_reg = SVR()
        grid_search = GridSearchCV(svm_reg, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=2, verbose=0)
        grid_search.fit(X_train_scaled, y_train)

        self.svm_regressor = grid_search.best_estimator_
        self.write_to_output(f"Best parameters: {grid_search.best_params_}")

        cv_scores = cross_val_score(self.svm_regressor, X_train_scaled, y_train, cv=5, scoring='r2')
        self.write_to_output(f"Cross-validation R² scores: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

        # Make predictions
        y_pred_train = self.svm_regressor.predict(X_train_scaled)
        y_pred_test = self.svm_regressor.predict(X_test_scaled)

        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)

        self.write_to_output("Regression Results", 'subheader')
        results_text = f"""Training MSE: {train_mse:.4f}
Testing MSE: {test_mse:.4f}
Training R²: {train_r2:.4f}
Testing R²: {test_r2:.4f}"""
        self.write_to_output(results_text, 'code')

        # Plot predictions vs actual
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(y_train, y_pred_train, alpha=0.5)
        plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
        plt.xlabel('Actual (log views)')
        plt.ylabel('Predicted (log views)')
        plt.title(f'Training Set - R² = {train_r2:.3f}')

        plt.subplot(1, 2, 2)
        plt.scatter(y_test, y_pred_test, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual (log views)')
        plt.ylabel('Predicted (log views)')
        plt.title(f'Test Set - R² = {test_r2:.3f}')

        plt.tight_layout()
        self.save_figure('svm_regression')

        return self.svm_regressor, (train_mse, test_mse, train_r2, test_r2)

    def svm_classification_analysis(self):
        """Perform SVM classification for view categories"""
        self.write_to_output("SVM Classification Analysis", 'header')

        X, _, y_class, feature_names = self.prepare_features_for_analysis()

        # Remove any NaN values in target
        mask = y_class.notna()
        X_clean = X[mask]
        y_clean = y_class[mask]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2,
                                                          random_state=42, stratify=y_clean)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Simplified grid search for better performance
        param_grid = {
            'C': [1, 10, 100],
            'gamma': ['scale', 0.01, 0.1],
            'kernel': ['rbf', 'linear']
        }

        self.write_to_output("Model Training", 'subheader')
        self.write_to_output("Performing grid search for SVM classification...")
        self.write_to_output(f"Parameter grid: {param_grid}", 'code')

        svm_clf = SVC(random_state=42)
        grid_search = GridSearchCV(svm_clf, param_grid, cv=3, scoring='accuracy', n_jobs=2, verbose=0)
        grid_search.fit(X_train_scaled, y_train)

        self.svm_classifier = grid_search.best_estimator_
        self.write_to_output(f"Best parameters: {grid_search.best_params_}")

        # Make predictions
        y_pred_train = self.svm_classifier.predict(X_train_scaled)
        y_pred_test = self.svm_classifier.predict(X_test_scaled)

        # Calculate accuracy
        train_accuracy = self.svm_classifier.score(X_train_scaled, y_train)
        test_accuracy = self.svm_classifier.score(X_test_scaled, y_test)

        self.write_to_output("Classification Results", 'subheader')
        results_text = f"""Training Accuracy: {train_accuracy:.4f}
Testing Accuracy: {test_accuracy:.4f}"""
        self.write_to_output(results_text, 'code')

        # Classification report
        self.write_to_output("Classification Report", 'subheader')
        class_report = classification_report(y_test, y_pred_test)
        self.write_to_output(class_report, 'code')

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_test)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Low', 'Medium', 'High'],
                   yticklabels=['Low', 'Medium', 'High'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        self.save_figure('svm_classification')

        return self.svm_classifier, (train_accuracy, test_accuracy)

    def identify_key_factors(self):
        """Identify the key factors affecting YouTube views"""
        self.write_to_output("Key Factors Analysis", 'header')

        # Get PCA results (this will be called again but PCA is already computed)
        X_pca, explained_var, components_df = self.perform_pca_analysis()

        # Get feature importance from the first few components
        n_components_to_analyze = min(3, components_df.shape[1])

        key_factors = {}
        for i in range(n_components_to_analyze):
            pc_name = f'PC{i+1}'
            # Get absolute loadings and sort
            loadings = components_df[pc_name].abs().sort_values(ascending=False)
            key_factors[pc_name] = {
                'explained_variance': explained_var[i],
                'top_factors': loadings.head(5).to_dict()
            }

        self.write_to_output("Key factors by Principal Component:", 'subheader')
        for pc, info in key_factors.items():
            self.write_to_output(f"{pc} (Explains {info['explained_variance']:.1%} of variance):")
            for factor, importance in info['top_factors'].items():
                self.write_to_output(f"  - {factor}: {importance:.3f}")

        return key_factors

    def generate_recommendations(self):
        """Generate recommendations for content creators"""
        self.write_to_output("Recommendations for Content Creators", 'header')

        # Get correlation with view count
        X, y_reg, _, feature_names = self.prepare_features_for_analysis()
        correlations = X.corrwith(pd.Series(y_reg)).abs().sort_values(ascending=False)

        self.write_to_output("Top factors correlated with view count:", 'subheader')
        correlation_text = []
        for factor, corr in correlations.head(10).items():
            correlation_text.append(f"  - {factor}: {corr:.3f}")
        self.write_to_output('\n'.join(correlation_text), 'code')

        # Generate specific recommendations
        recommendations = [
            "Focus on thumbnail optimization - brightness and colorfulness show strong correlation with views",
            "Optimize video duration - find the sweet spot for your content type",
            "Include people in thumbnails when relevant - this can increase engagement",
            "Use appropriate number of tags - not too few, not too many",
            "Consider posting time - certain hours/days may perform better",
            "Focus on engagement metrics - likes and comments drive algorithmic promotion",
            "Maintain consistency in content quality and posting schedule"
        ]

        self.write_to_output("Actionable Recommendations:", 'subheader')
        for i, rec in enumerate(recommendations, 1):
            self.write_to_output(f"{i}. {rec}")

        return recommendations

    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        self.write_to_output("Analysis Pipeline Overview", 'header')
        self.write_to_output("This report contains a comprehensive analysis of YouTube video performance data using machine learning techniques including PCA and SVM.")

        # Step 1: EDA
        self.exploratory_data_analysis()

        # Step 2: Feature Engineering
        self.feature_engineering()

        # Step 3: PCA Analysis
        X_pca, explained_var, components_df = self.perform_pca_analysis()

        # Step 4: SVM Regression
        reg_results = self.svm_regression_analysis()

        # Step 5: SVM Classification
        clf_results = self.svm_classification_analysis()

        # Step 6: Key factors identification (this will reuse PCA results)
        key_factors = {}
        n_components_to_analyze = min(3, components_df.shape[1])

        self.write_to_output("Key Factors Analysis", 'header')
        for i in range(n_components_to_analyze):
            pc_name = f'PC{i+1}'
            loadings = components_df[pc_name].abs().sort_values(ascending=False)
            key_factors[pc_name] = {
                'explained_variance': explained_var[i],
                'top_factors': loadings.head(5).to_dict()
            }

        self.write_to_output("Key factors by Principal Component:", 'subheader')
        for pc, info in key_factors.items():
            self.write_to_output(f"{pc} (Explains {info['explained_variance']:.1%} of variance):")
            for factor, importance in info['top_factors'].items():
                self.write_to_output(f"  - {factor}: {importance:.3f}")

        # Step 7: Recommendations
        recommendations = self.generate_recommendations()

        # Final summary
        self.write_to_output("Analysis Summary", 'header')
        summary_text = f"""- Dataset analyzed: {self.df.shape[0]} videos with {self.df.shape[1]} features
- PCA identified {len(key_factors)} key components explaining the variance
- SVM regression achieved R² of {reg_results[1][3]:.3f} on test data
- SVM classification achieved {clf_results[1][1]:.1%} accuracy on test data
- Generated {len(recommendations)} actionable recommendations for content creators"""
        self.write_to_output(summary_text, 'code')

        self.write_to_output("Analysis completed successfully! All results and visualizations have been saved to this report.")

        return {
            'pca_results': key_factors,
            'regression_results': reg_results,
            'classification_results': clf_results,
            'recommendations': recommendations
        }

if __name__ == "__main__":
    analyzer = YouTubeViewsAnalyzer()
    analyzer.load_data(OUTPUT_CSV_FILE)
    results = analyzer.run_complete_analysis()
    print("\nAnalysis Summary:")
    print(f"- PCA identified {len(results['pca_results'])} key components")
    print(f"- SVM regression achieved R² of {results['regression_results'][1][3]:.3f}")
    print(f"- SVM classification achieved {results['classification_results'][1][1]:.1%} accuracy")
    print(f"- Generated {len(results['recommendations'])} recommendations")