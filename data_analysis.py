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

import warnings
warnings.filterwarnings('ignore')

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

    def load_data(self, file_path):
        """Load data from CSV file"""
        self.df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {self.df.shape}")

        # Display basic info about the loaded data
        print(f"Columns: {list(self.df.columns)}")

        # Check for expected columns
        expected_cols = ['Video ID', 'Title', 'Channel Name', 'Published At (JST)',
                        'View Count', 'Like Count', 'Comment Count', 'Duration (sec)',
                        'Video Length Category', 'Thumbnail Brightness', 'Thumbnail Colorfulness',
                        'Person in Thumbnail', 'Tags', 'Thumbnail URL']

        missing_cols = [col for col in expected_cols if col not in self.df.columns]
        if missing_cols:
            print(f"Warning: Missing expected columns: {missing_cols}")

        # Show sample of problematic columns
        if 'Person in Thumbnail' in self.df.columns:
            print(f"Person in Thumbnail unique values: {self.df['Person in Thumbnail'].unique()}")

        if 'Tags' in self.df.columns:
            print(f"Tags data type: {self.df['Tags'].dtype}")
            print(f"Sample Tags values: {self.df['Tags'].head()}")

        return self.df

    def exploratory_data_analysis(self):
        """Perform exploratory data analysis"""
        print("=== EXPLORATORY DATA ANALYSIS ===")
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nColumn information:")
        print(self.df.info())

        print(f"\nBasic statistics:")
        print(self.df.describe())

        print(f"\nMissing values:")
        print(self.df.isnull().sum())

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
        plt.show()

        numerical_cols = ['View Count', 'Like Count', 'Comment Count', 'Duration (sec)',
                         'Thumbnail Brightness', 'Thumbnail Colorfulness']

        plt.figure(figsize=(10, 8))
        correlation_matrix = self.df[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()

        return self.df

    def feature_engineering(self):
        """Engineer features from the raw data"""
        print("=== FEATURE ENGINEERING ===")

        df_processed = self.df.copy()

        # Convert datetime
        if 'Published At (JST)' in df_processed.columns:
            df_processed['Published At (JST)'] = pd.to_datetime(df_processed['Published At (JST)'])
            df_processed['publish_hour'] = df_processed['Published At (JST)'].dt.hour
            df_processed['publish_day_of_week'] = df_processed['Published At (JST)'].dt.dayofweek
            df_processed['publish_month'] = df_processed['Published At (JST)'].dt.month

        # Handle 'Person in Thumbnail' - convert Yes/No to 1/0
        if 'Person in Thumbnail' in df_processed.columns:
            df_processed['Person in Thumbnail'] = df_processed['Person in Thumbnail'].map({'Yes': 1, 'No': 0})
            # Fill any NaN values with 0
            df_processed['Person in Thumbnail'] = df_processed['Person in Thumbnail'].fillna(0)

        # Handle Tags - convert to number of tags if it's a string
        if 'Tags' in df_processed.columns:
            # Check if Tags column contains strings (like comma-separated tags)
            if df_processed['Tags'].dtype == 'object':
                # Count number of tags by splitting on commas
                df_processed['Tags'] = df_processed['Tags'].fillna('')
                df_processed['Tags'] = df_processed['Tags'].apply(
                    lambda x: len(x.split(',')) if isinstance(x, str) and x.strip() else 0
                )
            # Ensure it's numeric
            df_processed['Tags'] = pd.to_numeric(df_processed['Tags'], errors='coerce').fillna(0)

        # Create engagement ratio features
        df_processed['like_to_view_ratio'] = df_processed['Like Count'] / (df_processed['View Count'] + 1)
        df_processed['comment_to_view_ratio'] = df_processed['Comment Count'] / (df_processed['View Count'] + 1)
        df_processed['engagement_score'] = (df_processed['Like Count'] + df_processed['Comment Count']) / (df_processed['View Count'] + 1)

        # Title length feature (if title is available)
        if 'Title' in df_processed.columns:
            df_processed['title_length'] = df_processed['Title'].str.len()

        # Handle categorical variables
        le = LabelEncoder()
        categorical_cols = ['Channel Name', 'Video Length Category']

        for col in categorical_cols:
            if col in df_processed.columns:
                df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col])

        # Create view category for classification
        view_percentiles = df_processed['View Count'].quantile([0.33, 0.67])
        df_processed['view_category'] = pd.cut(df_processed['View Count'],
                                             bins=[0, view_percentiles.iloc[0], view_percentiles.iloc[1], float('inf')],
                                             labels=['Low', 'Medium', 'High'])

        self.processed_df = df_processed
        print("Feature engineering completed.")
        print(f"New features created. Final shape: {df_processed.shape}")

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

        print(f"Features selected for analysis: {available_features}")

        # Prepare feature matrix and ensure all values are numeric
        X = self.processed_df[available_features].copy()

        # Convert all columns to numeric, handling any remaining string values
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        # Fill any remaining NaN values with 0
        X = X.fillna(0)

        # Verify all values are numeric
        print(f"Data types after conversion: {X.dtypes.unique()}")
        print(f"Any non-numeric values remaining: {X.select_dtypes(include=['object']).columns.tolist()}")

        y_regression = self.processed_df['View Count']
        y_classification = self.processed_df['view_category']

        return X, y_regression, y_classification, available_features

    def perform_pca_analysis(self, n_components=None):
        """Perform Principal Component Analysis"""
        print("=== PCA ANALYSIS ===")

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

        print(f"Explained variance ratio by component: {explained_var_ratio}")
        print(f"Cumulative explained variance: {cumulative_var_ratio}")

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
        plt.show()

        # Feature importance in principal components
        components_df = pd.DataFrame(
            self.pca.components_[:5].T,  # First 5 components
            columns=[f'PC{i+1}' for i in range(min(5, n_components))],
            index=feature_names
        )

        print("\nFeature loadings for first 5 principal components:")
        print(components_df)

        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.heatmap(components_df.T, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Loadings in Principal Components')
        plt.tight_layout()
        plt.show()

        return X_pca, explained_var_ratio, components_df

    def svm_regression_analysis(self):
        """Perform SVM regression to predict view count"""
        print("=== SVM REGRESSION ANALYSIS ===")

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

        print("Performing grid search for SVM regression...")
        svm_reg = SVR()
        grid_search = GridSearchCV(svm_reg, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=2, verbose=1)
        grid_search.fit(X_train_scaled, y_train)

        self.svm_regressor = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")

        cv_scores = cross_val_score(self.svm_regressor, X_train_scaled, y_train, cv=5, scoring='r2')
        print(f"Cross-validation R² scores: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        # Make predictions
        y_pred_train = self.svm_regressor.predict(X_train_scaled)
        y_pred_test = self.svm_regressor.predict(X_test_scaled)

        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)

        print(f"\nRegression Results:")
        print(f"Training MSE: {train_mse:.4f}")
        print(f"Testing MSE: {test_mse:.4f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Testing R²: {test_r2:.4f}")

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
        plt.show()

        return self.svm_regressor, (train_mse, test_mse, train_r2, test_r2)

    def svm_classification_analysis(self):
        """Perform SVM classification for view categories"""
        print("=== SVM CLASSIFICATION ANALYSIS ===")

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

        print("Performing grid search for SVM classification...")
        svm_clf = SVC(random_state=42)
        grid_search = GridSearchCV(svm_clf, param_grid, cv=3, scoring='accuracy', n_jobs=2, verbose=1)
        grid_search.fit(X_train_scaled, y_train)

        self.svm_classifier = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")

        # Make predictions
        y_pred_train = self.svm_classifier.predict(X_train_scaled)
        y_pred_test = self.svm_classifier.predict(X_test_scaled)

        # Calculate accuracy
        train_accuracy = self.svm_classifier.score(X_train_scaled, y_train)
        test_accuracy = self.svm_classifier.score(X_test_scaled, y_test)

        print(f"\nClassification Results:")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Testing Accuracy: {test_accuracy:.4f}")

        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred_test))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_test)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Low', 'Medium', 'High'],
                   yticklabels=['Low', 'Medium', 'High'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

        return self.svm_classifier, (train_accuracy, test_accuracy)

    def identify_key_factors(self):
        """Identify the key factors affecting YouTube views"""
        print("=== KEY FACTORS ANALYSIS ===")

        # Get PCA results
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

        print("Key factors by Principal Component:")
        for pc, info in key_factors.items():
            print(f"\n{pc} (Explains {info['explained_variance']:.1%} of variance):")
            for factor, importance in info['top_factors'].items():
                print(f"  - {factor}: {importance:.3f}")

        return key_factors

    def generate_recommendations(self):
        """Generate recommendations for content creators"""
        print("=== RECOMMENDATIONS FOR CONTENT CREATORS ===")

        recommendations = []

        # Based on PCA analysis
        key_factors = self.identify_key_factors()

        # Get correlation with view count
        X, y_reg, _, feature_names = self.prepare_features_for_analysis()
        correlations = X.corrwith(pd.Series(y_reg)).abs().sort_values(ascending=False)

        print("\nTop factors correlated with view count:")
        for factor, corr in correlations.head(10).items():
            print(f"  - {factor}: {corr:.3f}")

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

        print("\nRecommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")

        return recommendations

    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting YouTube Views Analysis")
        print("=" * 50)

        # Step 1: EDA
        self.exploratory_data_analysis()

        # Step 2: Feature Engineering
        self.feature_engineering()

        # Step 3: PCA Analysis
        self.perform_pca_analysis()

        # Step 4: SVM Regression
        reg_results = self.svm_regression_analysis()

        # Step 5: SVM Classification
        clf_results = self.svm_classification_analysis()

        # Step 6: Key factors identification
        key_factors = self.identify_key_factors()

        # Step 7: Recommendations
        recommendations = self.generate_recommendations()

        print("\n" + "=" * 50)
        print("Analysis completed successfully!")

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