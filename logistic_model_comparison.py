# import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Libraries for image processing
from PIL import Image, ImageStat
import os
from datetime import datetime
import re

# Read data
df = pd.read_csv('all_countries_videos_cleaned.csv')

# Create is_viral label
df["is_viral"] = ((df["views"] > 100_000) & (df["likes"] > 10_000)).astype(int)

print("Data Overview:")
print(f"Total samples: {len(df)}")
print(f"Viral videos count: {df['is_viral'].sum()}")
print(f"Viral video percentage: {df['is_viral'].mean():.2%}")
print("\n" + "="*60)

# =============================================================================
# 1. Title Feature Analysis (keeping original logic)
# =============================================================================

def analyze_title_features(df):
    """Analyze the relationship between title features and video virality"""
    print("\n【Title Feature Analysis】")
    
    # Create title-related features
    df['title_length'] = df['title'].str.len()
    df['title_word_count'] = df['title'].str.split().str.len()
    df['title_has_caps'] = df['title'].str.contains(r'[A-Z]{2,}', na=False).astype(int)
    df['title_has_numbers'] = df['title'].str.contains(r'\d', na=False).astype(int)
    df['title_has_exclamation'] = df['title'].str.contains('!', na=False).astype(int)
    df['title_has_question'] = df['title'].str.contains('\?', na=False).astype(int)
    
    # Title feature comparison statistics
    title_features = ['title_length', 'title_word_count', 'title_has_caps', 
                     'title_has_numbers', 'title_has_exclamation', 'title_has_question']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    print("Title feature comparison:")
    for i, feature in enumerate(title_features):
        viral_mean = df[df['is_viral']==1][feature].mean()
        non_viral_mean = df[df['is_viral']==0][feature].mean()
        
        print(f"{feature}: Non-viral={non_viral_mean:.3f}, Viral={viral_mean:.3f}")
        
        axes[i].bar(['Non-viral', 'Viral'], [non_viral_mean, viral_mean], 
                   color=['lightcoral', 'lightgreen'])
        axes[i].set_title(f'{feature}')
        axes[i].set_ylabel('Mean value')
        
        # Display values
        axes[i].text(0, non_viral_mean + max(non_viral_mean*0.05, 0.1), f'{non_viral_mean:.2f}', 
                    ha='center', va='bottom')
        axes[i].text(1, viral_mean + max(viral_mean*0.05, 0.1), f'{viral_mean:.2f}', 
                    ha='center', va='bottom')
    
    plt.suptitle('Title Feature Comparison: Viral vs Non-viral Videos', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Use TF-IDF for title text analysis
    print("\nBuilding logistic regression model using title text features:")
    
    # Clean title text
    titles_clean = df['title'].fillna('').astype(str)
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', 
                                min_df=5, max_df=0.8)
    X_title_tfidf = vectorizer.fit_transform(titles_clean)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_title_tfidf, df['is_viral'], test_size=0.2, random_state=42, stratify=df['is_viral']
    )
    
    # Logistic regression model
    lr_title = LogisticRegression(random_state=42, max_iter=1000)
    lr_title.fit(X_train, y_train)
    
    # Prediction and evaluation
    y_pred = lr_title.predict(X_test)
    y_pred_proba = lr_title.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Title text model accuracy: {accuracy:.4f}")
    print(f"Title text model AUC: {auc_score:.4f}")
    
    return lr_title, accuracy, auc_score

# =============================================================================
# 2. Thumbnail Image Feature Analysis (using downloaded images)
# =============================================================================

def extract_image_features(image_path, sample_size=None):
    """Extract features from an image"""
    features = {}
    
    try:
        # Open image
        img = Image.open(image_path)
        
        # Basic info
        features['width'], features['height'] = img.size
        features['aspect_ratio'] = features['width'] / features['height']
        features['total_pixels'] = features['width'] * features['height']
        
        # Convert to RGB mode
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Color statistics
        stat = ImageStat.Stat(img)
        features['mean_red'] = stat.mean[0]
        features['mean_green'] = stat.mean[1] 
        features['mean_blue'] = stat.mean[2]
        features['std_red'] = stat.stddev[0]
        features['std_green'] = stat.stddev[1]
        features['std_blue'] = stat.stddev[2]
        
        # Brightness and contrast
        features['brightness'] = sum(stat.mean) / 3
        features['contrast'] = sum(stat.stddev) / 3
        
        # Color saturation (simplified calculation)
        features['saturation'] = np.std([stat.mean[0], stat.mean[1], stat.mean[2]])
        
        # Calculate mean_yellow (before determining warm tone)
        features['mean_yellow'] = (features['mean_red'] + features['mean_green']) / 2
        
        # Determine warm/cool tone
        features['warm_tone'] = 1 if (features['mean_red'] + features['mean_yellow']) > features['mean_blue'] else 0
        
        # Image complexity (based on standard deviation)
        features['complexity'] = features['contrast']
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        # Return default values
        default_features = ['width', 'height', 'aspect_ratio', 'total_pixels', 
                          'mean_red', 'mean_green', 'mean_blue', 'std_red', 'std_green', 'std_blue',
                          'brightness', 'contrast', 'saturation', 'warm_tone', 'mean_yellow', 'complexity']
        features = {f: 0 for f in default_features}
    
    return features

def get_thumbnail_path(video_id, thumbnails_folder='thumbnails'):
    """Build image path based on video_id"""
    return os.path.join(thumbnails_folder, f"{video_id}.jpg")

def analyze_thumbnail_features(df, sample_size=10000):
    """Analyze the relationship between thumbnail features and video virality"""
    print("\n" + "="*60)
    print("\n【Thumbnail Feature Analysis】")
    
    # Check if thumbnails folder exists
    thumbnails_folder = 'thumbnails'
    if not os.path.exists(thumbnails_folder):
        print(f"Error: {thumbnails_folder} folder not found")
        return None, 0, 0
    
    # Build image path for each video
    df['thumbnail_path'] = df['video_id'].apply(lambda x: get_thumbnail_path(x, thumbnails_folder))
    
    # Check actual existing image files
    df['thumbnail_exists'] = df['thumbnail_path'].apply(os.path.exists)
    df_with_thumbs = df[df['thumbnail_exists']].copy()
    
    print(f"Total dataset size: {len(df)}")
    print(f"Number of images in thumbnails folder: {len(os.listdir(thumbnails_folder))}")
    print(f"Number of matched thumbnails: {len(df_with_thumbs)}")
    
    if len(df_with_thumbs) == 0:
        print("Error: No matching image files found")
        print("Please check:")
        print("1. Is the thumbnails folder in the correct location?")
        print("2. Do image filenames match video_id?")
        print("3. Are image files in .jpg format?")
        return None, 0, 0
    
    # If dataset is too large, sample to speed up processing
    if len(df_with_thumbs) > sample_size:
        print(f"Large dataset, randomly sampling {sample_size} images for analysis")
        df_sample = df_with_thumbs.sample(n=sample_size, random_state=42)
    else:
        df_sample = df_with_thumbs
    
    print(f"Actual number of images analyzed: {len(df_sample)}")
    print("Starting image feature extraction...")
    
    # Extract image features
    image_features_list = []
    valid_indices = []
    
    for idx, row in df_sample.iterrows():
        if len(valid_indices) % 1000 == 0 and len(valid_indices) > 0:
            print(f"Successfully processed {len(valid_indices)} images")
        
        features = extract_image_features(row['thumbnail_path'])
        if features and features['width'] > 0:  # Ensure feature extraction was successful and valid
            image_features_list.append(features)
            valid_indices.append(idx)
    
    print(f"Number of images with successfully extracted features: {len(image_features_list)}")
    
    if len(image_features_list) == 0:
        print("Error: Unable to extract any image features")
        return None, 0, 0
    
    # Create feature DataFrame
    features_df = pd.DataFrame(image_features_list, index=valid_indices)
    
    # Merge with original data
    df_analysis = df_sample.loc[valid_indices].copy()
    for col in features_df.columns:
        df_analysis[col] = features_df[col]
    
    # Select main image features for analysis
    image_features = ['width', 'height', 'aspect_ratio', 'brightness', 'contrast', 
                     'saturation', 'warm_tone', 'complexity']
    
    # Visualize image feature comparison
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.ravel()
    
    print("\nImage feature comparison:")
    for i, feature in enumerate(image_features):
        viral_mean = df_analysis[df_analysis['is_viral']==1][feature].mean()
        non_viral_mean = df_analysis[df_analysis['is_viral']==0][feature].mean()
        
        print(f"{feature}: Non-viral={non_viral_mean:.3f}, Viral={viral_mean:.3f}")
        
        if feature == 'warm_tone':  # For binary features use bar chart
            axes[i].bar(['Non-viral', 'Viral'], [non_viral_mean, viral_mean], 
                       color=['lightcoral', 'lightgreen'])
            axes[i].text(0, non_viral_mean + 0.02, f'{non_viral_mean:.3f}', 
                        ha='center', va='bottom')
            axes[i].text(1, viral_mean + 0.02, f'{viral_mean:.3f}', 
                        ha='center', va='bottom')
        else:  # For continuous features use histogram
            viral_data = df_analysis[df_analysis['is_viral']==1][feature]
            non_viral_data = df_analysis[df_analysis['is_viral']==0][feature]
            
            axes[i].hist(non_viral_data, alpha=0.6, label='Non-viral', bins=30, color='lightcoral', density=True)
            axes[i].hist(viral_data, alpha=0.6, label='Viral', bins=30, color='lightgreen', density=True)
            axes[i].legend()
        
        axes[i].set_title(f'{feature}')
        axes[i].set_ylabel('Density' if feature != 'warm_tone' else 'Proportion')
    
    plt.suptitle('Thumbnail Feature Comparison: Viral vs Non-viral Videos', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Build logistic regression model using thumbnail features
    print("\nBuilding logistic regression model using thumbnail features:")
    
    X_thumbnail = df_analysis[image_features].fillna(0)
    y = df_analysis['is_viral']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_thumbnail, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr_thumbnail = LogisticRegression(random_state=42)
    lr_thumbnail.fit(X_train_scaled, y_train)
    
    y_pred = lr_thumbnail.predict(X_test_scaled)
    y_pred_proba = lr_thumbnail.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Thumbnail feature model accuracy: {accuracy:.4f}")
    print(f"Thumbnail feature model AUC: {auc_score:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': image_features,
        'coefficient': lr_thumbnail.coef_[0],
        'abs_coefficient': np.abs(lr_thumbnail.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)
    
    print("\nThumbnail feature importance ranking:")
    print(feature_importance.to_string(index=False))
    
    return lr_thumbnail, accuracy, auc_score

# =============================================================================
# 3. Variable Analysis
# =============================================================================

def analyze_specific_variables(df):
    """Analyze the relationship between specific variables and virality"""
    print("\n" + "="*60)
    print("\n【Specific Variable Analysis】")
    
    # Process publish_time
    df['publish_time'] = pd.to_datetime(df['publish_time'])
    df['publish_hour'] = df['publish_time'].dt.hour
    df['publish_day_of_week'] = df['publish_time'].dt.dayofweek  # 0=Monday
    df['publish_month'] = df['publish_time'].dt.month
    
    # Process tags
    df['tags_count'] = df['tags'].fillna('').str.split('|').str.len()
    df['tags_count'] = df['tags_count'].replace(1, 0)  # Empty tags counted as 1, corrected to 0
    df['has_tags'] = (df['tags_count'] > 0).astype(int)
    
    # Process description
    df['description_length'] = df['description'].fillna('').str.len()
    df['has_description'] = (df['description_length'] > 0).astype(int)
    df['description_word_count'] = df['description'].fillna('').str.split().str.len()
    
    # Process boolean fields
    df['comments_disabled_int'] = df['comments_disabled'].astype(int)
    df['ratings_disabled_int'] = df['ratings_disabled'].astype(int)
    
    # Define features to analyze
    specific_features = {
        'publish_time': ['publish_hour', 'publish_day_of_week', 'publish_month'],
        'tags': ['tags_count', 'has_tags'],
        'description': ['description_length', 'has_description', 'description_word_count'],
        'comments_disabled': ['comments_disabled_int'],
        'ratings_disabled': ['ratings_disabled_int']
    }
    
    all_features = [f for features in specific_features.values() for f in features]
    
    # Visualization analysis
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.ravel()
    
    print("Specific variable feature comparison:")
    for i, feature in enumerate(all_features):
        if i >= len(axes):
            break
            
        viral_data = df[df['is_viral']==1][feature]
        non_viral_data = df[df['is_viral']==0][feature]
        viral_mean = viral_data.mean()
        non_viral_mean = non_viral_data.mean()
        
        print(f"{feature}: Non-viral={non_viral_mean:.3f}, Viral={viral_mean:.3f}")
        
        # Use bar chart for discrete variables, histogram for continuous variables
        if feature in ['comments_disabled_int', 'ratings_disabled_int', 'has_tags', 'has_description']:
            axes[i].bar(['Non-viral', 'Viral'], [non_viral_mean, viral_mean], 
                       color=['lightcoral', 'lightgreen'])
            axes[i].text(0, non_viral_mean + 0.02, f'{non_viral_mean:.3f}', 
                        ha='center', va='bottom')
            axes[i].text(1, viral_mean + 0.02, f'{viral_mean:.3f}', 
                        ha='center', va='bottom')
            axes[i].set_ylabel('Proportion')
        else:
            axes[i].hist(non_viral_data, alpha=0.6, label='Non-viral', bins=30, color='lightcoral', density=True)
            axes[i].hist(viral_data, alpha=0.6, label='Viral', bins=30, color='lightgreen', density=True)
            axes[i].legend()
            axes[i].set_ylabel('Density')
        
        axes[i].set_title(f'{feature}')
    
    # Hide extra subplots
    for i in range(len(all_features), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Specific Variable Comparison: Viral vs Non-viral Videos', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Build models for each variable category
    results = {}
    
    for var_name, features in specific_features.items():
        print(f"\n--- {var_name} Variable Analysis ---")
        
        X = df[features].fillna(0)
        y = df['is_viral']
        
        # Handle outliers
        for feature in features:
            if df[feature].dtype in ['int64', 'float64']:
                Q1 = X[feature].quantile(0.25)
                Q3 = X[feature].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    X[feature] = X[feature].clip(lower_bound, upper_bound)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        lr = LogisticRegression(random_state=42)
        lr.fit(X_train_scaled, y_train)
        
        y_pred = lr.predict(X_test_scaled)
        y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"{var_name} model accuracy: {accuracy:.4f}")
        print(f"{var_name} model AUC: {auc_score:.4f}")
        
        # Feature importance
        if len(features) > 1:
            feature_importance = pd.DataFrame({
                'feature': features,
                'coefficient': lr.coef_[0],
                'abs_coefficient': np.abs(lr.coef_[0])
            }).sort_values('abs_coefficient', ascending=False)
            print(f"{var_name} feature importance:")
            print(feature_importance.to_string(index=False))
        
        results[var_name] = {'accuracy': accuracy, 'auc': auc_score, 'model': lr}
    
    return results

# =============================================================================
# 4. Comprehensive Model Performance Comparison
# =============================================================================

def compare_all_models(title_acc, title_auc, thumbnail_acc, thumbnail_auc, specific_results):
    """Compare performance of all models"""
    print("\n" + "="*60)
    print("\n【Comprehensive Model Performance Comparison】")
    
    # Prepare comparison data
    model_names = ['Title Features']
    accuracies = [title_acc]
    auc_scores = [title_auc]
    
    if thumbnail_acc > 0:  # If thumbnail analysis was successful
        model_names.append('Thumbnail Features')
        accuracies.append(thumbnail_acc)
        auc_scores.append(thumbnail_auc)
    
    for var_name, results in specific_results.items():
        model_names.append(f'{var_name}')
        accuracies.append(results['accuracy'])
        auc_scores.append(results['auc'])
    
    # Create comparison table
    comparison_df = pd.DataFrame({
        'Model Type': model_names,
        'Accuracy': accuracies,
        'AUC Score': auc_scores
    })
    
    print("Model performance comparison:")
    print(comparison_df.to_string(index=False))
    
    # Visualization comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    bars1 = ax1.bar(range(len(model_names)), accuracies, 
                    color=['skyblue', 'lightgreen', 'coral', 'gold', 'plum', 'lightgray', 'pink'][:len(model_names)])
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    
    # Display values on bars
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
    
    # AUC comparison
    bars2 = ax2.bar(range(len(model_names)), auc_scores, 
                    color=['skyblue', 'lightgreen', 'coral', 'gold', 'plum', 'lightgray', 'pink'][:len(model_names)])
    ax2.set_title('Model AUC Score Comparison')
    ax2.set_ylabel('AUC Score')
    ax2.set_ylim(0, 1)
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    
    # Display values on bars
    for i, v in enumerate(auc_scores):
        ax2.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Analysis conclusions
    print("\n【Analysis Conclusions】")
    best_acc_idx = np.argmax(accuracies)
    best_auc_idx = np.argmax(auc_scores)
    
    print(f"• Highest accuracy: {model_names[best_acc_idx]} ({accuracies[best_acc_idx]:.4f})")
    print(f"• Highest AUC score: {model_names[best_auc_idx]} ({auc_scores[best_auc_idx]:.4f})")
    
    # Evaluate feature importance
    high_performance_threshold = 0.60
    significant_features = []
    
    for i, (name, acc, auc) in enumerate(zip(model_names, accuracies, auc_scores)):
        if acc > high_performance_threshold or auc > high_performance_threshold:
            significant_features.append(name)
    
    if significant_features:
        print(f"\n• Features with significant predictive power: {', '.join(significant_features)}")
    else:
        print("\n• All single features have relatively weak predictive power, recommend combining multiple features")
    
    print(f"\n• Baseline accuracy (random guess): {max(df['is_viral'].mean(), 1-df['is_viral'].mean()):.4f}")
    print("• AUC > 0.6 indicates the feature has some predictive value")
    print("• AUC > 0.7 indicates the feature has strong predictive value")
    
    return comparison_df

# =============================================================================
# Execute Analysis
# =============================================================================

if __name__ == "__main__":
    print("Starting feature selection analysis...")
    
    # 1. Title feature analysis
    print("Analyzing title features...")
    lr_title, title_acc, title_auc = analyze_title_features(df)
    
    # 2. Thumbnail feature analysis (using downloaded images)
    print("Analyzing thumbnail features...")
    try:
        lr_thumbnail, thumbnail_acc, thumbnail_auc = analyze_thumbnail_features(df, sample_size=10000)
        if lr_thumbnail is None:
            thumbnail_acc, thumbnail_auc = 0, 0
    except Exception as e:
        print(f"Thumbnail analysis error: {e}")
        thumbnail_acc, thumbnail_auc = 0, 0
    
    # 3. Specific variable analysis
    print("Analyzing specific variables...")
    specific_results = analyze_specific_variables(df)
    
    # 4. Comprehensive comparison
    print("Generating comprehensive comparison...")
    comparison_results = compare_all_models(title_acc, title_auc, thumbnail_acc, thumbnail_auc, specific_results)
    
    print("\nAnalysis completed!")
