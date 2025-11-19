import matplotlib.pyplot as plt
import seaborn as sns

def set_plot_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12


def plot_feature_importance(importance_df, top_n=15):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    top_features = importance_df.head(top_n)
    bars = ax.barh(range(len(top_features)), top_features['importance'].values)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Top {top_n} Most Important Features')
    ax.invert_yaxis()
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.4f}', ha='left', va='center')
    plt.tight_layout()
    return fig


def plot_actual_vs_predicted(y_true, y_pred):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(y_true, y_pred, alpha=0.5, s=10)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Trip Count')
    ax.set_ylabel('Predicted Trip Count')
    ax.set_title('Actual vs Predicted Trip Counts')
    ax.legend()
    plt.tight_layout()
    return fig


def plot_residuals(y_true, y_pred):
    set_plot_style()
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Predicted Values')
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Residual Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Residuals')
    plt.tight_layout()
    return fig


def plot_hourly_patterns(df):
    set_plot_style()
    hourly_avg = df.groupby('hour')['trip_count'].mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2, markersize=8)
    ax.fill_between(hourly_avg.index, hourly_avg.values, alpha=0.3)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average Trip Count')
    ax.set_title('Average Bike Trips by Hour of Day')
    ax.set_xticks(range(24))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_weather_correlation(df):
    set_plot_style()
    weather_cols = ['temperature', 'humidity', 'wind_speed', 'precipitation', 'sunshine', 'trip_count']
    corr_matrix = df[weather_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True, 
                linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax, fmt='.2f')
    ax.set_title('Correlation Matrix: Weather vs Trip Count')
    plt.tight_layout()
    return fig


def plot_model_comparison(comparison_df):
    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ['RMSE', 'MAE', 'R2']
    colors = sns.color_palette("husl", len(comparison_df))
    for idx, metric in enumerate(metrics):
        bars = axes[idx].bar(comparison_df.index, comparison_df[metric], color=colors)
        axes[idx].set_title(f'Model Comparison: {metric}')
        axes[idx].set_ylabel(metric)
        axes[idx].tick_params(axis='x', rotation=45)
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.4f}', ha='center', va='bottom')
    plt.tight_layout()
    return fig


def plot_station_map(stations_df, recommendations=None, user_location=None):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(stations_df['start_station_longitude'], stations_df['start_station_latitude'], 
               c='gray', s=20, alpha=0.5, label='All Stations')
    if recommendations is not None and not recommendations.empty:
        ax.scatter(recommendations['longitude'], recommendations['latitude'],
                  c='green', s=100, marker='*', label='Recommended Stations', zorder=5)
        for _, rec in recommendations.iterrows():
            ax.annotate(f"{rec['station_name'][:20]}...", 
                       (rec['longitude'], rec['latitude']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
    if user_location:
        ax.scatter(user_location[1], user_location[0], c='red', s=150, 
                  marker='X', label='Your Location', zorder=6)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Bike Stations Map')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_seasonal_patterns(df):
    set_plot_style()
    season_map = {0: 'Spring', 1: 'Summer', 2: 'Fall', 3: 'Winter'}
    df_temp = df.copy()
    df_temp['season_name'] = df_temp['season'].map(season_map)
    seasonal_avg = df_temp.groupby('season_name')['trip_count'].mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#90EE90', '#FFD700', '#FFA500', '#87CEEB']
    bars = ax.bar(seasonal_avg.index, seasonal_avg.values, color=colors, edgecolor='black')
    ax.set_xlabel('Season')
    ax.set_ylabel('Average Trip Count')
    ax.set_title('Average Bike Trips by Season')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}', ha='center', va='bottom')
    plt.tight_layout()
    return fig