import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import shap
import warnings
warnings.filterwarnings('ignore')

from src.data_processing.preprocessor import JobFraudDataPipeline
from src.models.model_trainer import ModelTrainingPipeline

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        color: black;
    }
    .fraud-alert {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
        color: black;
    }
    .safe-alert {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load the preprocessor and trained model"""
    try:
        preprocessor = JobFraudDataPipeline.load_pipeline('data/models/preprocessor.pkl')
        with open('./data/models/final_model.pkl', "rb") as f:
            final_model = pickle.load(f)
        return preprocessor, final_model
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'preprocessor.pkl' and 'best_model.pkl' are in the same directory.")
        return None, None

def preprocess_data(df, preprocessor):
    """Preprocess the data using the trained preprocessor"""
    try:
        # Store original data for display
        original_df = df.copy()
        
        # Apply preprocessing
        processed_data = preprocessor.transform_test_data(df)
        
        # Convert to DataFrame if it's not already
        if hasattr(processed_data, 'toarray'):
            processed_data = processed_data.toarray()
        
        return processed_data, original_df
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return None, None

def create_predictions_table(df, predictions, probabilities):
    """Create a formatted predictions table"""
    results_df = df[['job_id', 'title']].copy()
    results_df['Prediction'] = ['Fraud' if p == 1 else 'Legitimate' for p in predictions]
    results_df['Fraud_Probability'] = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities
    results_df['Risk_Level'] = pd.cut(
        results_df['Fraud_Probability'],
        bins=[0, 0.3, 0.7, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    return results_df

def create_histogram(probabilities):
    """Create histogram of fraud probabilities"""
    fig = px.histogram(
        x=probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities,
        nbins=30,
        title="Distribution of Fraud Probabilities",
        labels={'x': 'Fraud Probability', 'y': 'Count'},
        color_discrete_sequence=['#1f77b4']
    )
    fig.update_layout(
        title_font_size=16,
        title_x=0.5,
        showlegend=False
    )
    return fig

def create_pie_chart(predictions):
    """Create pie chart of fraud vs legitimate predictions"""
    counts = pd.Series(predictions).value_counts()
    labels = ['Legitimate', 'Fraud']
    values = [counts.get(0, 0), counts.get(1, 0)]
    
    fig = px.pie(
        values=values,
        names=labels,
        title="Fraud vs Legitimate Predictions",
        color_discrete_sequence=['#2ecc71', '#e74c3c']
    )
    fig.update_layout(title_font_size=16, title_x=0.5)
    return fig

def create_suspicious_listings_chart(results_df, top_n=10):
    """Create chart for top suspicious listings"""
    top_suspicious = results_df.nlargest(top_n, 'Fraud_Probability')
    
    fig = px.bar(
        top_suspicious,
        x=top_suspicious.index.astype(str),
        y='Fraud_Probability',
        title=f"Top {top_n} Most Suspicious Listings",
        labels={'x': 'Listing Index', 'Fraud_Probability': 'Fraud Probability'},
        color='Fraud_Probability',
        color_continuous_scale='Reds'
    )
    fig.update_layout(
        title_font_size=16,
        title_x=0.5,
        height=400,
        xaxis_title="Listing Index",
        yaxis_title="Fraud Probability",
        xaxis={'type': 'category'}
    )
    return fig

def create_wordcloud(df, text_column=None):
    """Create word cloud from text data"""
    try:
        # Try to find text columns
        text_columns = df.select_dtypes(include=['object']).columns
        
        if text_column and text_column in df.columns:
            text_data = df[text_column].dropna().astype(str)
        elif len(text_columns) > 0:
            # Use the first text column found
            text_data = df[text_columns[0]].dropna().astype(str)
        else:
            return None
        
        # Combine all text
        text = ' '.join(text_data.values)
        
        if len(text.strip()) == 0:
            return None
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis'
        ).generate(text)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Word Cloud of Text Features', fontsize=16, pad=20)
        
        return fig
    except Exception as e:
        st.warning(f"Could not generate word cloud: {str(e)}")
        return None

def create_shap_plots(model, X_processed, feature_names=None):
    """Create SHAP explanation plots"""
    try:
        # Create SHAP explainer
        explainer = shap.Explainer(model.predict, X_processed[:50])  # Use subset for speed
        shap_values = explainer(X_processed[:50])
        
        # Summary plot
        fig_summary, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X_processed[:50],
            feature_names=feature_names,
            show=False,
        )
        ax.set_title('SHAP Feature Importance Summary', fontsize=16, pad=20)
        
        # Waterfall plot for first prediction
        fig_waterfall, ax2 = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(shap_values[0], show=False)
        ax2.set_title('SHAP Waterfall Plot - First Prediction', fontsize=16, pad=20)
        
        return fig_summary, fig_waterfall
    except Exception as e:
        st.warning(f"Could not generate SHAP plots: {str(e)}")
        return None, None

def main():
    st.markdown('<h1 class="main-header">🔍 Fraud Detection Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("Upload your CSV file and configure the analysis settings.")
    
    # Load models
    preprocessor, model = load_models()
    
    if preprocessor is None or model is None:
        st.stop()
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file containing the data you want to analyze for fraud detection."
    )
    
    if uploaded_file is not None:
        # Load data
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✅ File uploaded successfully! Shape: {df.shape}")
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            st.stop()
        
        # Data preview
        with st.expander("📊 Data Preview", expanded=False):
            st.write("**First 5 rows:**")
            st.dataframe(df.head())
            st.write("**Data Info:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
        
        # Preprocess data
        with st.spinner("🔄 Preprocessing data..."):
            X_processed, original_df = preprocess_data(df, preprocessor)
        
        if X_processed is not None:
            # Make predictions
            with st.spinner("🤖 Making predictions..."):
                predictions = model.predict(X_processed)
                probabilities = model.predict_proba(X_processed)
            
            # Create results dataframe
            results_df = create_predictions_table(df, predictions, probabilities)
            
            # Main metrics
            st.markdown("## 📈 Key Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            fraud_count = sum(predictions)
            total_count = len(predictions)
            fraud_rate = fraud_count / total_count * 100
            avg_fraud_prob = np.mean(probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Total Records</h3>
                    <h2>{total_count:,}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="fraud-alert">
                    <h3>Fraud Detected</h3>
                    <h2>{fraud_count:,}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Fraud Rate</h3>
                    <h2>{fraud_rate:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Avg Fraud Probability</h3>
                    <h2>{avg_fraud_prob:.3f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Visualizations
            st.markdown("## 📊 Analysis Results")
            
            # Row 1: Histogram and Pie Chart
            col1, col2 = st.columns(2)
            
            with col1:
                fig_hist = create_histogram(probabilities)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                fig_pie = create_pie_chart(predictions)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Row 2: Top Suspicious Listings
            st.markdown("### 🚨 Top Suspicious Listings")
            fig_suspicious = create_suspicious_listings_chart(results_df)
            st.plotly_chart(fig_suspicious, use_container_width=True)
            
            # Row 3: Results Table
            st.markdown("### 📋 Detailed Results")
            
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                risk_filter = st.selectbox("Filter by Risk Level", ["All", "High", "Medium", "Low"])
            with col2:
                prob_threshold = st.slider("Minimum Fraud Probability", 0.0, 1.0, 0.0, 0.01)
            
            # Apply filters
            filtered_df = results_df.copy()
            if risk_filter != "All":
                filtered_df = filtered_df[filtered_df['Risk_Level'] == risk_filter]
            filtered_df = filtered_df[filtered_df['Fraud_Probability'] >= prob_threshold]
            
            # Display table
            st.dataframe(
                filtered_df,
                use_container_width=True,
                height=400
            )
            
            # Advanced Analysis
            st.markdown("## 🧠 Advanced Analysis")
            
            tab1, tab2 = st.tabs(["☁️ Word Cloud", "🔍 SHAP Analysis",])
            
            with tab1:
                st.markdown("### Word Cloud of Text Features")
                wordcloud_fig = create_wordcloud(df)
                
                if wordcloud_fig is not None:
                    st.pyplot(wordcloud_fig)
                else:
                    st.info("No text features found for word cloud generation.")
            
            with tab2:
                st.markdown("### SHAP Feature Importance")
                model_type = type(model).__name__
                if hasattr(model, 'tree_') or 'Forest' in model_type or 'Tree' in model_type:
                    shap_summary, shap_waterfall = create_shap_plots(model, X_processed)
                    
                    if shap_summary is not None:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.pyplot(shap_summary)
                        with col2:
                            if shap_waterfall is not None:
                                st.pyplot(shap_waterfall)
                    else:
                        st.info("SHAP analysis not available for this model type.")
                else:
                    st.markdown("For `SVM models` with Kernel: `rbf` SHAP takes too much time.")
            
            # Download results
            st.markdown("## 💾 Download Results")
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_data,
                file_name="fraud_detection_results.csv",
                mime="text/csv"
            )
            
    else:
        # Landing page
        st.markdown("""
        ### Welcome to the Fraud Detection Dashboard! 👋
        
        This dashboard provides comprehensive fraud detection analysis with the following features:
        
        **📊 Key Features:**
        - 📁 **CSV File Upload**: Easy data import
        - 🔄 **Automated Preprocessing**: Using your trained preprocessor
        - 🤖 **Real-time Predictions**: Binary classification for fraud detection
        - 📈 **Interactive Visualizations**: Histograms, pie charts, and bar charts
        - 🚨 **Risk Assessment**: Top suspicious listings identification
        - 🧠 **Explainable AI**: SHAP plots for model interpretability
        - ☁️ **Text Analysis**: Word clouds from text features
        - 💾 **Export Results**: Download analysis results
        
        **🚀 Getting Started:**
        1. Upload your CSV file using the sidebar
        2. Review the data preview and metrics
        3. Explore the interactive visualizations
        4. Analyze suspicious listings and explanations
        5. Download your results
        
        **📋 Requirements:**
        - Your CSV file should contain the same features used to train the model
        - Ensure `preprocessor.pkl` and `best_model.pkl` are in the same directory
        """)
        
        # Sample data format info
        with st.expander("📖 Expected Data Format", expanded=False):
            st.markdown("""
            Your CSV file should contain the same columns that were used to train your model.
            The preprocessor will handle feature engineering and scaling automatically.
            """)

            df = pd.read_csv('data/example/example.csv')
            st.dataframe(df.head())

if __name__ == "__main__":
    main()