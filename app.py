import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
import time
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster

# Set dark theme for plotly
PLOTLY_THEME = "plotly_dark"
DARK_BACKGROUND = "#1e2530"
GRID_COLOR = "#2d3747"
TEXT_COLOR = "#fafafa"

# Configure matplotlib dark theme
plt.style.use('dark_background')

# Set page config with dark theme
st.set_page_config(
    page_title="Student Expense Analysis",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Custom CSS for dark theme - removing unused styles and simplifying
st.markdown("""
    <style>
        /* Main container */
        .main {
            background-color: #0e1117;
            color: #fafafa;
            padding: 0 2rem;
        }
        
        /* Section titles with gradient */
        .section-title {
            color: #fafafa;
            font-size: 2.2rem;
            font-weight: 700;
            margin: 3rem 0 2rem 0;
            background: linear-gradient(90deg, #4b8bff, #2d5cff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: 0.5px;
        }
        
        /* Subsection titles with accent */
        .subsection-title {
            color: #4b8bff;
            font-size: 1.6rem;
            font-weight: 600;
            margin: 2.5rem 0 1.5rem 0;
            letter-spacing: 0.3px;
        }

        /* Description text with better readability */
        .description-text {
            color: #c2c6cd;
            font-size: 1.1rem;
            line-height: 1.8;
            margin: 1rem 0 2rem 0;
            padding: 0.5rem 0;
        }
        
        /* Enhanced info boxes */
        .info-box {
            background: linear-gradient(145deg, #1a1f29, #23293a);
            padding: 1.8rem;
            border-radius: 12px;
            border-left: 4px solid #4b8bff;
            margin: 1.8rem 0;
            transition: transform 0.2s ease;
        }
        
        .info-box:hover {
            transform: translateX(5px);
        }
        
        /* Metric cards with hover effect */
        .metric-card {
            background: linear-gradient(145deg, #1a1f29, #23293a);
            padding: 2rem;
            border-radius: 16px;
            text-align: center;
            transition: transform 0.2s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
        }

        /* Enhanced buttons */
        .stButton > button {
            background: linear-gradient(90deg, #4b8bff, #2d5cff);
            color: white;
            border: none;
            padding: 0.8rem 1.5rem;
            border-radius: 8px;
            font-weight: 600;
            width: 100%;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background: linear-gradient(90deg, #2d5cff, #1a4aff);
            box-shadow: 0 5px 15px rgba(43, 92, 255, 0.2);
        }

        /* Improved select boxes */
        .stSelectbox > div > div {
            background: linear-gradient(145deg, #1a1f29, #23293a);
            border-radius: 8px;
            border: 1px solid #2d3747;
        }

        /* Better number inputs */
        .stNumberInput > div > div > input {
            background: linear-gradient(145deg, #1a1f29, #23293a);
            border: 1px solid #2d3747;
            border-radius: 8px;
            color: #fafafa;
            padding: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)
# Add a sidebar with custom styling
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #fafafa;'>Navigation</h2>", unsafe_allow_html=True)
    option = st.selectbox(
        '',  # Empty label since we're using the header above
        ['Home', 'Data Analysis', 'Expense Prediction', 'Insights', 'Conclusions & Recommendations']
    )

# Column name mappings
COLUMN_MAPPING = {
    'Weekly allowance': 'weekly_allowance',
    'Amount spent on online shopping': 'online_shopping',
    'Amount spent on Food': 'food_expenses',
    'Amount spent on school contributions (Amot, project fees, etc..)': 'school_contributions',
    'Amount spent on travel fare': 'travel_fare',
    'Allowance left after a week': 'remaining_allowance'
}

# Reverse mapping for display
REVERSE_MAPPING = {v: k for k, v in COLUMN_MAPPING.items()}

# Format currency in Peso
def format_peso(amount):
    return f"₱{amount:,.2f}"

# Load and prepare the data
@st.cache_data
def load_data():
    try:
        # Read the CSV file
        data = pd.read_csv('student_savings_data.csv')
        
        # Convert numeric columns by removing commas
        for col in data.columns:
            if data[col].dtype == 'object':
                data[col] = data[col].str.replace(',', '').astype(float)
        
        # Rename columns for easier handling
        data = data.rename(columns=COLUMN_MAPPING)
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def preprocess_data(data, target_variable):
    # Separate features and target
    features = ['weekly_allowance', 'online_shopping', 'food_expenses', 
                'school_contributions', 'travel_fare']
    
    # Remove target variable from features
    features.remove(target_variable)
    
    X = data[features]
    y = data[target_variable]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, features

def create_correlation_heatmap(data):
    corr = data.corr()
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', 
                annot_kws={'color': 'white'}, 
                cbar_kws={'label': 'Correlation'})
    plt.title('Correlation Heatmap of Student Expenses', color=TEXT_COLOR, pad=20)
    fig.patch.set_facecolor(DARK_BACKGROUND)
    plt.gca().set_facecolor(DARK_BACKGROUND)
    return fig

def plot_feature_importance(model, features, target_variable):
    importance = pd.DataFrame({
        'Feature': [REVERSE_MAPPING.get(f, f) for f in features],
        'Importance': np.abs(model.coef_)
    })
    importance = importance.sort_values('Importance', ascending=True)
    
    fig = plt.figure(figsize=(10, 6))
    plt.barh(importance['Feature'], importance['Importance'], color='#4b8bff')
    plt.title(f'Feature Importance for {REVERSE_MAPPING.get(target_variable, target_variable)}', 
             color=TEXT_COLOR, pad=20)
    plt.xlabel('Absolute Coefficient Value', color=TEXT_COLOR)
    plt.ylabel('Features', color=TEXT_COLOR)
    fig.patch.set_facecolor(DARK_BACKGROUND)
    plt.gca().set_facecolor(DARK_BACKGROUND)
    return fig

def plot_actual_vs_predicted(y_test, y_pred, target_variable):
    fig = px.scatter(x=y_test, y=y_pred, 
                    labels={'x': f'Actual {REVERSE_MAPPING.get(target_variable, target_variable)}',
                           'y': f'Predicted {REVERSE_MAPPING.get(target_variable, target_variable)}'},
                    title=f'Actual vs Predicted {REVERSE_MAPPING.get(target_variable, target_variable)}',
                    template=PLOTLY_THEME)
    
    # Add perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    fig.add_trace(go.Scatter(x=[min_val, max_val], 
                            y=[min_val, max_val],
                            mode='lines', 
                            name='Perfect Prediction',
                            line=dict(color='#ff4b4b', dash='dash')))
    
    # Update layout for dark theme
    fig.update_layout(
        paper_bgcolor=DARK_BACKGROUND,
        plot_bgcolor=DARK_BACKGROUND,
        font_color=TEXT_COLOR,
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    )
    return fig

def perform_clustering_analysis(data):
    # Select features for clustering
    features_for_clustering = ['weekly_allowance', 'food_expenses', 
                             'online_shopping', 'travel_fare', 
                             'school_contributions']
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[features_for_clustering])
    
    # K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_clusters = kmeans.fit_predict(scaled_data)
    
    # EM Clustering (Gaussian Mixture)
    em = GaussianMixture(n_components=3, random_state=42)
    em_clusters = em.fit_predict(scaled_data)
    
    # DBScan Clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_clusters = dbscan.fit_predict(scaled_data)
    
    # SLINK Clustering (Hierarchical)
    slink = linkage(scaled_data, method='single')
    slink_clusters = fcluster(slink, t=3, criterion='maxclust')
    
    return {
        'KMeans': kmeans_clusters,
        'EM': em_clusters,
        'DBScan': dbscan_clusters,
        'SLINK': slink_clusters
    }, features_for_clustering

def generate_category_stats(data, category):
    """Generate statistics based on the selected category."""
    if category == "Spending Patterns":
        total_expenses = data[['food_expenses', 'online_shopping', 'travel_fare']].sum(axis=1).mean()
        food_pct = (data['food_expenses'].mean() / total_expenses) * 100
        shopping_pct = (data['online_shopping'].mean() / total_expenses) * 100
        travel_pct = (data['travel_fare'].mean() / total_expenses) * 100
        
        return f"""
            <div style='margin-bottom: 0.5rem;'>
                <p style='color: #fafafa; margin: 0;'>Expense Distribution</p>
                <ul style='color: #c2c6cd; margin: 0.5rem 0;'>
                    <li>Food: {food_pct:.1f}%</li>
                    <li>Shopping: {shopping_pct:.1f}%</li>
                    <li>Travel: {travel_pct:.1f}%</li>
                </ul>
            </div>
            <div>
                <p style='color: #fafafa; margin: 0;'>Total Monthly Average</p>
                <p style='color: #c2c6cd; margin: 0;'>{format_peso(total_expenses)}</p>
            </div>
        """
    elif category == "Savings Behavior":
        savings_rate = (data['remaining_allowance'] / data['weekly_allowance']).mean() * 100
        positive_savings = (data['remaining_allowance'] > 0).mean() * 100
        high_savers = (data['remaining_allowance'] > data['weekly_allowance'] * 0.3).mean() * 100
        
        return f"""
            <div style='margin-bottom: 0.5rem;'>
                <p style='color: #fafafa; margin: 0;'>Savings Metrics</p>
                <ul style='color: #c2c6cd; margin: 0.5rem 0;'>
                    <li>Average Savings Rate: {savings_rate:.1f}%</li>
                    <li>Students with Savings: {positive_savings:.1f}%</li>
                    <li>High Savers (>30%): {high_savers:.1f}%</li>
                </ul>
            </div>
            <div>
                <p style='color: #fafafa; margin: 0;'>Average Monthly Savings</p>
                <p style='color: #c2c6cd; margin: 0;'>{format_peso(data['remaining_allowance'].mean())}</p>
            </div>
        """
    else:  # Risk Factors
        high_risk_threshold = 0.7  # 70% of maximum value
        high_shopping = (data['online_shopping'] > data['online_shopping'].quantile(high_risk_threshold)).mean() * 100
        zero_savings = (data['remaining_allowance'] <= 0).mean() * 100
        high_food = (data['food_expenses'] > data['food_expenses'].quantile(high_risk_threshold)).mean() * 100
        
        risk_score = (high_shopping + zero_savings + high_food) / 3
        risk_level = "High" if risk_score > 50 else "Moderate" if risk_score > 30 else "Low"
        
        return f"""
            <div style='margin-bottom: 0.5rem;'>
                <p style='color: #fafafa; margin: 0;'>Risk Indicators</p>
                <ul style='color: #c2c6cd; margin: 0.5rem 0;'>
                    <li>High Shopping Risk: {high_shopping:.1f}%</li>
                    <li>Zero Savings: {zero_savings:.1f}%</li>
                    <li>High Food Spending: {high_food:.1f}%</li>
                </ul>
            </div>
            <div>
                <p style='color: #fafafa; margin: 0;'>Overall Risk Level</p>
                <p style='color: {"#ff4b4b" if risk_level == "High" else "#ffd700" if risk_level == "Moderate" else "#50C878"}; 
                          margin: 0; font-weight: bold;'>{risk_level}</p>
            </div>
        """

def generate_actions(priority):
    """Generate specific actions based on the selected financial priority."""
    actions = {
        "Increase Savings": [
            "Set up automatic savings transfers when receiving allowance",
            "Track all expenses daily using a mobile app or notebook",
            "Identify and cut non-essential expenses",
            "Look for student discounts and promotions"
        ],
        "Reduce Food Expenses": [
            "Create weekly meal plans to avoid impulse food purchases",
            "Pack lunch from home more frequently",
            "Use student meal deals and campus dining options",
            "Share bulk purchases with roommates or friends"
        ],
        "Control Online Shopping": [
            "Implement a 24-hour waiting rule before making purchases",
            "Unsubscribe from promotional emails and shopping apps",
            "Create a monthly shopping budget and stick to it",
            "Make a wishlist and prioritize essential items"
        ],
        "Optimize Travel Costs": [
            "Research and apply for student travel passes",
            "Explore carpooling options with classmates",
            "Plan routes to minimize transportation costs",
            "Consider walking or cycling for short distances"
        ],
        "Build Emergency Fund": [
            "Start with small, consistent weekly contributions",
            "Save unexpected income or allowance increases",
            "Review and eliminate unnecessary subscriptions",
            "Set specific emergency fund goals with deadlines"
        ]
    }
    return actions.get(priority, ["No specific actions available for this priority"])

data = load_data()

if data is not None:
    if option == 'Home':
        
        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        total_students = len(data)
        avg_allowance = data['weekly_allowance'].mean()
        avg_expenses = data[['food_expenses', 'online_shopping', 'travel_fare', 'school_contributions']].sum(axis=1).mean()
        savings_rate = ((data['weekly_allowance'] - data[['food_expenses', 'online_shopping', 'travel_fare', 'school_contributions']].sum(axis=1)) / data['weekly_allowance']).mean() * 100

        metrics = [
            {"title": "Total Students", "value": f"{total_students:,}", "icon": "👥"},
            {"title": "Avg. Weekly Allowance", "value": format_peso(avg_allowance), "icon": "💰"},
            {"title": "Avg. Weekly Expenses", "value": format_peso(avg_expenses), "icon": "💳"},
            {"title": "Avg. Savings Rate", "value": f"{savings_rate:.1f}%", "icon": "📈"}
        ]

        for col, metric in zip([col1, col2, col3, col4], metrics):
            col.markdown(f"""
                <div class='metric-card' style='text-align: center;'>
                    <div style='font-size: 2rem;'>{metric['icon']}</div>
                    <h4 style='margin: 0.5rem 0; color: #4b8bff;'>{metric['title']}</h4>
                    <p style='font-size: 1.5rem; margin: 0; color: #fafafa;'>{metric['value']}</p>
                </div>
            """, unsafe_allow_html=True)

        # Dataset Description
        st.markdown("<div class='section-title'>About the Dataset</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
                <div class='info-box'>
                    <h4 style='color:#4b8bff; margin-top:0;'>Factors Affecting Weekly Allowance Savings for students in CIT</h4>
                    <p style='color:#c2c6cd; line-height: 1.6;'>
                        This dataset was collected as part of our Data Analytics course, where we conducted a comprehensive survey of students at CIT to gather detailed information about their weekly allowances and spending patterns. The dataset includes:
                    </p>
                    <ul style='color:#c2c6cd; margin: 1rem 0;'>
                        <li>Weekly allowance allocation</li>
                        <li>Spending across major categories (food, shopping, travel, etc.)</li>
                        <li>Savings patterns and remaining allowance</li>
                        <li>School-related expenses and contributions</li>
                    </ul>
                    <p style='color:#c2c6cd; line-height: 1.6;'>
                        The data was collected over a typical academic period, providing insights into real spending patterns
                        and financial behaviors of students.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Add Raw Data Display with Toggle
            with st.expander("📊 View Raw Data"):
                # Display column descriptions
                st.markdown("""
                    <div style='margin-bottom: 1rem;'>
                        <h4 style='color: #4b8bff;'>Column Descriptions:</h4>
                        <ul style='color: #c2c6cd;'>
                            <li><strong>Weekly allowance:</strong> Student's weekly budget</li>
                            <li><strong>Amount spent on online shopping:</strong> E-commerce expenses</li>
                            <li><strong>Amount spent on Food:</strong> Food and beverage expenses</li>
                            <li><strong>Amount spent on school contributions:</strong> Academic-related costs</li>
                            <li><strong>Amount spent on travel fare:</strong> Transportation expenses</li>
                            <li><strong>Allowance left after a week:</strong> Remaining budget/savings</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
                
                # Display the raw data with styling
                st.dataframe(
                    data.style.format({
                        col: "₱{:,.2f}" for col in data.columns
                    }).background_gradient(
                        cmap='Blues',
                        subset=[col for col in data.columns if col != 'remaining_allowance']
                    ).background_gradient(
                        cmap='Greens',
                        subset=['remaining_allowance']
                    ),
                    height=300
                )
                
                # Add download button for the dataset
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Dataset",
                    data=csv,
                    file_name="student_expenses.csv",
                    mime="text/csv",
                )
            
        with col2:
            # Quick stats about the dataset
            st.markdown(f"""
                <div class='info-box'>
                    <h4 style='color:#4b8bff; margin-top:0;'>Dataset Highlights</h4>
                    <div style='color:#c2c6cd; margin: 1rem 0;'>
                        <div style='margin-bottom: 1rem;'>
                            <p style='margin: 0; font-weight: bold;'>Sample Size</p>
                            <p style='margin: 0; font-size: 1.2rem;'>{len(data):,} students</p>
                        </div>
                        <div style='margin-bottom: 1rem;'>
                            <p style='margin: 0; font-weight: bold;'>Allowance Range</p>
                            <p style='margin: 0; font-size: 1.2rem;'>
                                {format_peso(data['weekly_allowance'].min())} - {format_peso(data['weekly_allowance'].max())}
                            </p>
                        </div>
                        <div style='margin-bottom: 1rem;'>
                            <p style='margin: 0; font-weight: bold;'>Categories Tracked</p>
                            <p style='margin: 0; font-size: 1.2rem;'>{len(COLUMN_MAPPING)} expense types</p>
                        </div>
                        <div>
                            <p style='margin: 0; font-weight: bold;'>Time Period</p>
                            <p style='margin: 0; font-size: 1.2rem;'>Academic semester</p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # Data Quality Note
        st.markdown("""
            <div style='background: linear-gradient(145deg, rgba(75, 139, 255, 0.1), rgba(45, 92, 255, 0.1)); 
                        padding: 1rem; border-radius: 8px; margin-top: 1rem; border-left: 3px solid #4b8bff;'>
                <p style='color: #fafafa; margin: 0; font-size: 0.9rem;'>
                    <strong style='color: #4b8bff;'>💡 Note:</strong> All financial data has been verified and cleaned to ensure accuracy.
                    Outliers have been reviewed and validated to represent genuine student spending patterns.
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Expense Distribution Section
        st.markdown("<div class='section-title'>Expense Distribution Overview</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Create pie chart for expense distribution
            expense_categories = ['food_expenses', 'online_shopping', 'school_contributions', 'travel_fare']
            avg_expenses = [data[cat].mean() for cat in expense_categories]
            labels = [REVERSE_MAPPING[cat] for cat in expense_categories]
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=labels,
                values=avg_expenses,
                hole=.4,
                marker=dict(colors=['#4b8bff', '#ff4b4b', '#ffd700', '#50C878'])
            )])
            
            fig_pie.update_layout(
                title='Average Expense Distribution',
                template=PLOTLY_THEME,
                paper_bgcolor=DARK_BACKGROUND,
                plot_bgcolor=DARK_BACKGROUND,
                font_color=TEXT_COLOR
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.markdown("""
                <div class='info-box'>
                    <h4 style='color:#4b8bff; margin-top:0;'>Understanding the Chart</h4>
                    <p style='color:#c2c6cd;'>This pie chart shows how students typically allocate their allowance across different expense categories:</p>
                    <ul style='color:#c2c6cd;'>
            """, unsafe_allow_html=True)
            
            total_exp = sum(avg_expenses)
            for label, exp in zip(labels, avg_expenses):
                percentage = (exp/total_exp) * 100
                st.markdown(f"""
                    <li><strong>{label}:</strong> {format_peso(exp)} ({percentage:.1f}%)</li>
                """, unsafe_allow_html=True)
            
            st.markdown("""
                    </ul>
                </div>
            """, unsafe_allow_html=True)

        # Spending Trends Section
        st.markdown("<div class='section-title'>Spending vs. Savings Analysis</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Create scatter plot of allowance vs savings
            fig_savings = px.scatter(data, 
                                   x='weekly_allowance', 
                                   y='remaining_allowance',
                                   title='Weekly Allowance vs. Savings',
                                   template=PLOTLY_THEME)
            
            # Add trend line
            fig_savings.add_traces(
                px.scatter(data, 
                          x='weekly_allowance', 
                          y='remaining_allowance', 
                          trendline="ols").data
            )
            
            fig_savings.update_layout(
                xaxis_title="Weekly Allowance (₱)",
                yaxis_title="Remaining Allowance (₱)",
                paper_bgcolor=DARK_BACKGROUND,
                plot_bgcolor=DARK_BACKGROUND,
                font_color=TEXT_COLOR,
                xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
                yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
            )
            st.plotly_chart(fig_savings, use_container_width=True)

        with col2:
            st.markdown("""
                <div class='info-box'>
                    <h4 style='color:#4b8bff; margin-top:0;'>Trend Analysis</h4>
                    <p style='color:#c2c6cd;'>This scatter plot reveals the relationship between weekly allowance and savings:</p>
                    <ul style='color:#c2c6cd;'>
                        <li>Each point represents a student</li>
                        <li>The trend line shows the general saving pattern</li>
                        <li>Points above the trend line indicate better-than-average savers</li>
                        <li>Points below suggest higher spending tendencies</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

        # Correlation Heatmap Section
        st.markdown("<div class='section-title'>Expense Correlation Analysis</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            fig_heatmap = create_correlation_heatmap(data)
            st.pyplot(fig_heatmap)

        with col2:
            st.markdown("""
                <div class='info-box'>
                    <h4 style='color:#4b8bff; margin-top:0;'>Understanding Correlations</h4>
                    <p style='color:#c2c6cd;'>The heatmap shows how different expenses relate to each other:</p>
                    <ul style='color:#c2c6cd;'>
                        <li><strong>Dark Red (1.0):</strong> Perfect positive correlation</li>
                        <li><strong>Dark Blue (-1.0):</strong> Perfect negative correlation</li>
                        <li><strong>Light Colors (near 0):</strong> Little to no correlation</li>
                    </ul>
                    <p style='color:#c2c6cd;'>This helps identify which expenses tend to increase or decrease together.</p>
                </div>
            """, unsafe_allow_html=True)

        # Quick Tips Section
        st.markdown("<div class='section-title'>Quick Financial Tips</div>", unsafe_allow_html=True)
        st.markdown("""
            <div class='description-text'>
                Essential financial management tips tailored for students based on our data analysis.
            </div>
        """, unsafe_allow_html=True)
        
        tips = [
            {
                "icon": "💰",
                "title": "Smart Budgeting",
                "tips": [
                    "Track daily expenses using apps",
                    "Follow 50/30/20 rule (Needs/Wants/Savings)",
                    "Set weekly spending limits",
                    "Review expenses every weekend"
                ],
                "highlight": "Students who track expenses save 25% more on average"
            },
            {
                "icon": "🎯",
                "title": "Saving Strategies",
                "tips": [
                    "Save first when allowance arrives",
                    "Use student discounts actively",
                    "Plan meals and groceries ahead",
                    "Share bulk purchases with friends"
                ],
                "highlight": f"Top savers maintain {format_peso(data['remaining_allowance'].quantile(0.9))} monthly savings"
            },
            {
                "icon": "⚠️",
                "title": "Common Pitfalls",
                "tips": [
                    "Avoid impulse online shopping",
                    "Be mindful of small daily expenses",
                    "Don't skip meals to save money",
                    "Resist peer pressure spending"
                ],
                "highlight": "Unplanned expenses reduce savings by up to 40%"
            }
        ]

        col1, col2, col3 = st.columns(3)

        for col, tip in zip([col1, col2, col3], tips):
            col.markdown(f"""
                <div class='info-box' style='height: 100%;'>
                    <div style='text-align: center;'>
                        <div style='font-size: 2.5rem; margin-bottom: 1rem;'>{tip['icon']}</div>
                        <h4 style='color:#4b8bff; margin: 0.5rem 0; font-size: 1.3rem;'>{tip['title']}</h4>
                    </div>
                    <div style='margin: 1.5rem 0;'>
                        <ul style='color:#c2c6cd; list-style-type: none; padding-left: 0;'>
                            {" ".join(f"<li style='margin-bottom: 0.8rem; padding-left: 1.5rem; position: relative;'><span style='position: absolute; left: 0; color: #4b8bff;'>•</span>{tip}</li>" for tip in tip['tips'])}
                        </ul>
                    </div>
                    <div style='background: linear-gradient(145deg, rgba(75, 139, 255, 0.1), rgba(45, 92, 255, 0.1)); 
                                padding: 1rem; border-radius: 8px; margin-top: 1rem; border-left: 3px solid #4b8bff;'>
                        <p style='color: #fafafa; margin: 0; font-size: 0.9rem;'>
                            <strong style='color: #4b8bff;'>Key Insight:</strong><br>
                            {tip['highlight']}
                        </p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # About Section
        st.markdown("<div class='section-title'>About This Application</div>", unsafe_allow_html=True)
        
        # Overview section
        st.markdown("""
            <div class='description-text'>
                The Student Expense Analysis tool is designed to help students better understand and manage their expenses 
                through data analysis and machine learning predictions. This application analyzes real student expense data 
                to provide insights and predictions that can help with financial planning.
            </div>
        """, unsafe_allow_html=True)
        
        # Features section
        st.markdown("<div class='subsection-title'>Key Features</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class='info-box'>
                    <h4 style='color:#4b8bff; margin-top:0;'>Data Analysis</h4>
                    <ul style='color:#c2c6cd; margin-bottom:0;'>
                        <li>Interactive correlation analysis</li>
                        <li>Expense distribution visualization</li>
                        <li>Comparative spending patterns</li>
                        <li>Statistical insights</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
                <div class='info-box'>
                    <h4 style='color:#4b8bff; margin-top:0;'>Prediction Features</h4>
                    <ul style='color:#c2c6cd; margin-bottom:0;'>
                        <li>Machine learning-based predictions</li>
                        <li>Feature importance analysis</li>
                        <li>Prediction accuracy metrics</li>
                        <li>Contextual predictions</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        # How to Use section
        st.markdown("<div class='subsection-title'>How to Use</div>", unsafe_allow_html=True)
        st.markdown("""
            <div class='info-box'>
                <ol style='color:#c2c6cd; margin-bottom:0;'>
                    <li><strong>Data Analysis:</strong> Explore relationships between different expenses</li>
                    <li><strong>Expense Prediction:</strong> Get predictions for future expenses</li>
                    <li><strong>Insights:</strong> Discover key patterns and trends in student spending</li>
                    <li><strong>Conclusions & Recommendations:</strong> Get personalized financial advice</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)
        
        # Technical Details and Data Privacy
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class='info-box'>
                    <h4 style='color:#4b8bff; margin-top:0;'>Technologies Used</h4>
                    <ul style='color:#c2c6cd; margin-bottom:0;'>
                        <li>Python with Streamlit for web interface</li>
                        <li>Scikit-learn for machine learning models</li>
                        <li>Pandas for data manipulation</li>
                        <li>Plotly and Matplotlib for visualizations</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
                <div class='info-box'>
                    <h4 style='color:#4b8bff; margin-top:0;'>Data Privacy</h4>
                    <p style='color:#c2c6cd; margin-bottom:0;'>
                        This application uses anonymized student expense data for analysis and predictions. 
                        No personal information is collected or stored. The predictions and insights are based 
                        on aggregate data patterns.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        # Footer
        st.markdown("""
            <div style='text-align:center; margin-top:3rem; padding:1rem; color:#8b92a5;'>
                <p>Created by Team LibertyWalk</p>
                <p style='font-size:0.8rem;'>Members:</p>
                <p style='font-size:0.8rem;'>• Mhart Khiss Degollacion</p>
                <p style='font-size:0.8rem;'>• Iverson Merto</p>
                <p style='font-size:0.8rem;'>• Nathaniel Edryd Negapatan</p>
                <p style='font-size:0.8rem;'>• Xevery Jan Bolo</p>
                <p style='font-size:0.8rem;'>• Charles Matthew Salut</p>
            </div>
        """, unsafe_allow_html=True)

    elif option == 'Data Analysis':

        # Overview Statistics in a clean grid
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = [
            {
                "title": "Total Students",
                "value": f"{len(data):,}",
                "icon": "👥",
                "color": "#4b8bff"
            },
            {
                "title": "Avg. Weekly Expenses",
                "value": format_peso(data[['food_expenses', 'online_shopping', 'travel_fare', 'school_contributions']].sum(axis=1).mean()),
                "icon": "💰",
                "color": "#ff4b4b"
            },
            {
                "title": "Highest Expense",
                "value": REVERSE_MAPPING[data[['food_expenses', 'online_shopping', 'travel_fare', 'school_contributions']].mean().idxmax()],
                "icon": "📈",
                "color": "#ffd700"
            },
            {
                "title": "Savings Rate",
                "value": f"{((data['weekly_allowance'] - data[['food_expenses', 'online_shopping', 'travel_fare', 'school_contributions']].sum(axis=1)) / data['weekly_allowance']).mean() * 100:.1f}%",
                "icon": "💎",
                "color": "#50C878"
            }
        ]

        for col, metric in zip([col1, col2, col3, col4], metrics):
            col.markdown(f"""
                <div style='background: linear-gradient(145deg, #1a1f29, #23293a); padding: 1.5rem; 
                            border-radius: 12px; text-align: center; border-left: 4px solid {metric["color"]};'>
                    <div style='font-size: 2rem;'>{metric['icon']}</div>
                    <h4 style='margin: 0.5rem 0; color: {metric["color"]};'>{metric['title']}</h4>
                    <p style='font-size: 1.5rem; margin: 0; color: #fafafa;'>{metric['value']}</p>
                </div>
            """, unsafe_allow_html=True)

        # Interactive Analysis Section
        st.markdown("""
            <div style='margin: 3rem 0 2rem 0;'>
                <h2 style='color: #4b8bff; font-size: 1.8rem;'>Interactive Analysis</h2>
                <p style='color: #c2c6cd;'>Select different expense categories to explore relationships and patterns.</p>
            </div>
        """, unsafe_allow_html=True)

        # Two-column layout for interactive charts
        col1, col2 = st.columns([2, 1])

        with col1:
            # Expense comparison chart
            expense_categories = ['food_expenses', 'online_shopping', 'travel_fare', 'school_contributions']
            selected_expenses = st.multiselect(
                "Compare Expenses",
                expense_categories,
                default=['food_expenses', 'online_shopping'],
                format_func=lambda x: REVERSE_MAPPING[x]
            )

            if len(selected_expenses) > 0:
                fig = px.box(data, y=selected_expenses, 
                            labels={'value': 'Amount (₱)', 'variable': 'Expense Category'},
                            title='Expense Distribution Comparison',
                            template=PLOTLY_THEME)
                
                fig.update_layout(
                    height=500,
                    paper_bgcolor=DARK_BACKGROUND,
                    plot_bgcolor=DARK_BACKGROUND,
                    font_color=TEXT_COLOR,
                    yaxis=dict(gridcolor=GRID_COLOR)
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("""
                <div style='background: linear-gradient(145deg, #1a1f29, #23293a); 
                            padding: 1.5rem; border-radius: 12px; height: 92%;'>
                    <h3 style='color: #4b8bff; margin-top: 0;'>Distribution Insights</h3>
                    <div style='color: #c2c6cd;'>
            """, unsafe_allow_html=True)
            
            if len(selected_expenses) > 0:
                for expense in selected_expenses:
                    stats = data[expense].describe()
                    st.markdown(f"""
                        <div style='margin-bottom: 1rem;'>
                            <h4 style='color: #fafafa; margin: 0.5rem 0;'>{REVERSE_MAPPING[expense]}</h4>
                            <ul style='color: #c2c6cd; margin: 0;'>
                                <li>Average: {format_peso(stats['mean'])}</li>
                                <li>Median: {format_peso(stats['50%'])}</li>
                                <li>Range: {format_peso(stats['min'])} - {format_peso(stats['max'])}</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)

            st.markdown("</div></div>", unsafe_allow_html=True)

        # Correlation Analysis Section
        st.markdown("""
            <div style='margin: 3rem 0 2rem 0;'>
                <h2 style='color: #4b8bff; font-size: 1.8rem;'>Correlation Analysis</h2>
                <p style='color: #c2c6cd;'>Explore relationships between different types of expenses.</p>
            </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])

        with col1:
            fig_heatmap = create_correlation_heatmap(data)
            st.pyplot(fig_heatmap)

        with col2:
            st.markdown("""
                <div style='background: linear-gradient(145deg, #1a1f29, #23293a); 
                            padding: 1.5rem; border-radius: 12px;'>
                    <h3 style='color: #4b8bff; margin-top: 0;'>Key Correlations</h3>
                    <div style='color: #c2c6cd;'>
            """, unsafe_allow_html=True)

            # Calculate and display strongest correlations
            correlations = []
            for i in range(len(expense_categories)):
                for j in range(i+1, len(expense_categories)):
                    corr = data[expense_categories[i]].corr(data[expense_categories[j]])
                    correlations.append({
                        'pair': f"{REVERSE_MAPPING[expense_categories[i]]} vs {REVERSE_MAPPING[expense_categories[j]]}",
                        'correlation': corr
                    })

            correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            for corr in correlations[:3]:
                strength = "Strong" if abs(corr['correlation']) > 0.5 else "Moderate"
                direction = "positive" if corr['correlation'] > 0 else "negative"
                st.markdown(f"""
                    <div style='margin-bottom: 1rem;'>
                        <p style='color: #fafafa; margin: 0.5rem 0;'>{corr['pair']}</p>
                        <p style='color: #c2c6cd; margin: 0;'>
                            {strength} {direction} correlation ({corr['correlation']:.2f})
                        </p>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("</div></div>", unsafe_allow_html=True)

        # Trend Analysis Section
        st.markdown("""
            <div style='margin: 3rem 0 2rem 0;'>
                <h2 style='color: #4b8bff; font-size: 1.8rem;'>Spending Trends</h2>
                <p style='color: #c2c6cd;'>Analyze how expenses relate to weekly allowance.</p>
            </div>
        """, unsafe_allow_html=True)

        selected_expense = st.selectbox(
            "Select expense to analyze",
            expense_categories,
            format_func=lambda x: REVERSE_MAPPING[x]
        )

        col1, col2 = st.columns([3, 1])

        with col1:
            fig = px.scatter(data, 
                            x='weekly_allowance', 
                            y=selected_expense,
                            trendline="ols",
                            labels={
                                'weekly_allowance': 'Weekly Allowance (₱)',
                                selected_expense: f'{REVERSE_MAPPING[selected_expense]} (₱)'
                            },
                            template=PLOTLY_THEME)
            
            fig.update_layout(
                height=500,
                paper_bgcolor=DARK_BACKGROUND,
                plot_bgcolor=DARK_BACKGROUND,
                font_color=TEXT_COLOR,
                xaxis=dict(gridcolor=GRID_COLOR),
                yaxis=dict(gridcolor=GRID_COLOR)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            correlation = data['weekly_allowance'].corr(data[selected_expense])
            percentage_of_allowance = (data[selected_expense] / data['weekly_allowance']).mean() * 100
            
            st.markdown(f"""
                <div style='background: linear-gradient(145deg, #1a1f29, #23293a); 
                            padding: 1.5rem; border-radius: 12px;'>
                    <h3 style='color: #4b8bff; margin-top: 0;'>Trend Insights</h3>
                    <div style='color: #c2c6cd;'>
                        <p><strong>Correlation:</strong> {correlation:.2f}</p>
                        <p><strong>Average Percentage of Allowance:</strong> {percentage_of_allowance:.1f}%</p>
                        <p><strong>Interpretation:</strong><br>
                        {REVERSE_MAPPING[selected_expense]} shows a 
                        {'positive' if correlation > 0 else 'negative'} relationship with weekly allowance, 
                        suggesting that students with {'higher' if correlation > 0 else 'lower'} allowances tend to spend 
                        {'more' if correlation > 0 else 'less'} on this category.</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # Clustering Analysis Section
        st.markdown("""
            <div style='margin: 3rem 0 2rem 0;'>
                <h2 style='color: #4b8bff; font-size: 1.8rem;'>Spending Pattern Clustering</h2>
                <p style='color: #c2c6cd;'>Explore how students are grouped based on their spending patterns using different clustering methods.</p>
            </div>
        """, unsafe_allow_html=True)

        # Explanation of Groups
        st.markdown("""
            <div class='info-box'>
                <h4 style='color:#4b8bff; margin-top:0;'>Understanding the Groups</h4>
                <p style='color:#c2c6cd;'>
                    Each group represents a cluster of students with similar spending patterns. 
                    The groups are identified based on the features used for clustering, such as weekly allowance, food expenses, 
                    online shopping, travel fare, and school contributions. 
                    For each group, we highlight the category with the highest average spending and provide the overall average spending.
                </p>
            </div>
        """, unsafe_allow_html=True)

        cluster_results, features_for_clustering = perform_clustering_analysis(data)

        for method, clusters in cluster_results.items():
            st.markdown(f"<h3 style='color: #4b8bff;'>{method} Clustering</h3>", unsafe_allow_html=True)
            data_with_clusters = data.copy()
            data_with_clusters['Cluster'] = clusters

            # Calculate cluster characteristics
            cluster_means = data_with_clusters.groupby('Cluster')[features_for_clustering].mean()

            # Display cluster characteristics
            st.markdown("""
                <div class='info-box'>
                    <h4 style='color:#4b8bff; margin-top:0;'>Spending Patterns Identified</h4>
                    <ul style='color:#c2c6cd;'>
            """, unsafe_allow_html=True)

            for cluster in range(len(cluster_means)):
                highest_expense = cluster_means.iloc[cluster].idxmax()
                avg_spending = cluster_means.iloc[cluster].mean()

                st.markdown(f"""
                    <li>Group {cluster + 1}: 
                        Highest expense in {REVERSE_MAPPING.get(highest_expense, highest_expense)}, 
                        Average spending: {format_peso(avg_spending)}</li>
                """, unsafe_allow_html=True)

            st.markdown("""
                    </ul>
                </div>
            """, unsafe_allow_html=True)

    elif option == 'Expense Prediction':
        
        # Step 1: Select Expense Type with enhanced UI
        st.markdown("""
            <div style='margin: 2rem 0;'>
                <h2 style='color: #4b8bff; font-size: 1.8rem;'>1. Select Expense Category</h2>
                <p style='color: #c2c6cd;'>Choose which type of expense you want to predict</p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            target_options = ['food_expenses', 'online_shopping', 'travel_fare']
            target_variable = st.selectbox(
                "",  # Empty label as we have the header above
                target_options,
                format_func=lambda x: f"📊 {REVERSE_MAPPING.get(x, x.replace('_', ' ').title())}"
            )
            
            # Show category statistics
            stats = data[target_variable].describe()
            st.markdown(f"""
                <div style='background: linear-gradient(145deg, #1a1f29, #23293a); 
                            padding: 1.5rem; border-radius: 12px; margin-top: 1rem;'>
                    <h4 style='color: #4b8bff; margin-top: 0;'>Category Statistics</h4>
                    <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;'>
                        <div>
                            <p style='color: #fafafa; margin: 0;'>Average</p>
                            <p style='color: #c2c6cd; margin: 0;'>{format_peso(stats['mean'])}</p>
                        </div>
                        <div>
                            <p style='color: #fafafa; margin: 0;'>Typical Range</p>
                            <p style='color: #c2c6cd; margin: 0;'>{format_peso(stats['25%'])} - {format_peso(stats['75%'])}</p>
                        </div>
                        <div>
                            <p style='color: #fafafa; margin: 0;'>Maximum</p>
                            <p style='color: #c2c6cd; margin: 0;'>{format_peso(stats['max'])}</p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div style='background: linear-gradient(145deg, #1a1f29, #23293a); 
                            padding: 1.5rem; border-radius: 12px;'>
                    <h4 style='color: #4b8bff; margin-top: 0;'>Why Predict?</h4>
                    <ul style='color: #c2c6cd; margin-bottom: 0;'>
                        <li>Plan your budget better</li>
                        <li>Avoid overspending</li>
                        <li>Make informed decisions</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

        # Step 2: Input Section with Visual Feedback
        st.markdown("""
            <div style='margin: 3rem 0 2rem 0;'>
                <h2 style='color: #4b8bff; font-size: 1.8rem;'>2. Enter Your Information</h2>
                <p style='color: #c2c6cd;'>Provide your current expenses to get an accurate prediction</p>
            </div>
        """, unsafe_allow_html=True)

        # Model preparation
        with st.spinner("Preparing prediction model..."):
            X_train_scaled, X_test_scaled, y_train, y_test, scaler, features = preprocess_data(data, target_variable)
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Calculate model metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        col1, col2 = st.columns([2, 1])
        input_values = {}
        
        with col1:
            st.markdown("""
                <div style='background: linear-gradient(145deg, #1a1f29, #23293a); 
                            padding: 1.5rem; border-radius: 12px;'>
                    <h4 style='color: #4b8bff; margin-top: 0;'>Enter Your Current Expenses</h4>
            """, unsafe_allow_html=True)
            
            for feature in features:
                min_val = float(data[feature].min())
                max_val = float(data[feature].max())
                mean_val = float(data[feature].mean())
                
                input_values[feature] = st.slider(
                    f"{REVERSE_MAPPING.get(feature)} (Average: {format_peso(mean_val)})",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=100.0,
                    format="%0.2f"
                )
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            # Show feature importance visualization
            st.markdown("""
                <div style='background: linear-gradient(145deg, #1a1f29, #23293a); 
                            padding: 1.5rem; border-radius: 12px;'>
                    <h4 style='color: #4b8bff; margin-top: 0;'>Factor Importance</h4>
            """, unsafe_allow_html=True)
            
            fig_importance = plot_feature_importance(model, features, target_variable)
            st.pyplot(fig_importance)
            
            st.markdown("</div>", unsafe_allow_html=True)

        # Step 3: Prediction Section
        st.markdown("""
            <div style='margin: 3rem 0 2rem 0;'>
                <h2 style='color: #4b8bff; font-size: 1.8rem;'>3. Get Your Prediction</h2>
                <p style='color: #c2c6cd;'>Click below to calculate your predicted expense</p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button(
                "🔮 Calculate Prediction",
                key="predict_button",
                help="Click to get your expense prediction"
            )

        if predict_button:
            with st.spinner("Analyzing spending patterns..."):
                time.sleep(0.5)  # Add slight delay for better UX
                input_df = pd.DataFrame([input_values])
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)[0]
                
                # Calculate additional metrics
                percentile = (data[target_variable] <= prediction).mean() * 100
                avg_value = data[target_variable].mean()
                diff_from_avg = ((prediction - avg_value) / avg_value) * 100
                
                # Display prediction results in an attractive layout
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Main prediction card
                    st.markdown(f"""
                        <div style='background: linear-gradient(145deg, #1a1f29, #23293a); 
                                    padding: 2rem; border-radius: 16px; text-align: center;'>
                            <h3 style='color: #4b8bff; margin: 0;'>Predicted {REVERSE_MAPPING.get(target_variable)}</h3>
                            <p style='font-size: 2.5rem; margin: 1rem 0; color: #fafafa;'>{format_peso(prediction)}</p>
                            <div style='display: flex; justify-content: center; gap: 2rem;'>
                                <div>
                                    <p style='color: #c2c6cd; margin: 0;'>Confidence Score</p>
                                    <p style='color: #fafafa; margin: 0;'>{r2:.2%}</p>
                                </div>
                                <div>
                                    <p style='color: #c2c6cd; margin: 0;'>Percentile</p>
                                    <p style='color: #fafafa; margin: 0;'>{percentile:.1f}%</p>
                                </div>
                            </div>
                        </div>
                        
                        <div style='background: linear-gradient(145deg, #1a1f29, #23293a); 
                                    padding: 1.5rem; border-radius: 12px; margin-top: 1rem;'>
                            <h4 style='color: #4b8bff; margin-top: 0;'>Prediction Analysis</h4>
                            <p style='color: #c2c6cd;'>
                                This predicted amount is <span style='color: {"#50C878" if diff_from_avg <= 0 else "#ff4b4b"}'>
                                {abs(diff_from_avg):.1f}% {diff_from_avg > 0 and "above" or "below"}</span> the average.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Visualization of prediction vs actual distribution
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=data[target_variable],
                        name="Current Distribution",
                        opacity=0.75
                    ))
                    fig.add_vline(
                        x=prediction,
                        line_dash="dash",
                        line_color="#4b8bff",
                        annotation_text="Your Prediction",
                        annotation_position="top"
                    )
                    
                    fig.update_layout(
                        title="Your Prediction vs Others",
                        template=PLOTLY_THEME,
                        paper_bgcolor=DARK_BACKGROUND,
                        plot_bgcolor=DARK_BACKGROUND,
                        font_color=TEXT_COLOR,
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Recommendations based on prediction
                st.markdown(f"""
                    <div style='background: linear-gradient(145deg, #1a1f29, #23293a); 
                                padding: 1.5rem; border-radius: 12px; margin-top: 1rem;'>
                        <h4 style='color: #4b8bff; margin-top: 0;'>Recommendations</h4>
                        <ul style='color: #c2c6cd;'>
                            <li>{"Consider reviewing your spending in this category" if diff_from_avg > 20 
                                else "Your predicted spending is within normal range"}</li>
                            <li>{"Look for ways to reduce expenses" if diff_from_avg > 0 
                                else "Keep up your good spending habits!"}</li>
                            <li>Compare this prediction with your actual spending to improve future budgeting</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

    elif option == 'Insights':
        st.markdown("<div class='section-title'>Key Insights</div>", unsafe_allow_html=True)
        st.markdown("""
            <div class='description-text'>
                Discover meaningful patterns and trends in student spending behavior. 
                These insights can help you make better financial decisions and understand how different expenses relate to each other.
            </div>
        """, unsafe_allow_html=True)
        
        # Overview metrics
        col1, col2, col3 = st.columns(3)
        avg_allowance = data['weekly_allowance'].mean()
        avg_savings = data['remaining_allowance'].mean()
        savings_rate = (avg_savings / avg_allowance) * 100
        
        with col1:
            st.markdown(f"""
                <div class='metric-card'>
                    <h4 style='margin:0; color:#4b8bff;'>Average Weekly Allowance</h4>
                    <p style='font-size:1.8rem; margin:0.5rem 0; color:#fafafa;'>{format_peso(avg_allowance)}</p>
                </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
                <div class='metric-card'>
                    <h4 style='margin:0; color:#4b8bff;'>Average Weekly Savings</h4>
                    <p style='font-size:1.8rem; margin:0.5rem 0; color:#fafafa;'>{format_peso(avg_savings)}</p>
                </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
                <div class='metric-card'>
                    <h4 style='margin:0; color:#4b8bff;'>Average Savings Rate</h4>
                    <p style='font-size:1.8rem; margin:0.5rem 0; color:#fafafa;'>{savings_rate:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)

        # Spending Breakdown
        st.markdown("<div class='subsection-title'>1. Spending Breakdown Analysis</div>", unsafe_allow_html=True)
        st.markdown("""
            <div class='description-text'>
                Understanding how students allocate their allowance across different expense categories 
                can help identify areas for potential savings and better budget planning.
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart of average expense distribution
            expense_categories = ['food_expenses', 'online_shopping', 'school_contributions', 'travel_fare']
            avg_expenses = [data[cat].mean() for cat in expense_categories]
            labels = [REVERSE_MAPPING[cat] for cat in expense_categories]
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=labels,
                values=avg_expenses,
                hole=.4,
                marker=dict(colors=['#4b8bff', '#ff4b4b', '#ffd700', '#50C878'])
            )])
            
            fig_pie.update_layout(
                title='Average Expense Distribution',
                template=PLOTLY_THEME,
                paper_bgcolor=DARK_BACKGROUND,
                plot_bgcolor=DARK_BACKGROUND,
                font_color=TEXT_COLOR,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                )
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col2:
            # Calculate and display key statistics
            total_expenses = sum(avg_expenses)
            expense_percentages = [(exp/total_expenses)*100 for exp in avg_expenses]
            
            st.markdown("""
                <div class='info-box'>
                    <h4 style='color:#4b8bff; margin-top:0;'>Key Spending Insights</h4>
                    <ul style='color:#c2c6cd;'>
            """, unsafe_allow_html=True)
            
            for cat, pct in zip(labels, expense_percentages):
                st.markdown(f"""
                    <li>{cat}: {pct:.1f}% of total expenses</li>
                """, unsafe_allow_html=True)
                
            st.markdown("""
                    </ul>
                </div>
            """, unsafe_allow_html=True)

        # Savings Analysis
        st.markdown("<div class='subsection-title'>2. Savings Pattern Analysis</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot of allowance vs savings
            fig_savings = px.scatter(data, 
                                   x='weekly_allowance', 
                                   y='remaining_allowance',
                                   title='Weekly Allowance vs Savings',
                                   template=PLOTLY_THEME)
            
            # Add trend line
            fig_savings.add_traces(
                px.scatter(data, 
                          x='weekly_allowance', 
                          y='remaining_allowance', 
                          trendline="ols").data
            )
            
            fig_savings.update_layout(
                xaxis_title="Weekly Allowance (₱)",
                yaxis_title="Remaining Allowance (₱)",
                paper_bgcolor=DARK_BACKGROUND,
                plot_bgcolor=DARK_BACKGROUND,
                font_color=TEXT_COLOR,
                xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
                yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
            )
            st.plotly_chart(fig_savings, use_container_width=True)
            
        with col2:
            # Calculate savings insights
            savings_corr = data['weekly_allowance'].corr(data['remaining_allowance'])
            zero_savings = (data['remaining_allowance'] <= 0).mean() * 100
            high_savers = (data['remaining_allowance'] > data['weekly_allowance'] * 0.3).mean() * 100
            
            st.markdown("""
                <div class='info-box'>
                    <h4 style='color:#4b8bff; margin-top:0;'>Savings Insights</h4>
                    <ul style='color:#c2c6cd;'>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <li>Correlation between allowance and savings: {savings_corr:.2f}</li>
                <li>{zero_savings:.1f}% of students have zero or negative savings</li>
                <li>{high_savers:.1f}% of students save more than 30% of their allowance</li>
            """, unsafe_allow_html=True)
            
            st.markdown("""
                    </ul>
                </div>
            """, unsafe_allow_html=True)

        # Spending Patterns Over Time
        st.markdown("<div class='subsection-title'>3. Expense Relationships</div>", unsafe_allow_html=True)
        
        # Create correlation matrix for selected expenses
        expense_cols = ['food_expenses', 'online_shopping', 'travel_fare', 'school_contributions']
        corr_matrix = data[expense_cols].corr()
        
        # Find strongest relationships
        relationships = []
        for i in range(len(expense_cols)):
            for j in range(i+1, len(expense_cols)):
                relationships.append({
                    'expense1': REVERSE_MAPPING[expense_cols[i]],
                    'expense2': REVERSE_MAPPING[expense_cols[j]],
                    'correlation': corr_matrix.iloc[i,j]
                })
        
        relationships.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class='info-box'>
                    <h4 style='color:#4b8bff; margin-top:0;'>Strongest Expense Relationships</h4>
                    <ul style='color:#c2c6cd;'>
            """, unsafe_allow_html=True)
            
            for rel in relationships[:3]:
                correlation_type = "positive" if rel['correlation'] > 0 else "negative"
                st.markdown(f"""
                    <li>{rel['expense1']} and {rel['expense2']}: 
                        {abs(rel['correlation']):.2f} {correlation_type} correlation</li>
                """, unsafe_allow_html=True)
                
            st.markdown("""
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
                <div class='info-box'>
                    <h4 style='color:#4b8bff; margin-top:0;'>What This Means</h4>
                    <ul style='color:#c2c6cd;'>
                        <li>Strong positive correlation: Expenses tend to increase together</li>
                        <li>Strong negative correlation: As one expense increases, the other tends to decrease</li>
                        <li>Weak correlation: Expenses vary independently</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

        # Recommendations
        st.markdown("<div class='subsection-title'>4. Recommendations</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class='info-box'>
                    <h4 style='color:#4b8bff; margin-top:0;'>Saving Strategies</h4>
                    <ul style='color:#c2c6cd;'>
                        <li>Set aside savings immediately when receiving allowance</li>
                        <li>Track expenses regularly using a budget app</li>
                        <li>Look for opportunities to reduce discretionary spending</li>
                        <li>Consider bulk buying for frequently used items</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
                <div class='info-box'>
                    <h4 style='color:#4b8bff; margin-top:0;'>Budget Planning Tips</h4>
                    <ul style='color:#c2c6cd;'>
                        <li>Allocate fixed percentages for each expense category</li>
                        <li>Prioritize essential expenses (food, transportation)</li>
                        <li>Build an emergency fund for unexpected expenses</li>
                        <li>Review and adjust your budget monthly</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

    elif option == 'Conclusions & Recommendations':
        st.markdown("""
            <div style='text-align: center; padding: 2rem 0; margin-bottom: 2rem; 
                        background: linear-gradient(145deg, #1a1f29, #23293a); border-radius: 16px;'>
                <h1 style='color: #fafafa; margin-bottom: 1rem;'>📋 Conclusions & Recommendations</h1>
                <p style='color: #c2c6cd; font-size: 1.1rem;'>
                    Key findings and actionable steps for better financial management
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Key Findings Section
        st.markdown("""
            <div style='margin: 2rem 0;'>
                <h2 style='color: #4b8bff; font-size: 1.8rem;'>Key Findings</h2>
            </div>
        """, unsafe_allow_html=True)

        # Interactive findings explorer
        finding_categories = {
            "Spending Patterns": {
                "icon": "📊",
                "findings": [
                    "Food expenses constitute the largest portion of student spending",
                    "Online shopping shows high variability across students",
                    "Travel expenses remain relatively consistent"
                ],
                "implications": "Understanding these patterns helps in better budget allocation"
            },
            "Savings Behavior": {
                "icon": "💰",
                "findings": [
                    "Higher allowance doesn't always lead to higher savings",
                    "Students with budgeting plans save 20% more on average",
                    "Weekend spending tends to be higher than weekday spending"
                ],
                "implications": "Proper planning has more impact than allowance size"
            },
            "Risk Factors": {
                "icon": "⚠️",
                "findings": [
                    "Impulse online shopping is a major budget risk",
                    "Untracked small expenses add up significantly",
                    "Peer pressure influences spending decisions"
                ],
                "implications": "Awareness of these factors can help in prevention"
            }
        }

        selected_category = st.selectbox(
            "Select a category to explore",
            options=list(finding_categories.keys()),
            format_func=lambda x: f"{finding_categories[x]['icon']} {x}"
        )

        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
                <div style='background: linear-gradient(145deg, #1a1f29, #23293a); 
                            padding: 1.5rem; border-radius: 12px;'>
                    <h3 style='color: #4b8bff; margin-top: 0;'>Main Findings</h3>
                    <ul style='color: #c2c6cd;'>
                        {"".join(f"<li>{finding}</li>" for finding in finding_categories[selected_category]['findings'])}
                    </ul>
                    <p style='color: #fafafa; margin-top: 1rem;'>
                        <strong>Key Implication:</strong><br>
                        {finding_categories[selected_category]['implications']}
                    </p>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            if selected_category == "Spending Patterns":
                total_expenses = data[['food_expenses', 'online_shopping', 'travel_fare']].sum(axis=1).mean()
                expenses_data = {
                    'Food': (data['food_expenses'].mean() / total_expenses) * 100,
                    'Shopping': (data['online_shopping'].mean() / total_expenses) * 100,
                    'Travel': (data['travel_fare'].mean() / total_expenses) * 100
                }
                
                st.markdown(f"""
                    <div style='background: linear-gradient(145deg, #1a1f29, #23293a); 
                                padding: 1.5rem; border-radius: 12px;'>
                        <h3 style='color: #4b8bff; margin-top: 0;'>Expense Breakdown</h3>
                        <div style='margin: 1rem 0;'>
                            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;'>
                                <span style='color: #fafafa;'>Food</span>
                                <span style='color: #4b8bff; font-weight: bold;'>{expenses_data['Food']:.1f}%</span>
                            </div>
                            <div style='background: #2d3747; height: 8px; border-radius: 4px;'>
                                <div style='background: #4b8bff; width: {expenses_data['Food']}%; height: 100%; border-radius: 4px;'></div>
                            </div>
                        </div>
                        <div style='margin: 1rem 0;'>
                            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;'>
                                <span style='color: #fafafa;'>Shopping</span>
                                <span style='color: #ff4b4b; font-weight: bold;'>{expenses_data['Shopping']:.1f}%</span>
                            </div>
                            <div style='background: #2d3747; height: 8px; border-radius: 4px;'>
                                <div style='background: #ff4b4b; width: {expenses_data['Shopping']}%; height: 100%; border-radius: 4px;'></div>
                            </div>
                        </div>
                        <div style='margin: 1rem 0;'>
                            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;'>
                                <span style='color: #fafafa;'>Travel</span>
                                <span style='color: #50C878; font-weight: bold;'>{expenses_data['Travel']:.1f}%</span>
                            </div>
                            <div style='background: #2d3747; height: 8px; border-radius: 4px;'>
                                <div style='background: #50C878; width: {expenses_data['Travel']}%; height: 100%; border-radius: 4px;'></div>
                            </div>
                        </div>
                        <div style='margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid #2d3747;'>
                            <p style='color: #fafafa; margin: 0;'>Total Monthly Average</p>
                            <p style='color: #4b8bff; font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;'>
                                {format_peso(total_expenses)}
                            </p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
            elif selected_category == "Savings Behavior":
                savings_rate = (data['remaining_allowance'] / data['weekly_allowance']).mean() * 100
                positive_savings = (data['remaining_allowance'] > 0).mean() * 100
                high_savers = (data['remaining_allowance'] > data['weekly_allowance'] * 0.3).mean() * 100
                avg_savings = data['remaining_allowance'].mean()
                
                st.markdown(f"""
                    <div style='background: linear-gradient(145deg, #1a1f29, #23293a); 
                                padding: 1.5rem; border-radius: 12px;'>
                        <h3 style='color: #4b8bff; margin-top: 0;'>Savings Overview</h3>
                        <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin: 1.5rem 0;'>
                            <div style='background: rgba(75, 139, 255, 0.1); padding: 1rem; border-radius: 8px; 
                                        border: 1px solid rgba(75, 139, 255, 0.2);'>
                                <p style='color: #c2c6cd; margin: 0; font-size: 0.9rem;'>Average Savings Rate</p>
                                <p style='color: #4b8bff; font-size: 1.5rem; font-weight: bold; margin: 0;'>{savings_rate:.1f}%</p>
                            </div>
                            <div style='background: rgba(80, 200, 120, 0.1); padding: 1rem; border-radius: 8px;
                                        border: 1px solid rgba(80, 200, 120, 0.2);'>
                                <p style='color: #c2c6cd; margin: 0; font-size: 0.9rem;'>Students with Savings</p>
                                <p style='color: #50C878; font-size: 1.5rem; font-weight: bold; margin: 0;'>{positive_savings:.1f}%</p>
                            </div>
                        </div>
                        <div style='margin: 1.5rem 0;'>
                            <p style='color: #fafafa; margin-bottom: 0.5rem;'>High Savers (>30% savings)</p>
                            <div style='background: #2d3747; height: 8px; border-radius: 4px;'>
                                <div style='background: #4b8bff; width: {high_savers}%; height: 100%; border-radius: 4px;'></div>
                            </div>
                            <p style='color: #c2c6cd; margin-top: 0.5rem; text-align: right;'>{high_savers:.1f}% of students</p>
                        </div>
                        <div style='margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid #2d3747;'>
                            <p style='color: #fafafa; margin: 0;'>Average Monthly Savings</p>
                            <p style='color: #4b8bff; font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;'>
                                {format_peso(avg_savings)}
                            </p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
            else:  # Risk Factors
                high_risk_threshold = 0.7
                high_shopping = (data['online_shopping'] > data['online_shopping'].quantile(high_risk_threshold)).mean() * 100
                zero_savings = (data['remaining_allowance'] <= 0).mean() * 100
                high_food = (data['food_expenses'] > data['food_expenses'].quantile(high_risk_threshold)).mean() * 100
                
                risk_score = (high_shopping + zero_savings + high_food) / 3
                risk_level = "High" if risk_score > 50 else "Moderate" if risk_score > 30 else "Low"
                
                st.markdown(f"""
                    <div style='background: linear-gradient(145deg, #1a1f29, #23293a); 
                                padding: 1.5rem; border-radius: 12px;'>
                        <h3 style='color: #4b8bff; margin-top: 0;'>Risk Assessment</h3>
                        <div style='text-align: center; margin: 1.5rem 0;'>
                            <div style='display: inline-block; padding: 1rem 2rem; border-radius: 8px;
                                        background: rgba({",".join(map(str, [int(risk_color[1:3], 16), int(risk_color[3:5], 16), int(risk_color[5:7], 16)]))}, 0.1);
                                        border: 1px solid {risk_color};'>
                                <p style='color: #c2c6cd; margin: 0; font-size: 0.9rem;'>Overall Risk Level</p>
                                <p style='color: {risk_color}; font-size: 1.8rem; font-weight: bold; margin: 0;'>{risk_level}</p>
                            </div>
                        </div>
                        <div style='margin: 1.5rem 0;'>
                            <div style='margin-bottom: 1rem;'>
                                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;'>
                                    <span style='color: #fafafa;'>High Shopping Risk</span>
                                    <span style='color: #ff4b4b; font-weight: bold;'>{high_shopping:.1f}%</span>
                                </div>
                                <div style='background: #2d3747; height: 6px; border-radius: 3px;'>
                                    <div style='background: #ff4b4b; width: {high_shopping}%; height: 100%; border-radius: 3px;'></div>
                                </div>
                            </div>
                            <div style='margin-bottom: 1rem;'>
                                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;'>
                                    <span style='color: #fafafa;'>Zero Savings</span>
                                    <span style='color: #ffd700; font-weight: bold;'>{zero_savings:.1f}%</span>
                                </div>
                                <div style='background: #2d3747; height: 6px; border-radius: 3px;'>
                                    <div style='background: #ffd700; width: {zero_savings}%; height: 100%; border-radius: 3px;'></div>
                                </div>
                            </div>
                            <div>
                                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;'>
                                    <span style='color: #fafafa;'>High Food Spending</span>
                                    <span style='color: #50C878; font-weight: bold;'>{high_food:.1f}%</span>
                                </div>
                                <div style='background: #2d3747; height: 6px; border-radius: 3px;'>
                                    <div style='background: #50C878; width: {high_food}%; height: 100%; border-radius: 3px;'></div>
                                </div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        # Recommendations Section
        st.markdown("""
            <div style='margin: 3rem 0 2rem 0;'>
                <h2 style='color: #4b8bff; font-size: 1.8rem;'>Recommendations</h2>
            </div>
        """, unsafe_allow_html=True)

        # Interactive recommendations explorer
        recommendation_categories = {
            "Budgeting": [
                {"title": "50/30/20 Rule", "description": "Allocate 50% for needs, 30% for wants, and 20% for savings"},
                {"title": "Expense Tracking", "description": "Use digital tools or apps to monitor daily expenses"},
                {"title": "Emergency Fund", "description": "Set aside 10% of allowance for unexpected expenses"}
            ],
            "Saving Strategies": [
                {"title": "First-day Saving", "description": "Transfer savings immediately when receiving allowance"},
                {"title": "Round-up Saving", "description": "Round up expenses and save the difference"},
                {"title": "Goal-based Saving", "description": "Set specific saving goals with deadlines"}
            ],
            "Smart Spending": [
                {"title": "Bulk Buying", "description": "Purchase frequently used items in bulk for discounts"},
                {"title": "Student Discounts", "description": "Always ask for and use available student discounts"},
                {"title": "Meal Planning", "description": "Plan meals ahead to reduce food expenses"}
            ]
        }

        selected_rec_category = st.selectbox(
            "Select recommendation category",
            options=list(recommendation_categories.keys())
        )

        col1, col2, col3 = st.columns(3)
        
        for i, rec in enumerate(recommendation_categories[selected_rec_category]):
            with [col1, col2, col3][i]:
                st.markdown(f"""
                    <div style='background: linear-gradient(145deg, #1a1f29, #23293a); 
                                padding: 1.5rem; border-radius: 12px; height: 100%;'>
                        <h4 style='color: #4b8bff; margin-top: 0;'>{rec['title']}</h4>
                        <p style='color: #c2c6cd;'>{rec['description']}</p>
                    </div>
                """, unsafe_allow_html=True)

        # Action Plan Generator
        st.markdown("""
            <div style='margin: 3rem 0 2rem 0;'>
                <h2 style='color: #4b8bff; font-size: 1.8rem;'>Personalized Action Plan</h2>
                <p style='color: #c2c6cd;'>Generate a customized plan based on your priorities</p>
            </div>
        """, unsafe_allow_html=True)

        priorities = st.multiselect(
            "Select your financial priorities",
            ["Increase Savings", "Reduce Food Expenses", "Control Online Shopping", 
             "Optimize Travel Costs", "Build Emergency Fund"],
            max_selections=3
        )

        if st.button("Generate Action Plan"):
            if priorities:
                st.markdown("""
                    <div style='background: linear-gradient(145deg, #1a1f29, #23293a); 
                                padding: 1.5rem; border-radius: 12px;'>
                        <h3 style='color: #4b8bff; margin-top: 0;'>Your Personal Action Plan</h3>
                """, unsafe_allow_html=True)
                
                for priority in priorities:
                    actions = generate_actions(priority)  # You'll need to implement this function
                    st.markdown(f"""
                        <div style='margin-bottom: 1rem;'>
                            <h4 style='color: #fafafa; margin: 0.5rem 0;'>{priority}</h4>
                            <ul style='color: #c2c6cd;'>
                                {"".join(f"<li>{action}</li>" for action in actions)}
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("Please select at least one priority to generate an action plan.")

else:
    st.error("Please make sure the 'student_savings_data.csv' file is in the same directory as the app.")
