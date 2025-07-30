import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import document_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from document_utils import display_document_upload_section, display_document_analysis, display_rag_demo

st.set_page_config(page_title="Observable AI", page_icon="👁️", layout="wide")

st.title("👁️ Observable AI")

st.markdown("""
## The Importance of Observability

In production AI systems, observability is crucial for maintaining reliability, performance, and user trust. 
This chapter covers monitoring, logging, tracing, and handling data drift in RAG systems.

### What is Observable AI?

Observable AI systems provide visibility into:
- **System Performance**: Response times, throughput, resource usage
- **Model Behavior**: Prediction quality, confidence scores, drift detection
- **User Experience**: Satisfaction, engagement, error rates
- **Data Quality**: Input validation, output verification, anomaly detection
""")

# Data and Concept Drift
st.markdown("## 📊 Data and Concept Drift")

st.markdown("""
### Understanding Drift Types

Drift occurs when the statistical properties of data change over time, affecting model performance.
""")

drift_tabs = st.tabs(["Data Drift", "Concept Drift", "Detection Methods"])

with drift_tabs[0]:
    st.markdown("### 📈 Data Drift")
    
    # Simulate data drift over time
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    # Original distribution
    original_mean = 0
    original_std = 1
    
    # Gradual drift
    drift_factor = np.linspace(0, 2, len(dates))
    drifted_data = [np.random.normal(original_mean + drift * 0.5, original_std + drift * 0.2, 100) 
                   for drift in drift_factor]
    
    # Calculate distribution statistics over time
    means = [np.mean(data) for data in drifted_data]
    stds = [np.std(data) for data in drifted_data]
    
    drift_df = pd.DataFrame({
        'Date': dates,
        'Mean': means,
        'Std': stds
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_mean = px.line(drift_df, x='Date', y='Mean',
                          title='Data Drift: Mean Over Time',
                          labels={'Mean': 'Distribution Mean'})
        fig_mean.add_hline(y=original_mean, line_dash="dash", 
                          annotation_text="Original Mean")
        st.plotly_chart(fig_mean, use_container_width=True)
    
    with col2:
        fig_std = px.line(drift_df, x='Date', y='Std',
                         title='Data Drift: Standard Deviation Over Time',
                         labels={'Std': 'Distribution Std'})
        fig_std.add_hline(y=original_std, line_dash="dash", 
                         annotation_text="Original Std")
        st.plotly_chart(fig_std, use_container_width=True)

with drift_tabs[1]:
    st.markdown("### 🎯 Concept Drift")
    
    # Simulate concept drift - relationship between features and target changes
    concept_data = {
        'Time Period': ['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024'],
        'Feature Importance': [
            [0.4, 0.3, 0.2, 0.1],  # Q1
            [0.3, 0.4, 0.2, 0.1],  # Q2
            [0.2, 0.3, 0.4, 0.1],  # Q3
            [0.1, 0.2, 0.3, 0.4]   # Q4
        ],
        'Features': ['Query Length', 'User Type', 'Time of Day', 'Device Type']
    }
    
    # Create heatmap for concept drift
    importance_matrix = np.array([concept_data['Feature Importance'][i] for i in range(4)])
    
    fig_concept = go.Figure(data=go.Heatmap(
        z=importance_matrix.T,
        x=concept_data['Time Period'],
        y=concept_data['Features'],
        colorscale='RdYlBu_r',
        text=importance_matrix.T,
        texttemplate="%{text:.2f}",
        textfont={"size": 12}
    ))
    
    fig_concept.update_layout(
        title='Concept Drift: Feature Importance Over Time',
        xaxis_title='Time Period',
        yaxis_title='Features'
    )
    
    st.plotly_chart(fig_concept, use_container_width=True)

with drift_tabs[2]:
    st.markdown("### 🔍 Drift Detection Methods")
    
    detection_methods = {
        'Method': ['Statistical Tests', 'Distribution Distance', 'Model Performance', 'Ensemble Methods'],
        'Sensitivity': [7, 8, 6, 9],
        'Computational Cost': [3, 5, 7, 8],
        'Interpretability': [9, 6, 8, 5],
        'Use Case': ['Simple features', 'Complex distributions', 'End-to-end', 'Robust detection']
    }
    
    df_detection = pd.DataFrame(detection_methods)
    
    fig_detection = px.scatter(df_detection, x='Computational Cost', y='Sensitivity',
                              size='Interpretability', color='Method',
                              title='Drift Detection Methods Comparison',
                              hover_data=['Use Case'])
    
    st.plotly_chart(fig_detection, use_container_width=True)

# Logging and Tracing
st.markdown("## 📝 Logging and Tracing")

logging_tabs = st.tabs(["Log Levels", "Distributed Tracing", "Log Analysis"])

with logging_tabs[0]:
    st.markdown("### 📊 Log Levels and Usage")
    
    log_data = {
        'Level': ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        'Volume (%)': [40, 35, 15, 8, 2],
        'Retention (days)': [7, 30, 90, 365, 365],
        'Alert': [False, False, True, True, True]
    }
    
    df_logs = pd.DataFrame(log_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_volume = px.pie(df_logs, values='Volume (%)', names='Level',
                           title='Log Volume Distribution',
                           color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig_volume, use_container_width=True)
    
    with col2:
        fig_retention = px.bar(df_logs, x='Level', y='Retention (days)',
                              title='Log Retention Policies',
                              color='Alert',
                              color_discrete_map={True: 'red', False: 'blue'})
        st.plotly_chart(fig_retention, use_container_width=True)

with logging_tabs[1]:
    st.markdown("### 🔗 Distributed Tracing")
    
    # Simulate a RAG request trace
    trace_data = {
        'Service': ['API Gateway', 'Query Processor', 'Vector DB', 'Retrieval Service', 'LLM Service', 'Response Formatter'],
        'Start Time (ms)': [0, 5, 15, 25, 100, 2500],
        'Duration (ms)': [2505, 2495, 80, 70, 2400, 5],
        'Status': ['Success', 'Success', 'Success', 'Success', 'Success', 'Success']
    }
    
    df_trace = pd.DataFrame(trace_data)
    df_trace['End Time (ms)'] = df_trace['Start Time (ms)'] + df_trace['Duration (ms)']
    
    # Create Gantt chart for trace
    fig_trace = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for i, row in df_trace.iterrows():
        fig_trace.add_trace(go.Scatter(
            x=[row['Start Time (ms)'], row['End Time (ms)']],
            y=[i, i],
            mode='lines',
            line=dict(color=colors[i % len(colors)], width=20),
            name=row['Service'],
            hovertemplate=f"<b>{row['Service']}</b><br>Duration: {row['Duration (ms)']}ms<extra></extra>"
        ))
    
    fig_trace.update_layout(
        title='Distributed Trace: RAG Request Flow',
        xaxis_title='Time (ms)',
        yaxis_title='Service',
        yaxis=dict(tickmode='array', tickvals=list(range(len(df_trace))), 
                  ticktext=df_trace['Service']),
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig_trace, use_container_width=True)

with logging_tabs[2]:
    st.markdown("### 📈 Log Analysis Dashboard")
    
    # Simulate log analysis metrics
    hours = list(range(24))
    error_rates = [0.1 + 0.05 * np.sin(2 * np.pi * h / 24) + 0.02 * np.random.random() for h in hours]
    response_times = [200 + 50 * np.sin(2 * np.pi * h / 24) + 20 * np.random.random() for h in hours]
    
    log_analysis_df = pd.DataFrame({
        'Hour': hours,
        'Error Rate (%)': error_rates,
        'Avg Response Time (ms)': response_times
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_errors = px.line(log_analysis_df, x='Hour', y='Error Rate (%)',
                            title='Error Rate Over 24 Hours',
                            markers=True)
        fig_errors.add_hline(y=0.15, line_dash="dash", line_color="red",
                            annotation_text="Alert Threshold")
        st.plotly_chart(fig_errors, use_container_width=True)
    
    with col2:
        fig_response = px.line(log_analysis_df, x='Hour', y='Avg Response Time (ms)',
                              title='Response Time Over 24 Hours',
                              markers=True)
        fig_response.add_hline(y=250, line_dash="dash", line_color="orange",
                              annotation_text="SLA Threshold")
        st.plotly_chart(fig_response, use_container_width=True)

# GenAI Monitoring
st.markdown("## 🤖 GenAI Monitoring")

genai_tabs = st.tabs(["Quality Metrics", "Cost Tracking", "Usage Patterns", "Anomaly Detection"])

with genai_tabs[0]:
    st.markdown("### 📊 Quality Metrics")
    
    quality_metrics = {
        'Metric': ['Relevance Score', 'Factual Accuracy', 'Coherence', 'Completeness', 'User Satisfaction'],
        'Current': [8.2, 7.8, 8.5, 7.9, 8.1],
        'Target': [8.5, 8.5, 8.5, 8.5, 8.5],
        'Trend': ['↗️', '↘️', '→', '↗️', '↗️']
    }
    
    df_quality = pd.DataFrame(quality_metrics)
    
    fig_quality = go.Figure()
    
    fig_quality.add_trace(go.Bar(
        name='Current',
        x=df_quality['Metric'],
        y=df_quality['Current'],
        marker_color='lightblue'
    ))
    
    fig_quality.add_trace(go.Bar(
        name='Target',
        x=df_quality['Metric'],
        y=df_quality['Target'],
        marker_color='lightcoral'
    ))
    
    fig_quality.update_layout(
        title='Quality Metrics Dashboard',
        yaxis_title='Score (1-10)',
        barmode='group'
    )
    
    st.plotly_chart(fig_quality, use_container_width=True)

with genai_tabs[1]:
    st.markdown("### 💰 Cost Tracking")
    
    # Simulate cost breakdown
    cost_data = {
        'Component': ['LLM API Calls', 'Vector Database', 'Compute Resources', 'Storage', 'Monitoring'],
        'Daily Cost ($)': [450, 120, 200, 30, 25],
        'Monthly Projection ($)': [13500, 3600, 6000, 900, 750]
    }
    
    df_cost = pd.DataFrame(cost_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_daily_cost = px.pie(df_cost, values='Daily Cost ($)', names='Component',
                               title='Daily Cost Breakdown')
        st.plotly_chart(fig_daily_cost, use_container_width=True)
    
    with col2:
        # Cost trend over time
        days = list(range(1, 31))
        daily_costs = [800 + 50 * np.sin(2 * np.pi * d / 7) + 20 * np.random.random() for d in days]
        
        cost_trend_df = pd.DataFrame({
            'Day': days,
            'Cost ($)': daily_costs
        })
        
        fig_cost_trend = px.line(cost_trend_df, x='Day', y='Cost ($)',
                                title='Daily Cost Trend',
                                markers=True)
        st.plotly_chart(fig_cost_trend, use_container_width=True)

with genai_tabs[2]:
    st.markdown("### 📱 Usage Patterns")
    
    # Simulate usage patterns
    usage_data = {
        'Hour': list(range(24)),
        'Requests': [100 + 200 * np.sin(2 * np.pi * h / 24) + 50 * np.random.random() for h in range(24)],
        'Unique Users': [50 + 100 * np.sin(2 * np.pi * h / 24) + 25 * np.random.random() for h in range(24)]
    }
    
    df_usage = pd.DataFrame(usage_data)
    
    fig_usage = go.Figure()
    
    fig_usage.add_trace(go.Scatter(
        x=df_usage['Hour'],
        y=df_usage['Requests'],
        mode='lines+markers',
        name='Requests',
        yaxis='y'
    ))
    
    fig_usage.add_trace(go.Scatter(
        x=df_usage['Hour'],
        y=df_usage['Unique Users'],
        mode='lines+markers',
        name='Unique Users',
        yaxis='y2'
    ))
    
    fig_usage.update_layout(
        title='Usage Patterns Over 24 Hours',
        xaxis_title='Hour of Day',
        yaxis=dict(title='Requests', side='left'),
        yaxis2=dict(title='Unique Users', side='right', overlaying='y')
    )
    
    st.plotly_chart(fig_usage, use_container_width=True)

with genai_tabs[3]:
    st.markdown("### 🚨 Anomaly Detection")
    
    # Simulate anomaly detection
    np.random.seed(42)
    timestamps = pd.date_range(start='2024-01-01', periods=100, freq='H')
    normal_pattern = 100 + 20 * np.sin(2 * np.pi * np.arange(100) / 24)
    noise = np.random.normal(0, 5, 100)
    
    # Add some anomalies
    anomaly_indices = [25, 45, 78]
    values = normal_pattern + noise
    for idx in anomaly_indices:
        values[idx] += np.random.choice([-50, 50])
    
    anomaly_df = pd.DataFrame({
        'Timestamp': timestamps,
        'Value': values,
        'Is_Anomaly': [i in anomaly_indices for i in range(100)]
    })
    
    fig_anomaly = px.scatter(anomaly_df, x='Timestamp', y='Value',
                            color='Is_Anomaly',
                            title='Anomaly Detection in Response Times',
                            color_discrete_map={True: 'red', False: 'blue'})
    
    # Add normal range
    fig_anomaly.add_hline(y=normal_pattern.mean() + 2*normal_pattern.std(), 
                         line_dash="dash", line_color="orange",
                         annotation_text="Upper Threshold")
    fig_anomaly.add_hline(y=normal_pattern.mean() - 2*normal_pattern.std(), 
                         line_dash="dash", line_color="orange",
                         annotation_text="Lower Threshold")
    
    st.plotly_chart(fig_anomaly, use_container_width=True)

# Monitoring Best Practices
st.markdown("## 🎯 Monitoring Best Practices")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### ✅ Do's:
    - Monitor end-to-end user experience
    - Set up proactive alerting
    - Track business metrics alongside technical metrics
    - Implement gradual rollouts with monitoring
    - Use synthetic monitoring for critical paths
    - Maintain monitoring during incidents
    """)

with col2:
    st.markdown("""
    ### ❌ Don'ts:
    - Monitor everything without prioritization
    - Set up alerts without clear action plans
    - Ignore false positive alerts
    - Monitor only technical metrics
    - Forget to monitor the monitoring system
    - Overlook user-facing metrics
    """)

# Interactive Monitoring Dashboard
st.markdown("## 📊 Interactive Monitoring Dashboard")

st.markdown("Configure your monitoring preferences:")

col1, col2, col3 = st.columns(3)

with col1:
    alert_threshold = st.slider("Error Rate Alert Threshold (%)", 0.1, 2.0, 0.5, 0.1)
    response_threshold = st.slider("Response Time Threshold (ms)", 100, 1000, 300, 50)

with col2:
    monitoring_interval = st.selectbox("Monitoring Interval", ["1 minute", "5 minutes", "15 minutes"])
    retention_period = st.selectbox("Data Retention", ["7 days", "30 days", "90 days", "1 year"])

with col3:
    notification_channels = st.multiselect("Notification Channels", 
                                         ["Email", "Slack", "PagerDuty", "SMS"],
                                         default=["Email", "Slack"])

# Display current monitoring configuration
st.markdown("### Current Configuration:")
config_data = {
    'Setting': ['Error Rate Threshold', 'Response Time Threshold', 'Monitoring Interval', 'Retention Period', 'Notifications'],
    'Value': [f"{alert_threshold}%", f"{response_threshold}ms", monitoring_interval, retention_period, ", ".join(notification_channels)]
}

df_config = pd.DataFrame(config_data)
st.dataframe(df_config, use_container_width=True)

# Key Takeaways
st.markdown("---")
st.markdown("## 🎯 Key Takeaways")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### ✅ Success Factors:
    - Comprehensive monitoring strategy
    - Proactive drift detection
    - Actionable alerting
    - Regular monitoring review
    """)

with col2:
    st.markdown("""
    ### 🚨 Common Pitfalls:
    - Alert fatigue from too many notifications
    - Monitoring lag in detection
    - Ignoring business impact metrics
    - Poor incident response procedures
    """)

# Interactive Document Processing Section
st.markdown("---")
st.markdown("## 🔬 Interactive Observable AI Demo")

# Document upload and processing
text, doc_name = display_document_upload_section()

if text and doc_name:
    # Display document analysis
    display_document_analysis(text, doc_name)
    
    # Observability analysis for the document
    st.markdown("### 👁️ Observability Analysis for Your Document")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Document Monitoring Metrics")
        
        doc_complexity = len(text.split()) / 100  # Complexity score
        readability = np.random.uniform(0.6, 0.9)  # Simulated readability
        
        st.metric("Document Complexity", f"{doc_complexity:.1f}")
        st.metric("Readability Score", f"{readability:.2f}")
        
        # Simulate monitoring alerts
        if doc_complexity > 50:
            st.warning("⚠️ High complexity detected - monitor performance closely")
        else:
            st.success("✅ Document complexity within normal range")
    
    with col2:
        st.markdown("#### Drift Detection Simulation")
        
        # Simulate concept drift detection
        baseline_similarity = np.random.uniform(0.8, 0.95)
        current_similarity = np.random.uniform(0.7, 0.9)
        drift_score = abs(baseline_similarity - current_similarity)
        
        st.metric("Baseline Similarity", f"{baseline_similarity:.3f}")
        st.metric("Current Similarity", f"{current_similarity:.3f}")
        st.metric("Drift Score", f"{drift_score:.3f}")
        
        if drift_score > 0.1:
            st.error("🚨 Concept drift detected!")
        else:
            st.success("✅ No significant drift detected")
    
    # RAG demonstration
    display_rag_demo(text, doc_name)

# Sidebar
with st.sidebar:
    st.markdown("## 👁️ Observable AI Summary")
    st.markdown("""
    ### Key Concepts:
    - Data and concept drift
    - Logging and tracing
    - GenAI monitoring
    - Anomaly detection
    
    ### Monitoring Areas:
    ✅ System performance  
    ✅ Model behavior  
    ✅ User experience  
    ✅ Cost tracking  
    """)
    
    st.markdown("### 💡 Key Takeaway")
    st.success("Observable AI systems enable proactive issue detection and continuous improvement through comprehensive monitoring.")

