import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="Chapter 3: Scalable AI", page_icon="🚀", layout="wide")

st.title("🚀 Chapter 3: Scalable AI")

st.markdown("""
## From Prototype to Production

You have a working prototype - now how do you make it available to users efficiently and flexibly? 
Deployment practices can vary considerably, even within the same organization. This chapter covers 
patterns for LLM application deployment that are independent of your specific technology choices.

### The Fundamental Shift
""")

# Prototype vs Production comparison
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🔬 Research/Prototype Phase
    - Static datasets
    - Proof of concept focus
    - Single-user environment
    - Manual processes
    - Experimental mindset
    - Performance secondary
    """)

with col2:
    st.markdown("""
    ### 🏭 Production Phase
    - Dynamic, real-time data
    - Reliability and scale focus
    - Multi-user environment
    - Automated processes
    - Operational mindset
    - Performance critical
    """)

# Production Readiness Maturity Model
st.markdown("## 📊 Production Readiness Maturity Model")

maturity_data = {
    'Level': ['Level 1: Prototype', 'Level 2: MVP', 'Level 3: Production', 'Level 4: Scale', 'Level 5: Enterprise'],
    'Characteristics': [
        'Local development, manual testing',
        'Basic deployment, limited users',
        'Automated CI/CD, monitoring',
        'Auto-scaling, load balancing',
        'Multi-region, disaster recovery'
    ],
    'Users': [1, 10, 100, 1000, 10000],
    'Availability': [90, 95, 99, 99.5, 99.9],
    'Complexity': [1, 3, 5, 7, 9]
}

df_maturity = pd.DataFrame(maturity_data)

# Maturity visualization
fig_maturity = go.Figure()

fig_maturity.add_trace(go.Scatter(
    x=df_maturity['Complexity'],
    y=df_maturity['Availability'],
    mode='markers+lines+text',
    marker=dict(size=np.log10(df_maturity['Users']) * 10, color=df_maturity['Complexity'], 
                colorscale='viridis', showscale=True),
    text=df_maturity['Level'],
    textposition="top center",
    line=dict(width=3),
    name='Maturity Path'
))

fig_maturity.update_layout(
    title='Production Readiness Maturity Model',
    xaxis_title='System Complexity',
    yaxis_title='Availability (%)',
    showlegend=False,
    height=500
)

st.plotly_chart(fig_maturity, use_container_width=True)

# Deployment Patterns
st.markdown("## 🏗️ Deployment Patterns")

deployment_tabs = st.tabs(["Monolithic", "Microservices", "Serverless", "Hybrid"])

with deployment_tabs[0]:
    st.markdown("### 🏢 Monolithic Deployment")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Monolithic architecture diagram
        fig_mono = go.Figure()
        
        # Single large component
        fig_mono.add_shape(
            type="rect",
            x0=1, y0=1, x1=4, y1=4,
            fillcolor="lightblue",
            line=dict(color="blue", width=2)
        )
        
        fig_mono.add_annotation(
            x=2.5, y=2.5,
            text="RAG Application<br>• Retrieval<br>• Generation<br>• API<br>• UI",
            showarrow=False,
            font=dict(size=12)
        )
        
        fig_mono.update_layout(
            title="Monolithic Architecture",
            xaxis=dict(range=[0, 5], showgrid=False, showticklabels=False),
            yaxis=dict(range=[0, 5], showgrid=False, showticklabels=False),
            height=300
        )
        
        st.plotly_chart(fig_mono, use_container_width=True)
    
    with col2:
        st.markdown("""
        **Pros:**
        - Simple deployment
        - Easy debugging
        - Lower latency
        - Shared resources
        
        **Cons:**
        - Scaling limitations
        - Technology lock-in
        - Single point of failure
        - Deployment complexity
        """)

with deployment_tabs[1]:
    st.markdown("### 🔧 Microservices Deployment")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Microservices architecture diagram
        fig_micro = go.Figure()
        
        services = [
            {"x": 1, "y": 3, "text": "Document\nService", "color": "lightcoral"},
            {"x": 3, "y": 3, "text": "Retrieval\nService", "color": "lightgreen"},
            {"x": 5, "y": 3, "text": "Generation\nService", "color": "lightblue"},
            {"x": 2, "y": 1, "text": "API\nGateway", "color": "lightyellow"},
            {"x": 4, "y": 1, "text": "UI\nService", "color": "lightpink"}
        ]
        
        for service in services:
            fig_micro.add_shape(
                type="rect",
                x0=service["x"]-0.4, y0=service["y"]-0.4,
                x1=service["x"]+0.4, y1=service["y"]+0.4,
                fillcolor=service["color"],
                line=dict(color="black", width=1)
            )
            
            fig_micro.add_annotation(
                x=service["x"], y=service["y"],
                text=service["text"],
                showarrow=False,
                font=dict(size=10)
            )
        
        fig_micro.update_layout(
            title="Microservices Architecture",
            xaxis=dict(range=[0, 6], showgrid=False, showticklabels=False),
            yaxis=dict(range=[0, 4], showgrid=False, showticklabels=False),
            height=300
        )
        
        st.plotly_chart(fig_micro, use_container_width=True)
    
    with col2:
        st.markdown("""
        **Pros:**
        - Independent scaling
        - Technology diversity
        - Fault isolation
        - Team autonomy
        
        **Cons:**
        - Network complexity
        - Data consistency
        - Operational overhead
        - Debugging difficulty
        """)

with deployment_tabs[2]:
    st.markdown("### ⚡ Serverless Deployment")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Serverless architecture
        serverless_data = {
            'Component': ['API Gateway', 'Lambda Functions', 'Vector DB', 'LLM API', 'Storage'],
            'Scaling': ['Auto', 'Auto', 'Managed', 'Managed', 'Auto'],
            'Cost Model': ['Pay-per-request', 'Pay-per-execution', 'Pay-per-query', 'Pay-per-token', 'Pay-per-GB']
        }
        
        df_serverless = pd.DataFrame(serverless_data)
        st.dataframe(df_serverless, use_container_width=True)
    
    with col2:
        st.markdown("""
        **Pros:**
        - Zero server management
        - Automatic scaling
        - Pay-per-use pricing
        - High availability
        
        **Cons:**
        - Cold start latency
        - Vendor lock-in
        - Limited control
        - Debugging challenges
        """)

with deployment_tabs[3]:
    st.markdown("### 🔄 Hybrid Deployment")
    
    st.markdown("""
    Combines multiple patterns based on component requirements:
    
    - **Critical Path**: Low-latency components on dedicated infrastructure
    - **Batch Processing**: Serverless for periodic tasks
    - **Data Storage**: Managed services for databases
    - **Edge Computing**: CDN for static assets
    """)

# Scaling Considerations
st.markdown("## 📈 Scaling Considerations")

scaling_factors = {
    'Factor': ['Compute', 'Memory', 'Storage', 'Network', 'Cost'],
    'Retrieval': [6, 8, 9, 7, 6],
    'Generation': [9, 7, 4, 6, 9],
    'Overall': [8, 7, 7, 6, 8]
}

df_scaling = pd.DataFrame(scaling_factors)

fig_scaling = go.Figure()

for component in ['Retrieval', 'Generation', 'Overall']:
    fig_scaling.add_trace(go.Scatterpolar(
        r=df_scaling[component],
        theta=df_scaling['Factor'],
        fill='toself',
        name=component
    ))

fig_scaling.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 10]
        )),
    showlegend=True,
    title="Resource Requirements by Component"
)

st.plotly_chart(fig_scaling, use_container_width=True)

# Performance Optimization Strategies
st.markdown("## ⚡ Performance Optimization Strategies")

optimization_strategies = st.tabs(["Caching", "Load Balancing", "Auto-scaling", "Monitoring"])

with optimization_strategies[0]:
    st.markdown("### 💾 Caching Strategies")
    
    cache_levels = {
        'Level': ['Browser Cache', 'CDN Cache', 'Application Cache', 'Database Cache', 'Model Cache'],
        'Hit Rate (%)': [85, 70, 60, 80, 40],
        'Latency Reduction (ms)': [500, 200, 50, 20, 1000]
    }
    
    df_cache = pd.DataFrame(cache_levels)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hit_rate = px.bar(df_cache, x='Level', y='Hit Rate (%)',
                             title='Cache Hit Rates',
                             color='Hit Rate (%)',
                             color_continuous_scale='RdYlGn')
        st.plotly_chart(fig_hit_rate, use_container_width=True)
    
    with col2:
        fig_latency = px.bar(df_cache, x='Level', y='Latency Reduction (ms)',
                            title='Latency Reduction by Cache Level',
                            color='Latency Reduction (ms)',
                            color_continuous_scale='viridis')
        st.plotly_chart(fig_latency, use_container_width=True)

with optimization_strategies[1]:
    st.markdown("### ⚖️ Load Balancing")
    
    lb_strategies = {
        'Strategy': ['Round Robin', 'Least Connections', 'Weighted', 'Geographic', 'AI-based'],
        'Simplicity': [9, 7, 6, 5, 3],
        'Effectiveness': [6, 7, 8, 8, 9],
        'Use Case': ['Equal servers', 'Varying loads', 'Different capacities', 'Global users', 'Dynamic optimization']
    }
    
    df_lb = pd.DataFrame(lb_strategies)
    
    fig_lb = px.scatter(df_lb, x='Simplicity', y='Effectiveness',
                       size=[10]*5, color='Strategy',
                       title='Load Balancing Strategies',
                       hover_data=['Use Case'])
    
    st.plotly_chart(fig_lb, use_container_width=True)

with optimization_strategies[2]:
    st.markdown("### 📊 Auto-scaling")
    
    # Simulate auto-scaling behavior
    time_points = list(range(24))
    base_load = [20 + 30 * np.sin(2 * np.pi * t / 24) + 10 * np.random.random() for t in time_points]
    instances = [max(1, int(load / 15)) for load in base_load]
    
    scaling_data = pd.DataFrame({
        'Hour': time_points,
        'Load': base_load,
        'Instances': instances
    })
    
    fig_scaling = go.Figure()
    
    fig_scaling.add_trace(go.Scatter(
        x=scaling_data['Hour'],
        y=scaling_data['Load'],
        mode='lines',
        name='Load',
        yaxis='y'
    ))
    
    fig_scaling.add_trace(go.Scatter(
        x=scaling_data['Hour'],
        y=scaling_data['Instances'],
        mode='lines+markers',
        name='Instances',
        yaxis='y2'
    ))
    
    fig_scaling.update_layout(
        title='Auto-scaling Behavior Over 24 Hours',
        xaxis_title='Hour of Day',
        yaxis=dict(title='Load', side='left'),
        yaxis2=dict(title='Instance Count', side='right', overlaying='y')
    )
    
    st.plotly_chart(fig_scaling, use_container_width=True)

with optimization_strategies[3]:
    st.markdown("### 📊 Monitoring & Observability")
    
    # Key metrics dashboard
    metrics = {
        'Metric': ['Response Time', 'Throughput', 'Error Rate', 'CPU Usage', 'Memory Usage'],
        'Current': [250, 1200, 0.5, 65, 70],
        'Target': [200, 1500, 0.1, 80, 80],
        'Status': ['Warning', 'Good', 'Critical', 'Good', 'Good']
    }
    
    df_metrics = pd.DataFrame(metrics)
    
    # Create gauge charts for key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig_response = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = 250,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Response Time (ms)"},
            delta = {'reference': 200},
            gauge = {'axis': {'range': [None, 500]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 200], 'color': "lightgray"},
                        {'range': [200, 300], 'color': "yellow"},
                        {'range': [300, 500], 'color': "red"}],
                    'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 300}}))
        
        st.plotly_chart(fig_response, use_container_width=True)
    
    with col2:
        fig_throughput = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = 1200,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Throughput (req/min)"},
            delta = {'reference': 1500},
            gauge = {'axis': {'range': [None, 2000]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 1000], 'color': "red"},
                        {'range': [1000, 1500], 'color': "yellow"},
                        {'range': [1500, 2000], 'color': "lightgray"}],
                    'threshold': {'line': {'color': "green", 'width': 4},
                                'thickness': 0.75, 'value': 1500}}))
        
        st.plotly_chart(fig_throughput, use_container_width=True)
    
    with col3:
        fig_error = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = 0.5,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Error Rate (%)"},
            delta = {'reference': 0.1},
            gauge = {'axis': {'range': [None, 2]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 0.1], 'color': "lightgray"},
                        {'range': [0.1, 0.5], 'color': "yellow"},
                        {'range': [0.5, 2], 'color': "red"}],
                    'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 0.5}}))
        
        st.plotly_chart(fig_error, use_container_width=True)

# Production Checklist
st.markdown("## ✅ Production Readiness Checklist")

checklist_items = [
    "Automated testing pipeline",
    "Monitoring and alerting",
    "Error handling and recovery",
    "Security and authentication",
    "Data backup and recovery",
    "Performance benchmarking",
    "Capacity planning",
    "Documentation and runbooks",
    "Incident response procedures",
    "Compliance and governance"
]

col1, col2 = st.columns(2)

for i, item in enumerate(checklist_items):
    if i < len(checklist_items) // 2:
        with col1:
            st.checkbox(item, value=np.random.choice([True, False]))
    else:
        with col2:
            st.checkbox(item, value=np.random.choice([True, False]))

# Cost Optimization
st.markdown("## 💰 Cost Optimization")

cost_factors = {
    'Component': ['Compute', 'Storage', 'Network', 'LLM API', 'Monitoring'],
    'Monthly Cost ($)': [2500, 800, 400, 5000, 300],
    'Optimization Potential (%)': [30, 50, 20, 40, 10]
}

df_cost = pd.DataFrame(cost_factors)

col1, col2 = st.columns(2)

with col1:
    fig_cost_breakdown = px.pie(df_cost, values='Monthly Cost ($)', names='Component',
                               title='Cost Breakdown')
    st.plotly_chart(fig_cost_breakdown, use_container_width=True)

with col2:
    fig_optimization = px.bar(df_cost, x='Component', y='Optimization Potential (%)',
                             title='Cost Optimization Potential',
                             color='Optimization Potential (%)',
                             color_continuous_scale='RdYlGn')
    st.plotly_chart(fig_optimization, use_container_width=True)

# Key Takeaways
st.markdown("---")
st.markdown("## 🎯 Key Takeaways")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### ✅ Success Factors:
    - Plan for scale from the beginning
    - Implement comprehensive monitoring
    - Automate deployment and testing
    - Design for failure and recovery
    """)

with col2:
    st.markdown("""
    ### 🚨 Common Pitfalls:
    - Premature optimization
    - Ignoring operational requirements
    - Underestimating complexity
    - Poor monitoring and alerting
    """)

# Sidebar
with st.sidebar:
    st.markdown("## 🚀 Chapter 3 Summary")
    st.markdown("""
    ### Key Concepts:
    - Production readiness
    - Deployment patterns
    - Scaling strategies
    - Performance optimization
    
    ### Deployment Options:
    ✅ Monolithic  
    ✅ Microservices  
    ✅ Serverless  
    ✅ Hybrid  
    """)
    
    st.markdown("### 💡 Key Takeaway")
    st.success("Successful production deployment requires careful planning, monitoring, and optimization across multiple dimensions.")

