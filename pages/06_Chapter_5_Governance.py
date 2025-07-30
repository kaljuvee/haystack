import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="Chapter 5: Governance of AI", page_icon="⚖️", layout="wide")

st.title("⚖️ Chapter 5: Governance of AI")

st.markdown("""
## The Governance Imperative

As AI systems become more prevalent in business operations, governance becomes critical for:
- **Risk Management**: Mitigating potential harms and liabilities
- **Compliance**: Meeting regulatory and industry requirements
- **Trust**: Building confidence with users and stakeholders
- **Sustainability**: Ensuring long-term viability and responsibility

This chapter covers the four pillars of AI governance: Cost Management, Data & Privacy, Security & Safety, and Model Licenses.
""")

# Governance Framework Overview
st.markdown("## 🏛️ AI Governance Framework")

governance_pillars = {
    'Pillar': ['Cost Management', 'Data & Privacy', 'Security & Safety', 'Model Licenses'],
    'Importance': [8, 10, 10, 7],
    'Complexity': [6, 9, 8, 5],
    'Regulatory Risk': [5, 10, 9, 6],
    'Business Impact': [9, 8, 9, 6]
}

df_governance = pd.DataFrame(governance_pillars)

fig_governance = go.Figure()

fig_governance.add_trace(go.Scatterpolar(
    r=df_governance['Importance'],
    theta=df_governance['Pillar'],
    fill='toself',
    name='Importance',
    line_color='blue'
))

fig_governance.add_trace(go.Scatterpolar(
    r=df_governance['Regulatory Risk'],
    theta=df_governance['Pillar'],
    fill='toself',
    name='Regulatory Risk',
    line_color='red'
))

fig_governance.add_trace(go.Scatterpolar(
    r=df_governance['Business Impact'],
    theta=df_governance['Pillar'],
    fill='toself',
    name='Business Impact',
    line_color='green'
))

fig_governance.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 10]
        )),
    showlegend=True,
    title="AI Governance Pillars Assessment"
)

st.plotly_chart(fig_governance, use_container_width=True)

# Cost Management
st.markdown("## 💰 Cost Management")

cost_tabs = st.tabs(["Cost Breakdown", "Optimization Strategies", "Budget Planning", "ROI Analysis"])

with cost_tabs[0]:
    st.markdown("### 📊 RAG System Cost Breakdown")
    
    cost_components = {
        'Component': ['LLM API Calls', 'Vector Database', 'Compute Infrastructure', 'Data Storage', 'Monitoring & Logging', 'Development & Maintenance'],
        'Monthly Cost ($)': [15000, 3000, 5000, 800, 500, 8000],
        'Variable': [True, False, True, False, False, False],
        'Optimization Potential (%)': [40, 20, 30, 50, 10, 25]
    }
    
    df_costs = pd.DataFrame(cost_components)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_cost_pie = px.pie(df_costs, values='Monthly Cost ($)', names='Component',
                             title='Monthly Cost Distribution')
        st.plotly_chart(fig_cost_pie, use_container_width=True)
    
    with col2:
        fig_optimization = px.bar(df_costs, x='Component', y='Optimization Potential (%)',
                                 title='Cost Optimization Potential',
                                 color='Variable',
                                 color_discrete_map={True: 'orange', False: 'blue'})
        st.plotly_chart(fig_optimization, use_container_width=True)

with cost_tabs[1]:
    st.markdown("### ⚡ Cost Optimization Strategies")
    
    optimization_strategies = {
        'Strategy': ['Model Caching', 'Batch Processing', 'Auto-scaling', 'Model Compression', 'Smart Routing'],
        'Implementation Effort': [3, 5, 7, 8, 6],
        'Cost Savings (%)': [25, 35, 20, 30, 15],
        'Risk Level': ['Low', 'Low', 'Medium', 'High', 'Medium']
    }
    
    df_optimization = pd.DataFrame(optimization_strategies)
    
    fig_optimization_scatter = px.scatter(df_optimization, x='Implementation Effort', y='Cost Savings (%)',
                                        size=[10]*5, color='Risk Level',
                                        title='Cost Optimization Strategies',
                                        hover_data=['Strategy'])
    
    st.plotly_chart(fig_optimization_scatter, use_container_width=True)
    
    # Cost optimization calculator
    st.markdown("#### 🧮 Cost Optimization Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        current_monthly_cost = st.number_input("Current Monthly Cost ($)", value=32300, step=100)
        cache_hit_rate = st.slider("Cache Hit Rate (%)", 0, 90, 60)
        batch_efficiency = st.slider("Batch Processing Efficiency (%)", 0, 50, 30)
    
    with col2:
        scaling_efficiency = st.slider("Auto-scaling Efficiency (%)", 0, 30, 15)
        compression_savings = st.slider("Model Compression Savings (%)", 0, 40, 20)
        
        # Calculate total savings
        total_savings = (cache_hit_rate * 0.25 + batch_efficiency * 0.35 + 
                        scaling_efficiency * 0.20 + compression_savings * 0.30) / 100
        
        optimized_cost = current_monthly_cost * (1 - total_savings)
        monthly_savings = current_monthly_cost - optimized_cost
        
        st.metric("Optimized Monthly Cost", f"${optimized_cost:,.0f}", 
                 f"-${monthly_savings:,.0f}")
        st.metric("Annual Savings", f"${monthly_savings * 12:,.0f}")

with cost_tabs[2]:
    st.markdown("### 📈 Budget Planning")
    
    # Simulate budget planning over time
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    budget_data = {
        'Month': months,
        'Planned Budget ($)': [30000, 32000, 35000, 38000, 40000, 42000, 45000, 47000, 50000, 52000, 55000, 58000],
        'Actual Spend ($)': [28000, 31000, 36000, 39000, 38000, 43000, 46000, 48000, 49000, 53000, 56000, 57000],
        'Forecast ($)': [29000, 32000, 37000, 40000, 41000, 44000, 47000, 49000, 51000, 54000, 57000, 59000]
    }
    
    df_budget = pd.DataFrame(budget_data)
    
    fig_budget = go.Figure()
    
    fig_budget.add_trace(go.Scatter(x=df_budget['Month'], y=df_budget['Planned Budget ($)'],
                                   mode='lines+markers', name='Planned Budget',
                                   line=dict(color='blue', dash='dash')))
    
    fig_budget.add_trace(go.Scatter(x=df_budget['Month'], y=df_budget['Actual Spend ($)'],
                                   mode='lines+markers', name='Actual Spend',
                                   line=dict(color='red')))
    
    fig_budget.add_trace(go.Scatter(x=df_budget['Month'], y=df_budget['Forecast ($)'],
                                   mode='lines+markers', name='Forecast',
                                   line=dict(color='green', dash='dot')))
    
    fig_budget.update_layout(title='Budget vs Actual Spend vs Forecast',
                            xaxis_title='Month',
                            yaxis_title='Cost ($)')
    
    st.plotly_chart(fig_budget, use_container_width=True)

with cost_tabs[3]:
    st.markdown("### 📊 ROI Analysis")
    
    # ROI calculation
    roi_metrics = {
        'Metric': ['Cost Reduction', 'Productivity Gain', 'Revenue Increase', 'Risk Mitigation'],
        'Annual Value ($)': [150000, 300000, 500000, 100000],
        'Confidence (%)': [90, 80, 70, 60]
    }
    
    df_roi = pd.DataFrame(roi_metrics)
    
    total_benefits = df_roi['Annual Value ($)'].sum()
    total_costs = 400000  # Annual AI system costs
    roi_percentage = ((total_benefits - total_costs) / total_costs) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Annual Benefits", f"${total_benefits:,}")
        st.metric("Total Annual Costs", f"${total_costs:,}")
        st.metric("ROI", f"{roi_percentage:.1f}%")
    
    with col2:
        fig_roi = px.bar(df_roi, x='Metric', y='Annual Value ($)',
                        title='ROI Components',
                        color='Confidence (%)',
                        color_continuous_scale='RdYlGn')
        st.plotly_chart(fig_roi, use_container_width=True)

# Data and Privacy
st.markdown("## 🔒 Data and Privacy")

privacy_tabs = st.tabs(["Data Classification", "Privacy Controls", "Compliance", "Data Lifecycle"])

with privacy_tabs[0]:
    st.markdown("### 📋 Data Classification")
    
    data_types = {
        'Data Type': ['Public', 'Internal', 'Confidential', 'Restricted'],
        'Volume (%)': [20, 40, 30, 10],
        'Risk Level': [1, 3, 7, 10],
        'Access Controls': ['Open', 'Employee Only', 'Need-to-Know', 'Executive Only']
    }
    
    df_data_types = pd.DataFrame(data_types)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_data_volume = px.pie(df_data_types, values='Volume (%)', names='Data Type',
                                title='Data Volume by Classification')
        st.plotly_chart(fig_data_volume, use_container_width=True)
    
    with col2:
        fig_risk = px.bar(df_data_types, x='Data Type', y='Risk Level',
                         title='Risk Level by Data Classification',
                         color='Risk Level',
                         color_continuous_scale='Reds')
        st.plotly_chart(fig_risk, use_container_width=True)

with privacy_tabs[1]:
    st.markdown("### 🛡️ Privacy Controls")
    
    privacy_controls = {
        'Control': ['Data Anonymization', 'Encryption at Rest', 'Encryption in Transit', 'Access Logging', 'Data Masking', 'Retention Policies'],
        'Implementation Status': ['Implemented', 'Implemented', 'Implemented', 'Implemented', 'Partial', 'Planned'],
        'Effectiveness': [8, 9, 9, 7, 6, 8],
        'Compliance Requirement': [True, True, True, True, False, True]
    }
    
    df_privacy = pd.DataFrame(privacy_controls)
    
    # Privacy controls status
    status_counts = df_privacy['Implementation Status'].value_counts()
    
    fig_privacy_status = px.pie(values=status_counts.values, names=status_counts.index,
                               title='Privacy Controls Implementation Status')
    st.plotly_chart(fig_privacy_status, use_container_width=True)

with privacy_tabs[2]:
    st.markdown("### 📜 Compliance Requirements")
    
    compliance_frameworks = {
        'Framework': ['GDPR', 'CCPA', 'HIPAA', 'SOX', 'ISO 27001'],
        'Applicability': ['EU Users', 'CA Users', 'Healthcare', 'Financial', 'Global'],
        'Compliance Score (%)': [85, 90, 75, 80, 88],
        'Priority': ['High', 'High', 'Medium', 'Medium', 'High']
    }
    
    df_compliance = pd.DataFrame(compliance_frameworks)
    
    fig_compliance = px.bar(df_compliance, x='Framework', y='Compliance Score (%)',
                           title='Compliance Framework Scores',
                           color='Priority',
                           color_discrete_map={'High': 'red', 'Medium': 'orange'})
    
    st.plotly_chart(fig_compliance, use_container_width=True)

with privacy_tabs[3]:
    st.markdown("### 🔄 Data Lifecycle Management")
    
    # Data lifecycle stages
    lifecycle_stages = ['Collection', 'Processing', 'Storage', 'Usage', 'Sharing', 'Retention', 'Deletion']
    privacy_measures = [
        'Consent Management',
        'Purpose Limitation',
        'Encryption & Access Control',
        'Audit Logging',
        'Data Agreements',
        'Automated Policies',
        'Secure Deletion'
    ]
    
    lifecycle_df = pd.DataFrame({
        'Stage': lifecycle_stages,
        'Privacy Measure': privacy_measures,
        'Automation Level': [7, 8, 9, 8, 6, 9, 8]
    })
    
    fig_lifecycle = px.bar(lifecycle_df, x='Stage', y='Automation Level',
                          title='Data Lifecycle Privacy Automation',
                          hover_data=['Privacy Measure'])
    
    st.plotly_chart(fig_lifecycle, use_container_width=True)

# Security and Safety
st.markdown("## 🛡️ Security and Safety")

security_tabs = st.tabs(["Threat Landscape", "Security Controls", "Safety Measures", "Incident Response"])

with security_tabs[0]:
    st.markdown("### ⚠️ AI Security Threat Landscape")
    
    threats = {
        'Threat': ['Prompt Injection', 'Data Poisoning', 'Model Extraction', 'Adversarial Attacks', 'Privacy Leakage'],
        'Likelihood': [8, 6, 5, 7, 7],
        'Impact': [7, 9, 6, 6, 9],
        'Current Mitigation': [6, 7, 8, 5, 7]
    }
    
    df_threats = pd.DataFrame(threats)
    df_threats['Risk Score'] = df_threats['Likelihood'] * df_threats['Impact']
    
    fig_threats = px.scatter(df_threats, x='Likelihood', y='Impact',
                            size='Risk Score', color='Current Mitigation',
                            title='AI Security Threat Matrix',
                            hover_data=['Threat'],
                            color_continuous_scale='RdYlGn')
    
    st.plotly_chart(fig_threats, use_container_width=True)

with security_tabs[1]:
    st.markdown("### 🔐 Security Controls")
    
    security_controls = {
        'Control Category': ['Authentication', 'Authorization', 'Input Validation', 'Output Filtering', 'Monitoring'],
        'Controls Implemented': [8, 6, 7, 5, 9],
        'Total Controls': [10, 8, 10, 8, 12],
        'Effectiveness': [85, 75, 70, 62, 90]
    }
    
    df_security = pd.DataFrame(security_controls)
    df_security['Implementation Rate (%)'] = (df_security['Controls Implemented'] / df_security['Total Controls']) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_implementation = px.bar(df_security, x='Control Category', y='Implementation Rate (%)',
                                   title='Security Controls Implementation Rate')
        st.plotly_chart(fig_implementation, use_container_width=True)
    
    with col2:
        fig_effectiveness = px.bar(df_security, x='Control Category', y='Effectiveness',
                                  title='Security Controls Effectiveness',
                                  color='Effectiveness',
                                  color_continuous_scale='RdYlGn')
        st.plotly_chart(fig_effectiveness, use_container_width=True)

with security_tabs[2]:
    st.markdown("### 🦺 AI Safety Measures")
    
    safety_measures = {
        'Measure': ['Content Filtering', 'Bias Detection', 'Hallucination Prevention', 'Human Oversight', 'Fail-safe Mechanisms'],
        'Implementation': ['Automated', 'Semi-automated', 'Manual Review', 'Human-in-loop', 'Automated'],
        'Coverage (%)': [95, 80, 60, 40, 90],
        'False Positive Rate (%)': [5, 15, 2, 1, 8]
    }
    
    df_safety = pd.DataFrame(safety_measures)
    
    fig_safety = px.scatter(df_safety, x='Coverage (%)', y='False Positive Rate (%)',
                           size=[10]*5, color='Implementation',
                           title='AI Safety Measures Effectiveness',
                           hover_data=['Measure'])
    
    st.plotly_chart(fig_safety, use_container_width=True)

with security_tabs[3]:
    st.markdown("### 🚨 Incident Response")
    
    # Incident response timeline
    incident_phases = {
        'Phase': ['Detection', 'Analysis', 'Containment', 'Eradication', 'Recovery', 'Lessons Learned'],
        'Target Time (hours)': [0.5, 2, 4, 8, 24, 72],
        'Automation Level': [90, 60, 70, 40, 50, 20]
    }
    
    df_incident = pd.DataFrame(incident_phases)
    
    fig_incident = go.Figure()
    
    fig_incident.add_trace(go.Bar(
        name='Target Time (hours)',
        x=df_incident['Phase'],
        y=df_incident['Target Time (hours)'],
        yaxis='y'
    ))
    
    fig_incident.add_trace(go.Scatter(
        name='Automation Level (%)',
        x=df_incident['Phase'],
        y=df_incident['Automation Level'],
        mode='lines+markers',
        yaxis='y2'
    ))
    
    fig_incident.update_layout(
        title='Incident Response Timeline and Automation',
        xaxis_title='Response Phase',
        yaxis=dict(title='Time (hours)', side='left'),
        yaxis2=dict(title='Automation Level (%)', side='right', overlaying='y')
    )
    
    st.plotly_chart(fig_incident, use_container_width=True)

# Model Licenses
st.markdown("## 📄 Model Licenses")

license_tabs = st.tabs(["License Types", "Compliance Matrix", "Risk Assessment", "Best Practices"])

with license_tabs[0]:
    st.markdown("### 📋 Model License Types")
    
    license_types = {
        'License': ['Apache 2.0', 'MIT', 'Custom Commercial', 'Research Only', 'GPL v3'],
        'Commercial Use': ['Yes', 'Yes', 'Yes', 'No', 'Limited'],
        'Modification': ['Yes', 'Yes', 'Limited', 'Yes', 'Yes'],
        'Distribution': ['Yes', 'Yes', 'Restricted', 'No', 'Yes'],
        'Attribution Required': ['Yes', 'Yes', 'Yes', 'Yes', 'Yes'],
        'Usage Count': [15, 8, 5, 3, 2]
    }
    
    df_licenses = pd.DataFrame(license_types)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_license_usage = px.pie(df_licenses, values='Usage Count', names='License',
                                  title='Model License Distribution')
        st.plotly_chart(fig_license_usage, use_container_width=True)
    
    with col2:
        # License compatibility matrix
        compatibility_matrix = np.array([
            [1, 1, 0.5, 0, 0.5],  # Apache 2.0
            [1, 1, 0.5, 0, 0.5],  # MIT
            [0.5, 0.5, 1, 0, 0],  # Custom Commercial
            [0, 0, 0, 1, 0],      # Research Only
            [0.5, 0.5, 0, 0, 1]   # GPL v3
        ])
        
        fig_compatibility = go.Figure(data=go.Heatmap(
            z=compatibility_matrix,
            x=df_licenses['License'],
            y=df_licenses['License'],
            colorscale='RdYlGn',
            text=compatibility_matrix,
            texttemplate="%{text}",
            textfont={"size": 12}
        ))
        
        fig_compatibility.update_layout(
            title='License Compatibility Matrix',
            xaxis_title='License A',
            yaxis_title='License B'
        )
        
        st.plotly_chart(fig_compatibility, use_container_width=True)

with license_tabs[1]:
    st.markdown("### ✅ License Compliance Matrix")
    
    models_compliance = {
        'Model': ['GPT-4', 'Claude-3', 'Llama-2', 'BERT', 'T5'],
        'License': ['Commercial API', 'Commercial API', 'Custom', 'Apache 2.0', 'Apache 2.0'],
        'Commercial Use': ['✅', '✅', '✅', '✅', '✅'],
        'Modification': ['❌', '❌', '✅', '✅', '✅'],
        'Redistribution': ['❌', '❌', '⚠️', '✅', '✅'],
        'Attribution': ['✅', '✅', '✅', '✅', '✅'],
        'Compliance Status': ['Compliant', 'Compliant', 'Review Needed', 'Compliant', 'Compliant']
    }
    
    df_model_compliance = pd.DataFrame(models_compliance)
    st.dataframe(df_model_compliance, use_container_width=True)

with license_tabs[2]:
    st.markdown("### ⚠️ License Risk Assessment")
    
    license_risks = {
        'Risk Factor': ['Viral Licensing', 'Commercial Restrictions', 'Attribution Failure', 'Modification Limits', 'Distribution Constraints'],
        'Probability': [3, 5, 7, 4, 6],
        'Impact': [9, 8, 4, 6, 7],
        'Current Mitigation': [8, 7, 9, 6, 7]
    }
    
    df_license_risks = pd.DataFrame(license_risks)
    df_license_risks['Risk Score'] = df_license_risks['Probability'] * df_license_risks['Impact']
    
    fig_license_risks = px.scatter(df_license_risks, x='Probability', y='Impact',
                                  size='Risk Score', color='Current Mitigation',
                                  title='License Risk Assessment',
                                  hover_data=['Risk Factor'],
                                  color_continuous_scale='RdYlGn')
    
    st.plotly_chart(fig_license_risks, use_container_width=True)

with license_tabs[3]:
    st.markdown("### 📚 License Management Best Practices")
    
    best_practices = [
        "Maintain a comprehensive license inventory",
        "Implement automated license scanning",
        "Regular compliance audits",
        "Clear approval processes for new models",
        "Legal review for commercial deployments",
        "Documentation of license obligations",
        "Training for development teams",
        "Vendor license management"
    ]
    
    col1, col2 = st.columns(2)
    
    for i, practice in enumerate(best_practices):
        if i < len(best_practices) // 2:
            with col1:
                st.checkbox(practice, value=True)
        else:
            with col2:
                st.checkbox(practice, value=np.random.choice([True, False]))

# Governance Maturity Assessment
st.markdown("## 📊 Governance Maturity Assessment")

maturity_areas = {
    'Area': ['Cost Management', 'Data Privacy', 'Security', 'Model Licenses'],
    'Current Maturity': [7, 8, 6, 5],
    'Target Maturity': [9, 9, 8, 7],
    'Gap': [2, 1, 2, 2]
}

df_maturity_assessment = pd.DataFrame(maturity_areas)

fig_maturity_assessment = go.Figure()

fig_maturity_assessment.add_trace(go.Bar(
    name='Current Maturity',
    x=df_maturity_assessment['Area'],
    y=df_maturity_assessment['Current Maturity'],
    marker_color='lightblue'
))

fig_maturity_assessment.add_trace(go.Bar(
    name='Target Maturity',
    x=df_maturity_assessment['Area'],
    y=df_maturity_assessment['Target Maturity'],
    marker_color='lightcoral'
))

fig_maturity_assessment.update_layout(
    title='Governance Maturity Assessment',
    yaxis_title='Maturity Level (1-10)',
    barmode='group'
)

st.plotly_chart(fig_maturity_assessment, use_container_width=True)

# Key Takeaways
st.markdown("---")
st.markdown("## 🎯 Key Takeaways")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### ✅ Governance Success Factors:
    - Comprehensive policy framework
    - Regular compliance monitoring
    - Cross-functional governance team
    - Continuous risk assessment
    - Stakeholder engagement
    """)

with col2:
    st.markdown("""
    ### 🚨 Common Governance Pitfalls:
    - Treating governance as an afterthought
    - Insufficient stakeholder involvement
    - Lack of regular policy updates
    - Poor incident response planning
    - Inadequate training and awareness
    """)

# Sidebar
with st.sidebar:
    st.markdown("## ⚖️ Chapter 5 Summary")
    st.markdown("""
    ### Governance Pillars:
    - Cost Management
    - Data & Privacy
    - Security & Safety
    - Model Licenses
    
    ### Key Activities:
    ✅ Risk assessment  
    ✅ Compliance monitoring  
    ✅ Policy development  
    ✅ Incident response  
    """)
    
    st.markdown("### 💡 Key Takeaway")
    st.success("Effective AI governance requires proactive management across cost, privacy, security, and licensing dimensions.")

