import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Car Insurance Claim Prediction", 
                   page_icon="🚗", 
                   layout="wide")

# --- Load Model ---
@st.cache_resource
def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# --- Load Encoder ---
@st.cache_resource
def load_encoder(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# --- Load Data ---
@st.cache_data
def load_data():
    try:
        # Try loading train data first (has target column)
        try:
            df = pd.read_csv("train.csv")
            data_source = "train.csv"
        except:
            df = pd.read_csv("test.csv")
            data_source = "test.csv"
        
        test_df = pd.read_csv("test.csv")
        importances_df = pd.read_csv("catboost_feature_importances.csv")
        return df, test_df, importances_df, data_source
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None, None, None, None

# --- Predict ---
def predict_claim(model, input_df):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    return prediction[0], prediction_proba[0]

# --- Navigation ---
st.sidebar.title("🚗 Navigation")
page = st.sidebar.selectbox("Select Page", ["🏠 Home","📊 Dashboard", "🔮 Predict Claim"])
#=================================
# HOME PAGE
#=================================
if page == "🏠 Home":
    st.title("Car Insurance Claim Prediction System")
    st.markdown("""
    ### Welcome to the Insurance Analytics Platform
    
    This application helps predict whether a policyholder will file an insurance claim based on various factors.
    
    #### Features:
    - 📊 **Interactive Dashboard**: Explore data insights and patterns
    - 🔮 **Claim Prediction**: Predict claim probability for individual policies
    - 📈 **Model Performance**: View feature importances and model metrics
    
    #### Navigation:
    - Use the sidebar to navigate between pages
    - **Dashboard**: View comprehensive analytics and visualizations
    - **Predict Claim**: Make predictions for new policies
    """)
# ================================
# DASHBOARD PAGE
# ================================
elif page == "📊 Dashboard":
    st.title("📊 Insurance Analytics Dashboard")
    
    # Load data
    df, test_df, importances_df, data_source = load_data()
    
    if df is None:
        st.error("Unable to load data. Please check if train.csv or test.csv exists.")
        st.stop()
    
    st.info(f"📁 Data loaded from: **{data_source}** ({len(df):,} records)")
    
    # Sidebar filters
    st.sidebar.subheader("🔍 Filters")
    
    # Add segment filter if available
    filtered_df = df.copy()
    if 'segment' in df.columns:
        segments = df['segment'].unique().tolist()
        selected_segments = st.sidebar.multiselect(
            "Select Segments", 
            segments, 
            default=segments
        )
        if selected_segments:
            filtered_df = filtered_df[filtered_df['segment'].isin(selected_segments)]
    
    # ===== SECTION 1: KPI OVERVIEW =====
    st.header("📌 Key Performance Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📋 Total Policies", f"{len(filtered_df):,}")
    
    with col2:
        if 'is_claim' in filtered_df.columns:
            claim_rate = filtered_df['is_claim'].mean() * 100
            st.metric("📊 Claim Rate", f"{claim_rate:.2f}%")
        else:
            st.metric("📊 Claim Rate", "N/A")
    
    with col3:
        if 'age_of_policyholder' in filtered_df.columns:
            avg_age = filtered_df['age_of_policyholder'].mean()
            st.metric("👤 Avg Age", f"{avg_age:.1f}")
        else:
            st.metric("👤 Avg Age", "N/A")
    
    with col4:
        if 'age_of_car' in filtered_df.columns:
            avg_car_age = filtered_df['age_of_car'].mean()
            st.metric("🚗 Avg Car Age", f"{avg_car_age:.1f} yrs")
        else:
            st.metric("🚗 Avg Car Age", "N/A")
    
    with col5:
        if 'ncap_rating' in filtered_df.columns:
            avg_ncap = filtered_df['ncap_rating'].mean()
            st.metric("⭐ Avg NCAP", f"{avg_ncap:.2f}")
        else:
            st.metric("⭐ Avg NCAP", "N/A")
    
    st.divider()
    
    # ===== SECTION 2: CLAIM DISTRIBUTION =====
    if 'is_claim' in filtered_df.columns:
        st.header("🎯 Claim Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            claim_counts = filtered_df['is_claim'].value_counts()
            fig_pie = px.pie(
                values=claim_counts.values,
                names=['No Claim', 'Claim'],
                title='Claim vs No Claim Distribution',
                color_discrete_sequence=['#4ECDC4', '#FF6B6B'],
                hole=0.4
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_bar = go.Figure(data=[
                go.Bar(name='Count', 
                       x=['No Claim', 'Claim'], 
                       y=claim_counts.values,
                       marker_color=['#4ECDC4', '#FF6B6B'],
                       text=claim_counts.values,
                       textposition='auto')
            ])
            fig_bar.update_layout(title='Claim Counts', showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        st.divider()
    
    # ===== SECTION 3: VEHICLE SEGMENT ANALYSIS =====
    if 'segment' in filtered_df.columns and 'is_claim' in filtered_df.columns:
        st.header("🚗 High-Value Analysis: Claim Rate by Segment")
        
        claim_by_segment = filtered_df.groupby('segment').agg({
            'is_claim': ['mean', 'count']
        }).reset_index()
        claim_by_segment.columns = ['Segment', 'Claim Rate', 'Policy Count']
        claim_by_segment['Claim Rate'] = claim_by_segment['Claim Rate'] * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_segment = px.bar(
                claim_by_segment,
                x='Segment',
                y='Claim Rate',
                title='Claim Rate by Car Segment (%)',
                color='Claim Rate',
                color_continuous_scale='RdYlGn_r',
                text='Claim Rate'
            )
            fig_segment.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig_segment, use_container_width=True)
            
            # Business insight
            highest_risk_segment = claim_by_segment.loc[claim_by_segment['Claim Rate'].idxmax(), 'Segment']
            st.warning(f"⚠️ **Highest Risk**: Segment **{highest_risk_segment}** has the highest claim rate")
        
        with col2:
            fig_count = px.bar(
                claim_by_segment,
                x='Segment',
                y='Policy Count',
                title='Number of Policies by Segment',
                color='Policy Count',
                color_continuous_scale='Blues',
                text='Policy Count'
            )
            fig_count.update_traces(textposition='outside')
            st.plotly_chart(fig_count, use_container_width=True)
        
        st.divider()
    
    # ===== SECTION 4: TOP RISKY MAKES (NEW - BUSINESS CRITICAL) =====
    if 'make' in filtered_df.columns and 'is_claim' in filtered_df.columns:
        st.header("⚠️ High-Risk Vehicle Manufacturers")
        
        risky_makes = filtered_df.groupby('make').agg({
            'is_claim': ['mean', 'count']
        }).reset_index()
        risky_makes.columns = ['Make', 'Claim Rate', 'Policy Count']
        risky_makes = risky_makes[risky_makes['Policy Count'] >= 10]  # Min sample size
        risky_makes['Claim Rate'] = risky_makes['Claim Rate'] * 100
        risky_makes = risky_makes.sort_values('Claim Rate', ascending=False).head(10)
        
        fig_risky = px.bar(
            risky_makes,
            x='Claim Rate',
            y='Make',
            orientation='h',
            title='Top 10 Vehicle Manufacturers by Claim Rate',
            color='Claim Rate',
            color_continuous_scale='Reds',
            hover_data=['Policy Count'],
            text='Claim Rate'
        )
        fig_risky.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_risky.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_risky, use_container_width=True)
        
        st.info("💡 **Business Action**: Consider premium adjustments for high-risk manufacturers")
        st.divider()
    
    # ===== SECTION 5: DEMOGRAPHIC ANALYSIS =====
    if 'age_of_policyholder' in filtered_df.columns and 'is_claim' in filtered_df.columns:
        st.header("👥 Demographic Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_age = px.histogram(
                filtered_df,
                x='age_of_policyholder',
                color='is_claim',
                title='Policyholder Age Distribution by Claim Status',
                barmode='overlay',
                opacity=0.7,
                color_discrete_map={0: '#4ECDC4', 1: '#FF6B6B'},
                labels={'is_claim': 'Claimed', 'age_of_policyholder': 'Age'}
            )
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            fig_box_age = px.box(
                filtered_df,
                x='is_claim',
                y='age_of_policyholder',
                title='Age Distribution Comparison',
                color='is_claim',
                color_discrete_map={0: '#4ECDC4', 1: '#FF6B6B'},
                labels={'is_claim': 'Claim Status', 'age_of_policyholder': 'Age'}
            )
            fig_box_age.update_xaxes(ticktext=['No Claim', 'Claim'], tickvals=[0, 1])
            st.plotly_chart(fig_box_age, use_container_width=True)
        
        st.divider()
    
    # ===== SECTION 6: CAR AGE ANALYSIS =====
    if 'age_of_car' in filtered_df.columns and 'is_claim' in filtered_df.columns:
        st.header("🚙 Vehicle Age Impact on Claims")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_car_age = px.histogram(
                filtered_df,
                x='age_of_car',
                color='is_claim',
                title='Car Age Distribution by Claim Status',
                barmode='overlay',
                opacity=0.7,
                color_discrete_map={0: '#4ECDC4', 1: '#FF6B6B'},
                labels={'is_claim': 'Claimed'}
            )
            st.plotly_chart(fig_car_age, use_container_width=True)
        
        with col2:
            # Claim rate by car age bins
            filtered_df['age_bin'] = pd.cut(filtered_df['age_of_car'], bins=5)
            age_bin_claims = filtered_df.groupby('age_bin')['is_claim'].mean().reset_index()
            age_bin_claims['age_bin'] = age_bin_claims['age_bin'].astype(str)
            age_bin_claims['is_claim'] = age_bin_claims['is_claim'] * 100
            
            fig_age_bin = px.bar(
                age_bin_claims,
                x='age_bin',
                y='is_claim',
                title='Claim Rate by Car Age Range',
                labels={'is_claim': 'Claim Rate (%)', 'age_bin': 'Car Age Range'},
                color='is_claim',
                color_continuous_scale='RdYlGn_r',
                text='is_claim'
            )
            fig_age_bin.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig_age_bin, use_container_width=True)
        
        st.divider()
    
    # ===== SECTION 7: SAFETY FEATURES =====
    if 'airbags' in filtered_df.columns and 'is_claim' in filtered_df.columns:
        st.header("🛡️ Safety Features Impact")
        
        col1, col2 = st.columns(2)
        
        with col1:
            claim_by_airbags = filtered_df.groupby('airbags').agg({
                'is_claim': ['mean', 'count']
            }).reset_index()
            claim_by_airbags.columns = ['Airbags', 'Claim Rate', 'Count']
            claim_by_airbags['Claim Rate'] = claim_by_airbags['Claim Rate'] * 100
            
            fig_airbags = px.bar(
                claim_by_airbags,
                x='Airbags',
                y='Claim Rate',
                title='Claim Rate by Number of Airbags',
                color='Claim Rate',
                color_continuous_scale='RdYlGn_r',
                text='Claim Rate',
                hover_data=['Count']
            )
            fig_airbags.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig_airbags, use_container_width=True)
        
        with col2:
            if 'ncap_rating' in filtered_df.columns:
                claim_by_ncap = filtered_df.groupby('ncap_rating')['is_claim'].mean().reset_index()
                claim_by_ncap['is_claim'] = claim_by_ncap['is_claim'] * 100
                
                fig_ncap = px.line(
                    claim_by_ncap,
                    x='ncap_rating',
                    y='is_claim',
                    title='Claim Rate by NCAP Safety Rating',
                    markers=True,
                    labels={'is_claim': 'Claim Rate (%)', 'ncap_rating': 'NCAP Rating'}
                )
                fig_ncap.update_traces(line_color='#FF6B6B', marker_size=10)
                st.plotly_chart(fig_ncap, use_container_width=True)
        
        # Safety features comparison
        safety_features = ['is_esc', 'is_tpms', 'is_parking_sensors', 
                          'is_parking_camera', 'is_brake_assist']
        available_safety = [col for col in safety_features if col in filtered_df.columns]
        
        if available_safety:
            st.subheader("Safety Feature Impact Comparison")
            safety_impact = []
            for feature in available_safety:
                with_feature = filtered_df[filtered_df[feature] == 1]['is_claim'].mean() * 100
                without_feature = filtered_df[filtered_df[feature] == 0]['is_claim'].mean() * 100
                safety_impact.append({
                    'Feature': feature.replace('is_', '').replace('_', ' ').title(),
                    'With Feature': with_feature,
                    'Without Feature': without_feature,
                    'Risk Reduction': without_feature - with_feature
                })
            
            safety_df = pd.DataFrame(safety_impact)
            
            fig_safety = go.Figure(data=[
                go.Bar(name='With Feature', x=safety_df['Feature'], 
                       y=safety_df['With Feature'], marker_color='#4ECDC4'),
                go.Bar(name='Without Feature', x=safety_df['Feature'], 
                       y=safety_df['Without Feature'], marker_color='#FF6B6B')
            ])
            fig_safety.update_layout(
                title='Claim Rate: With vs Without Safety Features',
                barmode='group',
                yaxis_title='Claim Rate (%)'
            )
            st.plotly_chart(fig_safety, use_container_width=True)
            
            # Show risk reduction
            st.dataframe(
                safety_df.style.format({
                    'With Feature': '{:.2f}%',
                    'Without Feature': '{:.2f}%',
                    'Risk Reduction': '{:.2f}%'
                }).background_gradient(subset=['Risk Reduction'], cmap='RdYlGn'),
                use_container_width=True
            )
        
        st.divider()
    
    # ===== SECTION 8: POLICY TENURE ANALYSIS (NEW - BUSINESS CRITICAL) =====
    if 'policy_tenure' in filtered_df.columns and 'is_claim' in filtered_df.columns:
        st.header("📅 Claim Timing Analysis")
        
        tenure_claims = filtered_df.groupby('policy_tenure').agg({
            'is_claim': ['mean', 'count']
        }).reset_index()
        tenure_claims.columns = ['Policy Tenure', 'Claim Rate', 'Count']
        tenure_claims['Claim Rate'] = tenure_claims['Claim Rate'] * 100
        
        fig_tenure = px.line(
            tenure_claims,
            x='Policy Tenure',
            y='Claim Rate',
            title='Claim Rate Throughout Policy Lifetime',
            markers=True,
            hover_data=['Count']
        )
        fig_tenure.update_traces(line_color='#FF6B6B', marker_size=8)
        st.plotly_chart(fig_tenure, use_container_width=True)
        
        # Business insight
        peak_tenure = tenure_claims.loc[tenure_claims['Claim Rate'].idxmax(), 'Policy Tenure']
        st.warning(f"📊 **Key Insight**: Claims peak at **{peak_tenure}** months into policy period")
        st.info("💡 **Action**: Implement proactive engagement programs around this period")
        
        st.divider()
    
    # ===== SECTION 9: GEOGRAPHIC ANALYSIS =====
    if 'area_cluster' in filtered_df.columns and 'is_claim' in filtered_df.columns:
        st.header("🗺️ Geographic Risk Distribution")
        
        geo_risk = filtered_df.groupby('area_cluster').agg({
            'is_claim': ['mean', 'count']
        }).reset_index()
        geo_risk.columns = ['Area', 'Claim Rate', 'Policies']
        geo_risk['Claim Rate'] = geo_risk['Claim Rate'] * 100
        geo_risk = geo_risk.sort_values('Claim Rate', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_geo_bar = px.bar(
                geo_risk.head(15),
                x='Area',
                y='Claim Rate',
                title='Top 15 Areas by Claim Rate',
                color='Claim Rate',
                color_continuous_scale='RdYlGn_r',
                hover_data=['Policies'],
                text='Claim Rate'
            )
            fig_geo_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig_geo_bar, use_container_width=True)
        
        with col2:
            fig_geo_scatter = px.scatter(
                geo_risk,
                x='Area',
                y='Claim Rate',
                size='Policies',
                title='Claim Rate vs Policy Volume by Area',
                color='Claim Rate',
                color_continuous_scale='RdYlGn_r',
                hover_data=['Policies']
            )
            st.plotly_chart(fig_geo_scatter, use_container_width=True)
        
        st.divider()
    
    # ===== SECTION 10: FEATURE IMPORTANCE =====
    if importances_df is not None:
        st.header("🎯 Model Feature Importance")
        
        importances_df = importances_df.sort_values(by='Importance', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_imp = px.bar(
                importances_df.head(15),
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 15 Most Important Features',
                color='Importance',
                color_continuous_scale='viridis',
                text='Importance'
            )
            fig_imp.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_imp, use_container_width=True)
        
        with col2:
            st.subheader("Feature Statistics")
            st.dataframe(
                importances_df.head(15)[['Feature', 'Importance']].reset_index(drop=True).style.format({
                    'Importance': '{:.4f}'
                }).background_gradient(subset=['Importance'], cmap='viridis'),
                height=500,
                use_container_width=True
            )
    
    st.divider()
    
    # ===== DATA PREVIEW =====
    with st.expander("📋 View Raw Data Sample"):
        st.dataframe(filtered_df.head(50), use_container_width=True)
        
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name="insurance_data.csv",
            mime="text/csv"
        )

# ================================
# PREDICTION PAGE
# ================================
elif page == "🔮 Predict Claim":
    st.title("🔮 Insurance Claim Prediction")
    st.write("Enter policy and customer details to predict claim probability")
    
    # Load data and model
    df, test_df, importances_df, data_source = load_data()
    
    if test_df is None:
        st.error("Unable to load test data.")
        st.stop()
    
    try:
        model = load_model('catboost_model.pkl')
        encoder = load_encoder('le_encoder.pkl')
    except Exception as e:
        st.error(f"Failed to load model or encoder: {e}")
        st.stop()
    
    # Create input form
    st.subheader("📝 Enter Policy Details")
    
    user_input = {}
    cols = [col for col in test_df.columns if col != 'is_claim']
    
    # Organize inputs in 3 columns
    col1, col2, col3 = st.columns(3)
    columns = [col1, col2, col3]
    
    for idx, col in enumerate(cols):
        with columns[idx % 3]:
            if test_df[col].dtype == 'object':
                unique_vals = test_df[col].dropna().unique()
                user_input[col] = st.selectbox(
                    f"{col.replace('_', ' ').title()}:",
                    unique_vals,
                    key=col
                )
            else:
                min_val = float(test_df[col].min())
                max_val = float(test_df[col].max())
                mean_val = float(test_df[col].mean())
                
                user_input[col] = st.number_input(
                    f"{col.replace('_', ' ').title()}:",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    key=col
                )
    
    st.divider()
    
    # Predict button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button("🔮 Predict Claim", type="primary", use_container_width=True)
    
    if predict_button:
        input_df = pd.DataFrame([user_input])
        
        # Encode categorical variables
        for col in input_df.columns:
            if input_df[col].dtype == 'object':
                try:
                    input_df[col] = encoder[col].transform(input_df[col])
                except Exception as e:
                    st.error(f"Error encoding {col}: {e}")
                    st.stop()
        
        # Make prediction
        try:
            prediction, prediction_proba = predict_claim(model, input_df)
            
            st.success("✅ Prediction Complete!")
            st.divider()
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Prediction Result",
                    "CLAIM" if prediction == 1 else "NO CLAIM",
                    delta="High Risk" if prediction == 1 else "Low Risk",
                    delta_color="inverse"
                )
            
            with col2:
                claim_prob = prediction_proba[1] * 100
                st.metric(
                    "Claim Probability",
                    f"{claim_prob:.2f}%",
                    delta=f"{claim_prob - 50:.2f}% from neutral"
                )
            
            with col3:
                no_claim_prob = prediction_proba[0] * 100
                st.metric(
                    "No Claim Probability",
                    f"{no_claim_prob:.2f}%"
                )
            
            # Probability visualization
            st.subheader("📊 Prediction Confidence")
            fig_proba = go.Figure()
            
            fig_proba.add_trace(go.Bar(
                x=['No Claim', 'Claim'],
                y=[prediction_proba[0], prediction_proba[1]],
                marker_color=['#4ECDC4', '#FF6B6B'],
                text=[f'{prediction_proba[0]:.2%}', f'{prediction_proba[1]:.2%}'],
                textposition='auto'
            ))
            
            fig_proba.update_layout(
                yaxis_title='Probability',
                yaxis=dict(range=[0, 1]),
                showlegend=False
            )
            
            st.plotly_chart(fig_proba, use_container_width=True)
            
            # Confidence assessment
            confidence = abs(prediction_proba[1] - 0.5) * 2
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Risk Assessment")
                if claim_prob > 70:
                    st.error("⚠️ **High Risk Policy**")
                    st.write("- Strong likelihood of claim")
                    st.write("- Consider premium adjustment")
                elif claim_prob > 50:
                    st.warning("⚡ **Moderate Risk Policy**")
                    st.write("- Average claim probability")
                    st.write("- Standard premium applicable")
                else:
                    st.success("✅ **Low Risk Policy**")
                    st.write("- Low likelihood of claim")
                    st.write("- Consider offering discounts")
            
            with col2:
                st.subheader("🔍 Model Confidence")
                st.metric("Confidence Score", f"{confidence*100:.1f}%")
                
                if confidence > 0.8:
                    st.success("🎯 High confidence - Proceed with automated processing")
                elif confidence > 0.5:
                    st.info("⚡ Moderate confidence - Standard review recommended")
                else:
                    st.warning("⚠️ Low confidence - Manual review required")
            
            # Business recommendations
            st.subheader("💼 Business Recommendations")
            if claim_prob > 60:
                st.write("✓ Increase premium by 10-20%")
                st.write("✓ Request additional documentation")
                st.write("✓ Assign to experienced claims handler")
            else:
                st.write("✓ Eligible for loyalty discounts")
                st.write("✓ Good candidate for upselling")
                st.write("✓ Fast-track approval recommended")
            
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# Footer
st.sidebar.divider()
st.sidebar.info("📧 Contact: support@insurance.com")
st.sidebar.info("🔒 Secure & Confidential")