import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Spot the Bias",
    page_icon="📊",
    layout="wide"
)

# Title and Introduction
st.title("📊 Spot the Bias: Analysis of Gender Disparity in Promotion Rates")
# st.markdown(
#     """
#     🚀 <span style="color: #29B6F6; font-size: 25px;"><strong>Welcome!</strong></span>  
#     This tool provides <span style="color: #FF6347;">**quick and valuable insights**</span> into potential gender biases in promotions within your organization.  
#     Whether you're analyzing overall trends or focusing on a specific department or team, our tool helps you spot disparities and make data-driven decisions.

#     <span style="color: #29B6F6; font-size: 15px;"><strong>How It Works?</strong></span>  
#     - **Sample Data**: By default, it shows the analysis for sample dataset loaded by us for demo purposes.  
#     - **Upload Your Data**: Simply click on "Upload Your Data" to upload your own data to get started.  
#     - **Quick Analysis**: The tool analyzes key metrics, like promotion rates, gender distribution, and more, to help you uncover biases in seconds.

#     🚨 <span style="color: #FF6347; font-size: 18px; font-weight: bold;">**Must Go Through Column Details:**</span>  
#     Please ensure your data aligns with the following column names to ensure smooth analysis:
    
#     - `gender` *(Required)*: 'm' for male, 'f' for female.
#     - `is_promoted` *(Required)*: 1 for promoted, 0 for not promoted.
#     - `employee_id` *(Optional)*: Include if you like, but not needed for analysis.
#     - `previous_year_rating` *(Optional)*: Employee rating from the previous year.
#     - `no_of_trainings` *(Optional)*: Number of training sessions completed.
#     - `age` *(Optional)*: Employee age.
#     - `awards_won?` *(Optional)*: Number of awards won (0, 1, or 2).
#     - `department` *(Optional but recommended)*: Helps with department-wise or region-wise analysis.
#     - `length_of_service` *(Optional)*: Employee's length of service in the organization.
#     - `avg_training_score` *(Optional)*: Average training score.

#     📌 **Note**:  
#     This tool is a **prototype** designed for educational and demonstration purposes. It is not intended for business-level deployment or critical decision-making. Feel free to explore!
    
#     **Get started by uploading your data and analyzing it instantly!**
#     """,
#     unsafe_allow_html=True
# )
st.markdown(
    """
    🚀 <span style="color: #29B6F6; font-size: 25px;"><strong>Welcome to my Quick Insights Tool!</strong></span>  
    Ready to discover if there's a gender bias lurking in your organization's promotion rates? With just a few clicks, my tool gives you <span style="color: #FF6347;">**quick insights**</span> into how men and women are promoted (or not) across different teams or departments. Let's dive in and see the trends!

   <span style="color: #29B6F6; font-size: 18px;"><strong>✨ Let's Get You Quick Results! Here's How:</strong></span>  
    - **Start with Sample Data**: We’ve got a demo dataset ready! Just scroll to see how the tool analyzes the data.  
    - **Upload Your Own Data**: Have your own data? Simply click "Upload Your Data".  

   
    🚨 <span style="color: #FF6347; font-size: 18px; font-weight: bold;">**Don't Skip These Column Details:**</span>  
    For the best experience, make sure your data column names matches the following column names:
    
    - `gender` *(Required)*: 'm' for male, 'f' for female.
    - `is_promoted` *(Required)*: 1 for promoted, 0 for not promoted.
    - `employee_id` *(Optional)*: You can include it, but we don’t need it for analysis.
    - `previous_year_rating` *(Optional)*: Employee rating from last year.
    - `no_of_trainings` *(Optional)*: Number of trainings completed.
    - `age` *(Optional)*: How old is the employee?
    - `awards_won?` *(Optional)*: Awards count (0, 1, or 2).
    - `department` *(Optional but recommended)*: Great for digging into department or region-wise trends.
    - `length_of_service` *(Optional)*: How long the employee has been with the company.
    - `avg_training_score` *(Optional)*: Average training score the employee achieved.

    """,
    unsafe_allow_html=True
)


# Sidebar for data selection
st.sidebar.header("Data Options")
data_option = st.sidebar.radio(
    "Choose Data Source:",
    ("Use Sample Data", "Upload Your Data")
)

# Function to load sample data
@st.cache_data
def load_sample_data():
    return pd.read_csv("sample_data.csv")  # Replace with the path to your sample CSV file

if data_option == "Use Sample Data":
    data = load_sample_data()
    st.sidebar.success("Using sample data.")
else:
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.sidebar.success("File uploaded successfully!")
    else:
        st.info("Please upload a CSV file or use the sample data.")
        data = load_sample_data()  # Default to sample data until file is uploaded

# Validate data
required_columns = [
    "gender", "is_promoted"
]
if not all(col in data.columns for col in required_columns):
    st.error(
        f"Your data must contain the following columns: {', '.join(required_columns)}"
    )
else:
    available_columns = data.columns.tolist()
    # Preprocess Data
    data['is_promoted'] = data['is_promoted'].astype(int)
    if "awards_won?" in available_columns:
        data['awards_won?'] = data['awards_won?'].astype(int)
    if "previous_year_rating" in available_columns:
        data['previous_year_rating'] = data['previous_year_rating'].fillna(0).astype(int)
        data['previous_year_rating'] = data['previous_year_rating'].astype(int)

    # Display Data Preview
    st.subheader("Data Preview")
    st.write(data.head())

    # Key Metrics
    st.subheader("📈 Key Metrics")
    total_promotion_rate = data['is_promoted'].mean() * 100
    st.metric("Overall Promotion Rate", f"{total_promotion_rate:.2f}%")

    gender_promotion_rate = data.groupby("gender")["is_promoted"].mean() * 100
    promotion_difference = (
        gender_promotion_rate["m"] - gender_promotion_rate["f"]
    )
    st.metric(
        label="Promotion Rate Difference (Men vs. Women)",
        value=f"{promotion_difference:.2f}%"
    )

    # Visualizations
    st.subheader("📊 Visualizations")

    # Color scheme and category order
    color_map = {"f": "#29B6F6", "m": "#FF7043"}  # Light blue for female and coral orange for male
    category_order = {"gender": ["f", "m"]}

    # 1. Gender vs. Promotion Rate
    gender_promo_chart = px.bar(
        gender_promotion_rate.reset_index(),
        x="gender",
        y="is_promoted",
        color="gender",
        color_discrete_map=color_map,
        category_orders=category_order,
        title="Promotion Rate by Gender",
        labels={"is_promoted": "Promotion Rate (%)", "gender": "Gender"}
    )
    st.plotly_chart(gender_promo_chart, use_container_width=True)

    # 2. Department-Wise Analysis
    if "department" in available_columns:
        st.subheader("Department-Wise Analysis")
        dept_gender_promo = data.groupby(["department", "gender"])["is_promoted"].mean().reset_index()
        dept_gender_promo_chart = px.bar(
            dept_gender_promo,
            x="department",
            y="is_promoted",
            color="gender",
            color_discrete_map=color_map,
            category_orders=category_order,
            barmode="group",
            title="Promotion Rate by Department and Gender",
            labels={"is_promoted": "Promotion Rate (%)", "department": "Department"}
        )
        st.plotly_chart(dept_gender_promo_chart, use_container_width=True)

        # Interactive Department Selection
        selected_department = st.selectbox(
            "Select a Department to Analyze in Detail:",
            options=sorted(data["department"].unique())
        )
        if selected_department:
            dept_data = data[data["department"] == selected_department]
            dept_gender_chart = px.bar(
                dept_data.groupby("gender")["is_promoted"].mean().reset_index(),
                x="gender",
                y="is_promoted",
                color="gender",
                color_discrete_map=color_map,
                category_orders=category_order,
                title=f"Promotion Rate in {selected_department}",
                labels={"is_promoted": "Promotion Rate (%)"}
            )
            st.plotly_chart(dept_gender_chart, use_container_width=True)

            # Metric Distributions for Promoted Employees (Men vs Women) in selected department
            st.subheader(f'📈 Metric Distributions "{selected_department}"')

        # Filter data for promoted employees
        promoted_dept_data = dept_data[dept_data["is_promoted"] == 1]

        # A.
        # Group data by gender and awards_won?
        # Group data by gender and awards_won?
        if "awards_won?" in available_columns:
            awards_distribution = promoted_dept_data.groupby(['gender', 'awards_won?']).size().reset_index(name='count')
            
            # Normalize the 'count' by gender (calculate the percentage of each award won group for each gender)
            awards_distribution['total'] = awards_distribution.groupby('gender')['count'].transform('sum')
            awards_distribution['percentage'] = (awards_distribution['count'] / awards_distribution['total']) * 100
            
            # Get the maximum value of 'awards_won?'
            max_awards = max(promoted_dept_data['awards_won?'])
            
            # Create the bar plot with normalized percentages
            awards_chart = px.bar(
                awards_distribution,
                x="awards_won?",
                y="percentage",  # Use percentage for normalized values
                color="gender",
                color_discrete_map=color_map,
                barmode="group",
                title=f'"awards_won?" in {selected_department}',
                labels={
                    "awards_won?": f"Number of Awards Won: 0 to {max_awards}",  
                    "percentage": "Percentage (%)",
                    "gender": "Gender"
                }
            )
            # Display the plot
            st.plotly_chart(awards_chart, use_container_width=True)


        # Group data by gender and previous_year_rating
        if "previous_year_rating" in available_columns:
            rating_distribution = promoted_dept_data.groupby(['gender', 'previous_year_rating']).size().reset_index(name='count')
            
            # Normalize the 'count' by gender (calculate the percentage of each rating group for each gender)
            rating_distribution['total'] = rating_distribution.groupby('gender')['count'].transform('sum')
            rating_distribution['percentage'] = (rating_distribution['count'] / rating_distribution['total']) * 100
            
            # Get the maximum value of 'previous_year_rating'
            max_rating = max(promoted_dept_data['previous_year_rating'])
            
            # Create the bar plot with normalized percentages
            rating_chart = px.bar(
                rating_distribution,
                x="previous_year_rating",
                y="percentage",  # Use percentage for normalized values
                color="gender",
                color_discrete_map=color_map,
                barmode="group",
                title=f'"previous_year_rating" in {selected_department}',
                labels={
                    "previous_year_rating": f"Previous Year Rating: 0 to {max_rating}",  
                    "percentage": "Percentage (%)",
                    "gender": "Gender"
                }
            )
            # Display the plot
            st.plotly_chart(rating_chart, use_container_width=True)


        # Group data by gender and no_of_trainings
        if "no_of_trainings" in available_columns and max(promoted_dept_data["no_of_trainings"]) <= 10:
            trainings_distribution = promoted_dept_data.groupby(['gender', 'no_of_trainings']).size().reset_index(name='count')
            
            # Normalize the 'count' by gender (calculate the percentage of each 'no_of_trainings' value for each gender)
            trainings_distribution['total'] = trainings_distribution.groupby('gender')['count'].transform('sum')
            trainings_distribution['percentage'] = (trainings_distribution['count'] / trainings_distribution['total']) * 100
            
            # Get the maximum value of 'no_of_trainings'
            max_trainings = max(promoted_dept_data['no_of_trainings'])
            
            # Create the bar plot with normalized percentages
            trainings_chart = px.bar(
                trainings_distribution,
                x="no_of_trainings",
                y="percentage",  # Use percentage for normalized values
                color="gender",
                color_discrete_map=color_map,
                barmode="group",
                title=f'"no_of_trainings" in {selected_department}',
                labels={
                    "no_of_trainings": f"Number of Trainings: 0 to {max_trainings}",  
                    "percentage": "Percentage (%)",
                    "gender": "Gender"
                }
            )
            # Display the plot
            st.plotly_chart(trainings_chart, use_container_width=True)


        # B.
        for metric in ["avg_training_score","length_of_service", "age"]:
            if metric in available_columns:
                # Boxplot for promoted men vs women
                fig = px.box(
                    promoted_dept_data,
                    x="gender",
                    y=metric,
                    color="gender",
                    color_discrete_map=color_map,
                    category_orders=category_order,
                    title=f'"{metric}" distribution for {selected_department}',
                    labels={"gender": "Gender", metric: metric}
                )
                st.plotly_chart(fig, use_container_width=True)

    # 4. Correlation Analysis
    st.subheader("🔍 Correlation Analysis")

    # Select only numeric columns for correlation
    numeric_data = data.select_dtypes(include=['number'])

    # Calculate correlation
    correlation = numeric_data.corr()["is_promoted"].drop("is_promoted")

    # Plot the correlation
    fig, ax = plt.subplots()
    sns.barplot(x=correlation.values, y=correlation.index, palette="coolwarm", ax=ax)
    ax.set_title("Correlation of Metrics with Promotions")
    st.pyplot(fig)

    # Recommendations
    st.subheader("💡 Recommendations")
    st.markdown(
        """
        - **Enhance Training**: Improve training programs for underrepresented groups.
        - **Address Bias**: Review promotion criteria to ensure fairness.
        - **Mentorship Programs**: Encourage mentorship for early-career employees.
        """
    )

    # Quick Heads Up
    st.subheader("📌 **Quick Heads Up**")
    st.markdown(
        """Thanks for exploring! Just a reminder, this tool is a **prototype**—designed for fun learning and data exploration. It's not a business-level solution, so while it's great for educational purposes, it shouldn't be used for making major business decisions. 
        """
    )