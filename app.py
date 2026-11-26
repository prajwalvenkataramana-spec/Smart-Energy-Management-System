import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Smart Energy Management System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# 2. PROFESSIONAL UI STYLING
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #F4F7FC;
    }

    [data-testid="stSidebar"] {
        background-color: #14213D;
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF;
    }

    h1, h2, h3 {
        color: #172554;
    }

    [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #E2E8F0;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    div.stButton > button {
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: #1D4ED8;
        color: white;
    }

    [data-testid="stDataFrame"] {
        background-color: white;
        border-radius: 10px;
    }

    .main-subtitle {
        color: #64748B;
        font-size: 15px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 3. LOAD DATASET AND MODEL
# --------------------------------------------------

DATA_PATH = "dataset/mall_energy.csv"
MODEL_PATH = "model/energy_model.pkl"

REQUIRED_COLUMNS = [
    "Temperature",
    "Occupancy",
    "AC_Usage",
    "Lighting_Usage",
    "Energy_Consumption"
]

FEATURE_COLUMNS = [
    "Temperature",
    "Occupancy",
    "AC_Usage",
    "Lighting_Usage"
]

if not os.path.exists(DATA_PATH):
    st.error("Dataset file not found: dataset/mall_energy.csv")
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found: model/energy_model.pkl")
    st.stop()

try:
    if "mall_data" not in st.session_state:
        st.session_state.mall_data = pd.read_csv(DATA_PATH)

    data = st.session_state.mall_data.copy()

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        st.error(f"Missing dataset columns: {missing_columns}")
        st.stop()

    for column in REQUIRED_COLUMNS:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    if data[REQUIRED_COLUMNS].isnull().any().any():
        st.error("The dataset contains missing or invalid values.")
        st.stop()

    if data.empty:
        st.error("The dataset is empty.")
        st.stop()

    model = joblib.load(MODEL_PATH)

except Exception as error:
    st.error(f"Unable to load project data: {error}")
    st.stop()

# --------------------------------------------------
# 4. SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:

    st.markdown(
        "<h1 style='text-align:center;'>⚡</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h3 style='text-align:center; color:white;'>
        Smart Energy Management System
        </h3>
        <p style='text-align:center; color:#CBD5E1;'>
        Shopping Malls
        </p>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Data Analysis",
            "Predictions",
            "Recommendations",
            "Data Management",
            "Reports",
            "About"
        ]
    )

    st.divider()

    st.markdown("### ⚙️ Settings")

    electricity_rate = st.number_input(
        "Electricity rate (₹ per unit)",
        min_value=0.0,
        value=8.0,
        step=0.5
    )

    st.caption(
        "Enter your applicable electricity rate "
        "to estimate energy costs."
    )

    st.divider()

    st.caption("Software-based energy management system")

# --------------------------------------------------
# 5. COMMON CALCULATIONS
# --------------------------------------------------

total_energy = data["Energy_Consumption"].sum()

average_energy = data["Energy_Consumption"].mean()

maximum_energy = data["Energy_Consumption"].max()

minimum_energy = data["Energy_Consumption"].min()

estimated_cost = total_energy * electricity_rate

peak_limit = (
    average_energy + data["Energy_Consumption"].std()
)

peak_records = data[
    data["Energy_Consumption"] > peak_limit
]

# --------------------------------------------------
# 6. DASHBOARD PAGE
# --------------------------------------------------

if page == "Dashboard":

    st.title("⚡ Smart Energy Management System for Shopping Malls")

    st.markdown(
        '<p class="main-subtitle">'
        'Shopping Mall Energy Monitoring Dashboard'
        '</p>',
        unsafe_allow_html=True
    )

    st.divider()

    st.header("Energy Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Energy Consumption",
        f"{total_energy:,.2f} units"
    )

    col2.metric(
        "Average Consumption",
        f"{average_energy:,.2f} units"
    )

    col3.metric(
        "Maximum Consumption",
        f"{maximum_energy:,.2f} units"
    )

    col4.metric(
        "Estimated Electricity Cost",
        f"₹{estimated_cost:,.2f}"
    )

    st.divider()

    left, right = st.columns([1, 1])

    with left:

        st.subheader("Energy Consumption Analysis")

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.plot(
            data["Energy_Consumption"].values,
            marker="o",
            linewidth=2
        )

        ax.set_xlabel("Data Records")
        ax.set_ylabel("Energy Consumption (units)")
        ax.set_title("Energy Consumption Trend")
        ax.grid(alpha=0.3)

        st.pyplot(fig)
        plt.close(fig)

    with right:

        st.subheader("Energy Usage Distribution")

        usage_values = [
            data["AC_Usage"].mean(),
            data["Lighting_Usage"].mean()
        ]

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.bar(
            ["AC Usage", "Lighting Usage"],
            usage_values
        )

        ax.set_ylabel("Average Usage")
        ax.set_title("Average AC and Lighting Usage")
        ax.grid(axis="y", alpha=0.3)

        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    st.subheader("Shopping Mall Energy Data")

    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Peak Demand Monitoring")

    if len(peak_records) > 0:

        st.warning(
            f"{len(peak_records)} record(s) are above "
            "the calculated peak threshold."
        )

        st.dataframe(
            peak_records,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.success(
            "No records are above the calculated peak threshold."
        )

    st.caption(
        "Peak threshold is calculated as the dataset mean "
        "plus one standard deviation. This is a basic "
        "statistical indicator, not a utility demand alert."
    )

# --------------------------------------------------
# 7. DATA ANALYSIS PAGE
# --------------------------------------------------

elif page == "Data Analysis":

    st.title("📊 Data Analysis")

    st.write(
        "Explore the shopping mall energy dataset "
        "and its statistical characteristics."
    )

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    col1.metric("Total Records", len(data))

    col2.metric(
        "Total Features",
        len(data.columns)
    )

    st.subheader("Statistical Summary")

    st.dataframe(
        data.describe(),
        use_container_width=True
    )

    st.subheader("Correlation Analysis")

    correlation = data[REQUIRED_COLUMNS].corr()

    fig, ax = plt.subplots(figsize=(8, 5))

    image = ax.imshow(
        correlation,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    ax.set_xticks(range(len(correlation.columns)))
    ax.set_yticks(range(len(correlation.columns)))

    ax.set_xticklabels(
        correlation.columns,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(correlation.columns)

    for i in range(len(correlation.columns)):
        for j in range(len(correlation.columns)):
            ax.text(
                j, i,
                f"{correlation.iloc[i, j]:.2f}",
                ha="center",
                va="center"
            )

    fig.colorbar(image, ax=ax)

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Missing Values")

    st.dataframe(
        data.isnull().sum().rename("Missing Values"),
        use_container_width=True
    )

# --------------------------------------------------
# 8. MACHINE LEARNING PREDICTION PAGE
# --------------------------------------------------

elif page == "Predictions":

    st.title("🔮 Energy Consumption Prediction")

    st.write(
        "Use the trained Linear Regression model "
        "to estimate energy consumption."
    )

    st.info(
        "Enter the operating conditions to predict "
        "the expected energy consumption."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            temperature = st.number_input(
                "Temperature (°C)",
                min_value=0.0,
                value=30.0
            )

            occupancy = st.number_input(
                "Occupancy",
                min_value=0,
                value=200
            )

        with col2:

            ac_usage = st.number_input(
                "AC Usage",
                min_value=0.0,
                value=75.0
            )

            lighting_usage = st.number_input(
                "Lighting Usage",
                min_value=0.0,
                value=70.0
            )

        predict_button = st.form_submit_button(
            "⚡ Predict Energy Consumption",
            use_container_width=True
        )

    if predict_button:

        input_data = pd.DataFrame([{
            "Temperature": temperature,
            "Occupancy": occupancy,
            "AC_Usage": ac_usage,
            "Lighting_Usage": lighting_usage
        }])

        try:

            prediction = model.predict(input_data)[0]

            st.success("Prediction completed successfully!")

            st.metric(
                "Predicted Energy Consumption",
                f"{prediction:.2f} units"
            )

            st.caption(
                "This is a model estimate based on the "
                "training dataset, not a measured reading."
            )

        except Exception as error:

            st.error(f"Prediction failed: {error}")

# --------------------------------------------------
# 9. ENERGY RECOMMENDATIONS PAGE
# --------------------------------------------------

elif page == "Recommendations":

    st.title("🌱 Energy Saving Recommendations")

    st.write(
        "Recommendations are generated using "
        "the energy usage averages in the current dataset."
    )

    average_ac = data["AC_Usage"].mean()

    average_lighting = data["Lighting_Usage"].mean()

    high_ac = data[
        data["AC_Usage"] > average_ac
    ]

    high_lighting = data[
        data["Lighting_Usage"] > average_lighting
    ]

    st.subheader("Energy Usage Overview")

    col1, col2 = st.columns(2)

    col1.metric(
        "Average AC Usage",
        f"{average_ac:.2f}"
    )

    col2.metric(
        "Average Lighting Usage",
        f"{average_lighting:.2f}"
    )

    st.divider()

    st.subheader("Suggested Actions")

    if len(high_ac) > 0:

        st.warning(
            "Consider reviewing AC usage in records "
            "where consumption is above the dataset average."
        )

    else:

        st.success(
            "AC usage is not above the dataset average."
        )

    if len(high_lighting) > 0:

        st.warning(
            "Review lighting usage in records "
            "where consumption is above the dataset average."
        )

    else:

        st.success(
            "Lighting usage is not above the dataset average."
        )

    if len(high_ac) == 0 and len(high_lighting) == 0:

        st.success(
            "Continue monitoring energy consumption."
        )

    st.caption(
        "These are basic rule-based recommendations, "
        "not recommendations generated by a separate AI model."
    )

# --------------------------------------------------
# 10. DATA MANAGEMENT PAGE
# --------------------------------------------------

elif page == "Data Management":

    st.title("📁 Energy Data Management")

    st.write(
        "Import energy records using CSV files "
        "or enter new records manually."
    )

    st.subheader("Upload CSV Dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            uploaded_data = pd.read_csv(uploaded_file)

            missing_columns = [
                column for column in REQUIRED_COLUMNS
                if column not in uploaded_data.columns
            ]

            if missing_columns:

                st.error(
                    f"Your file is missing these columns: "
                    f"{missing_columns}"
                )

            else:

                for column in REQUIRED_COLUMNS:
                    uploaded_data[column] = pd.to_numeric(
                        uploaded_data[column],
                        errors="coerce"
                    )

                if uploaded_data[REQUIRED_COLUMNS].isnull().any().any():

                    st.error(
                        "The uploaded file contains missing "
                        "or invalid numeric values."
                    )

                else:

                    if st.button("Use Uploaded Dataset"):

                        st.session_state.mall_data = (
                            uploaded_data[REQUIRED_COLUMNS].copy()
                        )

                        st.success(
                            "Uploaded dataset is now active "
                            "in this dashboard session."
                        )

                        st.rerun()

        except Exception as error:

            st.error(f"Unable to read CSV file: {error}")

    st.divider()

    st.subheader("Manual Energy Data Entry")

    with st.form("manual_data_form"):

        col1, col2 = st.columns(2)

        with col1:

            new_temperature = st.number_input(
                "Temperature",
                min_value=0.0,
                value=30.0
            )

            new_occupancy = st.number_input(
                "Occupancy",
                min_value=0,
                value=200
            )

        with col2:

            new_ac = st.number_input(
                "AC Usage",
                min_value=0.0,
                value=75.0
            )

            new_lighting = st.number_input(
                "Lighting Usage",
                min_value=0.0,
                value=70.0
            )

        new_energy = st.number_input(
            "Recorded Energy Consumption",
            min_value=0.0,
            value=200.0
        )

        add_record = st.form_submit_button(
            "Add Energy Record"
        )

    if add_record:

        new_record = pd.DataFrame([{
            "Temperature": new_temperature,
            "Occupancy": new_occupancy,
            "AC_Usage": new_ac,
            "Lighting_Usage": new_lighting,
            "Energy_Consumption": new_energy
        }])

        st.session_state.mall_data = pd.concat(
            [
                st.session_state.mall_data,
                new_record
            ],
            ignore_index=True
        )

        st.success(
            "New energy record added to the current session."
        )

        st.rerun()

    st.caption(
        "Uploaded and manually entered data are kept "
        "in the current application session. "
        "They do not automatically overwrite your original CSV "
        "or retrain the saved model."
    )

    if st.button("Restore Original Sample Dataset"):

        st.session_state.mall_data = pd.read_csv(DATA_PATH)

        st.success("Original sample dataset restored.")

        st.rerun()

# --------------------------------------------------
# 11. REPORTS PAGE
# --------------------------------------------------

elif page == "Reports":

    st.title("📄 Energy Reports")

    st.write(
        "View and download reports from the active dataset."
    )

    st.subheader("Energy Summary")

    report = pd.DataFrame({
        "Metric": [
            "Total Energy Consumption",
            "Average Energy Consumption",
            "Maximum Energy Consumption",
            "Minimum Energy Consumption",
            "Estimated Electricity Cost"
        ],

        "Value": [
            total_energy,
            average_energy,
            maximum_energy,
            minimum_energy,
            estimated_cost
        ]
    })

    st.dataframe(
        report,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download Energy Summary",
        data=report.to_csv(index=False).encode("utf-8"),
        file_name="energy_summary_report.csv",
        mime="text/csv"
    )

    st.divider()

    st.subheader("Download Energy Dataset")

    st.download_button(
        "⬇️ Download Complete Dataset",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="mall_energy_report.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# 12. ABOUT PAGE
# --------------------------------------------------

elif page == "About":

    st.title("ℹ️ About the Project")

    st.header(
        "Smart Energy Management System for Shopping Malls"
    )

    st.write(
        """
        This project is a software-based energy management
        system designed to monitor, analyse and predict
        shopping mall energy consumption.
        """
    )

    st.subheader("Main Technologies")

    st.markdown("""
    - Python
    - Streamlit
    - Pandas
    - NumPy
    - Scikit-learn
    - Matplotlib
    """)

    st.subheader("Machine Learning Algorithm")

    st.write("Linear Regression")

    st.subheader("Main Features")

    st.markdown("""
    - Energy consumption monitoring
    - Dataset analysis
    - Machine Learning predictions
    - Energy-saving recommendations
    - CSV data upload
    - Manual energy data entry
    - Peak-demand indication
    - Energy reports
    """)

    st.info(
        "This project uses manual, uploaded and sample data. "
        "It does not require physical hardware, IoT devices "
        "or sensors."
    )

# --------------------------------------------------
# 13. FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <p style='text-align:center; color:#64748B;'>
    Smart Energy Management System for Shopping Malls
    <br>
    Software-Based Energy Monitoring and Management
    </p>
    """,
    unsafe_allow_html=True
)