import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore

st.set_page_config(
    page_title="Food Nutrition Analytics & Recommendation System",
    page_icon="🥗",
    layout="wide"
)


st.markdown("""
<style>

/* ── Background image ─────────────────────────────────────── */
[data-testid="stAppViewContainer"] {
    background-image: url("https://www.topsdaynurseries.co.uk/wp-content/uploads/2024/06/Building-Healthy-Food-Habits-on-Healthy-Eating-Week-at-Tops-e1725444109248-1.png");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Semi-transparent overlay so content stays readable */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    background: rgba(255, 255, 255, 0.20);
    z-index: 0;
}

/* Keep all content above the overlay */
[data-testid="stAppViewContainer"] > * {
    position: relative;
    z-index: 1;
}

/* ── Sidebar background ───────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: rgba(20, 20, 20, 0.95) !important;
}

/* ── Sidebar nav title ────────────────────────────────────── */
.nav-title {
    color: #ffffff !important;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0.6rem 0 0.8rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.25);
    margin-bottom: 0.5rem;
}

/* ── Sidebar radio — container gap ───────────────────────── */
[data-testid="stSidebar"] .stRadio > div {
    gap: 0.3rem;
}

/* ── Force all label wrappers to be block-level ──────────── */
[data-testid="stSidebar"] .stRadio label {
    display: flex !important;
    align-items: center;
    width: 100%;
    padding: 0.55rem 1rem !important;
    border-radius: 8px;
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.18s ease;
    border-left: 4px solid transparent;
}

/* ── Target the actual text node inside the label ─────────
   Streamlit wraps label text in a <p> or <span> inside
   a <div> inside <label>. We target every descendant.      */
[data-testid="stSidebar"] .stRadio label *,
[data-testid="stSidebar"] .stRadio label {
    color: #f0f0f0 !important;
}

/* ── Color the left border per nav item ──────────────────── */
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] ~ div label:nth-of-type(1),
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:nth-child(1) {
    border-left-color: #4CAF50 !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:nth-child(2) {
    border-left-color: #2196F3 !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:nth-child(3) {
    border-left-color: #FF9800 !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:nth-child(4) {
    border-left-color: #E91E63 !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:nth-child(5) {
    border-left-color: #9C27B0 !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:nth-child(6) {
    border-left-color: #00BCD4 !important;
}

/* ── Hover highlight ─────────────────────────────────────── */
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255, 255, 255, 0.10) !important;
}

/* ── Hide the radio circle bullet ───────────────────────── */
[data-testid="stSidebar"] .stRadio input[type="radio"] {
    display: none;
}

/* ── Also hide the outer span Streamlit adds around bullet ─ */
[data-testid="stSidebar"] .stRadio label > span:first-child {
    display: none !important;
}

/* ── Main content card ────────────────────────────────────── */
[data-testid="stMain"] > div {
    background: rgba(255, 255, 255, 0.72);
    border-radius: 14px;
    padding: 1.5rem 2rem;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}

</style>
""", unsafe_allow_html=True)

# DATA Loading & Scoring for Recommendation Engine

@st.cache_data
def load_and_compute(csv_path="categorized_food_data.csv"):

    df = pd.read_csv(csv_path)

    #Calculating Macronutrient contribution to Calories
    df["Protein_Calories"] = df["Protein"] * 4
    df["Carb_Calories"]    = df["Carbohydrates"] * 4
    df["Fat_Calories"]     = df["Fat"] * 9

    df["Total_Macro_Calories"] = (
        df["Protein_Calories"] + df["Carb_Calories"] + df["Fat_Calories"]
    )

    df["Protein_pct"] = (df["Protein_Calories"] / df["Total_Macro_Calories"]) * 100
    df["Carb_pct"]    = (df["Carb_Calories"]    / df["Total_Macro_Calories"]) * 100
    df["Fat_pct"]     = (df["Fat_Calories"]     / df["Total_Macro_Calories"]) * 100

    norm_cols = ["Nutrition Density", "Protein", "Dietary Fiber","Caloric Value", "Carbohydrates", "Fat"]

    #Applying Minmax Scaling to all the above columns in norm_cols
    scaler = MinMaxScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(df[norm_cols]),
        columns=[c + "_norm" for c in norm_cols]
    )
    df = pd.concat([df, scaled], axis=1) #adding the column_norm columns to data

    #Computing Weight Loss Scores to find best foods for Weight Loss
    df["weight_loss_score"] = (
        0.4 * df["Nutrition Density_norm"]
        + 0.3 * df["Protein_norm"]
        + 0.2 * df["Dietary Fiber_norm"]
        - 0.1 * df["Caloric Value_norm"]
    )

    #Computing Muscle Gain Scores to find Best foods for Muscle Gain
    df["muscle_gain_score"] = (
        0.40 * df["Protein_norm"]
        + 0.25 * df["Caloric Value_norm"]
        + 0.15 * df["Carbohydrates_norm"]
        + 0.10 * df["Nutrition Density_norm"]
        + 0.10 * df["Fat_norm"]
    )

    #Computing Diabets_Score to find best Diabetes Friendly Scores
    healthy_cols   = ["Dietary Fiber", "Protein"]
    unhealthy_cols = ["Sugars", "Carbohydrates"]

    #Applying Min MAX scaling to the above Healthy and Unhealthy Columns
    scaler2 = MinMaxScaler()
    diabetes_scaled = pd.DataFrame(
        scaler2.fit_transform(df[healthy_cols + unhealthy_cols]),
        columns=healthy_cols + unhealthy_cols
    )
    df["Diabetes_Score"] = (
        diabetes_scaled["Dietary Fiber"]
        + diabetes_scaled["Protein"]
        - diabetes_scaled["Sugars"]
        - diabetes_scaled["Carbohydrates"]
    )

    #Computing Heart Health Score to find best Heart Healthy Foods
    heart_nutrients = ["Dietary Fiber", "Protein", "Saturated Fats",
                       "Sodium", "Cholesterol", "Sugars", "Carbohydrates"]
    #applying z-score standardization to find Heart Health Score
    z_food = df[heart_nutrients].apply(zscore)
    df["Heart_Health_Score"] = (
        3.0 * z_food["Dietary Fiber"]
        + 0.5 * z_food["Protein"]
        - 2.0 * z_food["Saturated Fats"]
        - 1.5 * z_food["Sodium"]
        - 1.0 * z_food["Cholesterol"]
        - 1.0 * z_food["Sugars"]
        - 0.5 * z_food["Carbohydrates"]
    )

    #Computing Balanced Score to find best balanced foods
    sugar_norm = (
        (df["Sugars"] - df["Sugars"].min())
        / (df["Sugars"].max() - df["Sugars"].min())
    )
    df["Balanced_Score"] = (
        0.4 * df["Nutrition Density_norm"]
        + 0.3 * df["Protein_norm"]
        + 0.2 * df["Dietary Fiber_norm"]
        - 0.1 * sugar_norm
    )

    return df


df = load_and_compute()

#To load the pages faster after each rerun, @st.cache_resource is used to cache data/results

# Creating Callable functions for generating multiple Plots across the analysis

#1)Bar Plot for Food Category Counts in Tab 1: Dataset Preview
@st.cache_resource
def fig_category_counts(categories: pd.Series):
    counts = categories.value_counts()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x=counts.values, y=counts.index, ax=ax)
    ax.set_title("Food Category Counts")
    return fig

#2)For Nutrient Correlation Heatmap in Tab 2: Nutrient Insights
@st.cache_resource
def fig_corr_heatmap(subset: pd.DataFrame):
    corr = subset.corr()
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(corr, annot=True,cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    return fig

#3)For Category-Wise Micronutrient Heatmap
@st.cache_resource
def fig_micronutrient_heatmap(subset: pd.DataFrame):
    cat_nut = subset.groupby("Food Category").mean()
    scaler  = MinMaxScaler()
    norm = pd.DataFrame(
        scaler.fit_transform(cat_nut),
        columns=cat_nut.columns, index=cat_nut.index
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(norm, annot=True, cmap="YlGnBu", ax=ax)
    return fig

#4) Average Nutrient Content by Food Category Plot in Tab 3: Category & Macronutrient Analysis

@st.cache_resource
def fig_category_plot(subset: pd.DataFrame, column: str):
    avg = (
        subset.groupby("Food Category")[column]
        .mean().sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=avg.values, y=avg.index, ax=ax)
    ax.set_title(f"Average {column} by Food Category")
    return fig

#5) Average Macronutrient Distribution by Food Category Bar Plot in Tab 3: Category & Macronutrient Analysis
@st.cache_resource
def fig_macro_stacked(subset: pd.DataFrame):
    macro_cat = subset.groupby("Food Category").mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    macro_cat.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Average % of Macro Calories")
    ax.set_title("Average Macronutrient Distribution by Food Category")
    return fig

#6) Macronutrient Composition Heatmap in Tab 3: Category & Macronutrient Analysis
@st.cache_resource
def fig_macro_heatmap(subset: pd.DataFrame):
    macro_cat = subset.groupby("Food Category").mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(macro_cat, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)
    ax.set_title("Macronutrient Composition by Food Category")
    return fig

#7) Barplots to show Macronutrient Profiles for Selected Food Categories in Tab 3
@st.cache_resource
def fig_category_macro_bar(subset: pd.DataFrame, selected_category: str):
    cat_data = subset[subset["Food Category"] == selected_category]
    summary  = pd.DataFrame({
        "Macronutrient": ["Protein", "Carbohydrates", "Fat"],
        "Percentage": [
            cat_data["Protein_pct"].mean(),
            cat_data["Carb_pct"].mean(),
            cat_data["Fat_pct"].mean()
        ]
    })
    fig, ax = plt.subplots(figsize=(5, 2))
    sns.barplot(data=summary, x="Macronutrient", y="Percentage", ax=ax)
    ax.set_title(f"{selected_category} Macronutrient Profile")
    return fig

#8)Z-score Heatmap of Vitamins by Food Category in Tab 4: Nutrient Heatmaps
@st.cache_resource
def fig_vitamin_heatmap(subset: pd.DataFrame):
    profile = subset.groupby("Food Category").mean()
    z = profile.apply(zscore)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(z, cmap="RdYlGn", center=0, ax=ax)
    ax.set_title("Z-Score Heatmap of Vitamins by Food Category")
    return fig

#8)Z-score Heatmap of Minerals by Food Category in Tab 4: Nutrient Heatmaps
@st.cache_resource
def fig_mineral_heatmap(subset: pd.DataFrame):
    profile = subset.groupby("Food Category").mean()
    z = profile.apply(zscore)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(z, cmap="RdYlGn", center=0, ax=ax)
    ax.set_title("Z-Score Heatmap of Minerals by Food Category")
    return fig

#8)Z-score Heatmap of Macronutrients by Food Category in Tab 4: Nutrient Heatmaps
@st.cache_resource
def fig_macro_zscore_heatmap(subset: pd.DataFrame):
    profile = subset.groupby("Food Category").mean()
    z = profile.apply(zscore)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(z, cmap="RdYlGn", center=0, ax=ax)
    ax.set_title("Z-Score Heatmap of Macronutrients by Food Category")
    return fig

#Storing all correlation columns to be used for correlation heatmap in Tab 2 in a list
_corr_cols = [
    "Caloric Value", "Protein", "Fat", "Carbohydrates",
    "Dietary Fiber", "Sugars", "Nutrition Density",
    "Vitamin A", "Vitamin B11", "Vitamin B2", "Vitamin B5",
    "Vitamin C", "Vitamin E", "Sodium", "Calcium", "Iron",
    "Manganese", "Potassium", "Zinc"
]

#Storing all Vitamin columns to be used for Vitamin Z-score heatmap in Tab 3 in a list
_vit_cols = [
    "Vitamin A", "Vitamin B1", "Vitamin B11", "Vitamin B12",
    "Vitamin B2", "Vitamin B3", "Vitamin B5", "Vitamin B6",
    "Vitamin C", "Vitamin D", "Vitamin E", "Vitamin K"
]

#Storing all Mineral columns to be used for Minerals Z-Score heatmap in Tab 3 in a list
_min_cols = [
    "Sodium", "Calcium", "Copper", "Iron", "Magnesium",
    "Manganese", "Phosphorus", "Potassium", "Selenium", "Zinc"
]

#Storing all Macronutrient columns to be used for Macronutrient Z-Score heatmap in Tab 3 in a list
_mac_zscore_cols = [
    "Caloric Value", "Fat", "Saturated Fats", "Monounsaturated Fats",
    "Polyunsaturated Fats", "Carbohydrates", "Sugars", "Protein",
    "Dietary Fiber", "Cholesterol", "Nutrition Density"
]

#For Macronutrient composition heatmap, Avg disitribution Stacked  an Macronutrient profiles by category in Tab 3 
_macro_pct_cols = ["Food Category", "Protein_pct", "Carb_pct", "Fat_pct"]

#For Category-Wise Micronutrient Heatmap in Tab#2
_micro_cols     = ["Food Category", "Calcium", "Iron", "Potassium", "Vitamin C"]

#Creating Seperated dataframes to be used for plotting
df_categories   = df["Food Category"]
df_corr_subset  = df[_corr_cols]
df_micro_subset = df[_micro_cols]
df_macro_pct    = df[_macro_pct_cols]
df_vit_subset   = df[["Food Category"] + _vit_cols]
df_min_subset   = df[["Food Category"] + _min_cols]
df_mac_z_subset = df[["Food Category"] + _mac_zscore_cols]


# PAGE NAMES IN SIDEBAR NAVIGATION

NAV_PAGES = [
    "🗂️  Dataset Preview",
    "💡  Nutrient Insights",
    "📊  Category & Macronutrient Analysis",
    "🔥  Nutrient Heatmaps",
    "📌  Key Insights & Recommendations",
    "🥗  Recommendation Engine",
]

with st.sidebar:
    st.markdown('<p class="nav-title">Navigation</p>', unsafe_allow_html=True)
    page = st.radio("", NAV_PAGES, label_visibility="collapsed")

st.title("🥗 Nutrition Analytics & Recommendation System")


# Tab 1: DATASET PREVIEW


if page == NAV_PAGES[0]:

    st.header("Dataset Preview")

    st.subheader("First 10 Rows")
    st.dataframe(df.head(10))

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Columns")
    st.write(df.columns.tolist())

    st.subheader("Data Types")
    st.dataframe(pd.DataFrame(df.dtypes, columns=["Data Type"]))

    st.subheader("Food Category Counts")
    st.pyplot(fig_category_counts(df_categories))

# Tab 2: NUTRIENT INSIGHTS

elif page == NAV_PAGES[1]:

    st.header("Nutrient Insights")

    st.subheader("Top 10 Foods by Macronutrient Calories")

    #Top 10 Protein-Rich Foods identified using Protein_Calories Column
    top_protein = (
        df.nlargest(10, "Protein_Calories")
        [["food", "Protein_Calories", "Food Category"]]
        .reset_index(drop=True)
    )

    #Top 10 Carb-Heavy Foods identified using Carb_Calories Column
    top_carbs = (
        df.nlargest(10, "Carb_Calories")
        [["food", "Carb_Calories", "Food Category"]]
        .reset_index(drop=True)
    )

    #Top 10 Fat-Heavy Foods identified using Fat_Calories column
    top_fat = (
        df.nlargest(10, "Fat_Calories")
        [["food", "Fat_Calories", "Food Category"]]
        .reset_index(drop=True)
    )

    #Creating 3 columns to display top 10 of Protein-Rich, Carb-Heavy and Fat-Heavy side by side
    col_p, col_c, col_f = st.columns(3)
    with col_p:
        st.markdown("**Top 10 Protein-Rich Foods**")
        st.dataframe(top_protein, use_container_width=True)
    with col_c:
        st.markdown("**Top 10 Carb-Heavy Foods**")
        st.dataframe(top_carbs, use_container_width=True)
    with col_f:
        st.markdown("**Top 10 Fat-Heavy Foods**")
        st.dataframe(top_fat, use_container_width=True)

    st.divider()

    #Nutrient Correlation Heatmap between multiple Micronutrients, Macronutrients and Vitamins
    st.subheader("Nutrient Correlation Heatmap")
    st.pyplot(fig_corr_heatmap(df_corr_subset))

    #Micronutrient Heatmap of minerals for all Food Categories
    st.subheader("Category-wise Micronutrient Heatmap")
    st.pyplot(fig_micronutrient_heatmap(df_micro_subset))


# Tab 3: CATEGORY & MACRONUTRIENT ANALYSIS

elif page == NAV_PAGES[2]:

    st.header("📊 Category & Macronutrient Analysis")

    st.subheader("Average Nutrient Content by Food Category")

    #Creating a radio button widget to allow users to select one of the Macronutrients
    selected_nutrient = st.radio(
        "Select Nutrient",
        ["Caloric Value", "Protein", "Fat",
         "Dietary Fiber", "Sugars", "Nutrition Density"],
        horizontal=True,
        key="nutrient_radio"
    )

    #Bar Plots for Selected Average Macronutrient Content by Food Category
    st.pyplot(fig_category_plot(
        df[["Food Category", selected_nutrient]], selected_nutrient
    ))

    st.divider()

    #Stacked Bar Plot displaying protein percent, Carb Percent and Fat Percent by Food Category
    st.subheader("Average Macronutrient Distribution by Food Category")
    st.pyplot(fig_macro_stacked(df_macro_pct))

    #A heatmap of the same Macronutrient Composition as the above plot
    st.subheader("Macronutrient Composition Heatmap")
    st.pyplot(fig_macro_heatmap(df_macro_pct))


# Tab 4: NUTRIENT HEATMAPS

elif page == NAV_PAGES[3]:

    st.header("🔥 Nutrient Heatmaps")

    #Below 3 plots are Z-Score heatmaps of Various ,Minerals, Vitamins, and Macronutrients
    st.subheader("Vitamin Profile by Food Category (Z-Score Heatmap)")
    st.pyplot(fig_vitamin_heatmap(df_vit_subset))

    st.subheader("Mineral Profile by Food Category (Z-Score Heatmap)")
    st.pyplot(fig_mineral_heatmap(df_min_subset))

    st.subheader("Macronutrient Profile by Food Category (Z-Score Heatmap)")
    st.pyplot(fig_macro_zscore_heatmap(df_mac_z_subset))

    st.subheader("Category Nutrient Drilldown")

    #Checking Average Protein, Fat, Carbohydrates, Dietary Fiber and Nutrition Density Content by Selectingg food Category

    selected_category_hm = st.selectbox(
        "Select Food Category",
        sorted(df["Food Category"].unique()),
        key="heatmap_category"
    )

    cat_data_hm = df[df["Food Category"] == selected_category_hm]

    st.dataframe(pd.DataFrame({
        "Protein":           [cat_data_hm["Protein"].mean()],
        "Fat":               [cat_data_hm["Fat"].mean()],
        "Carbohydrates":     [cat_data_hm["Carbohydrates"].mean()],
        "Dietary Fiber":     [cat_data_hm["Dietary Fiber"].mean()],
        "Nutrition Density": [cat_data_hm["Nutrition Density"].mean()]
    }).round(2))


# Tab 5: RECOMMENDATION ENGINE

elif page == NAV_PAGES[5]:

    st.header("🥗 Personalized Recommendation Engine")

    # USER INPUTS

    name = st.text_input("Enter your Name")
    gender = st.radio("Select Gender", ["Male", "Female"])
    age    = st.slider("Age", 1, 100, 25)

    height = st.number_input("Height (cm)", min_value=75.0, max_value=250.0, value=170.0)
    weight = st.number_input("Weight (kg)", min_value=20.0, max_value=250.0, value=70.0)

    activity_level = st.selectbox(
        "Activity Level",
        [
            "Sedentary(0 days/week)",
            "Light(1-2 days/week)",
            "Moderate(3-5 days/week)",
            "Active(6-7 days/week)",
            "Very Active(2x/day)"
        ]
    )

    goal = st.selectbox(
        "Select Goal",
        [
            "Maintain Weight",
            "Weight Loss",
            "Muscle Gain",
            "Heart Health",
            "Diabetic Friendly"
        ]
    )

    #weeks = st.slider("Commitment Period (Weeks)", 1, 16, 8)

    if st.button("Generate Recommendations"):

        # BMI Calculations

        h_m = height / 100  #height in Meters
        bmi = weight / (h_m ** 2) 

        #Displaying BMI Value
        st.subheader("BMI")
        st.metric("Your BMI is", f"{bmi:.2f}")

        if bmi < 18.5:
            st.warning("Underweight")
        elif bmi <= 24.9:
            st.success("Healthy Weight")
        elif bmi <= 29.9:
            st.warning("Overweight")
        else:
            st.error("Obese")

        #Calculating Basic Metabolic Rate (VMR0)
        if gender == "Male":
            bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
        else:
            bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161

        #Storng Activity Multiplier values in a dictionary
        activity_multiplier = {
            "Sedentary(0 days/week)":   1.2,
            "Light(1-2 days/week)":     1.375,
            "Moderate(3-5 days/week)":  1.55,
            "Active(6-7 days/week)":    1.725,
            "Very Active(2x/day)":      1.9
        }

        #Total Daily Energy Expenditure Calculation
        tdee = bmr * activity_multiplier[activity_level]

        # CALORIE / PROTEIN TARGETS

        goal_map = {
            "Maintain Weight":   ("Balanced_Score",     tdee),
            "Weight Loss":       ("weight_loss_score",  tdee - 500),
            "Muscle Gain":       ("muscle_gain_score",  tdee + 300),
            "Heart Health":      ("Heart_Health_Score", tdee),
            "Diabetic Friendly": ("Diabetes_Score",     tdee),
        }

        #Displaying Calorie Target, Protein Target and TDEE
        score_col, calorie_target = goal_map[goal]
        protein_target = 1.6 * weight if goal == "Muscle Gain" else 1.2 * weight

        st.subheader("Daily Targets")
        col1, col2, col3 = st.columns(3)
        col1.metric("Calories", f"{calorie_target:.0f}")
        col2.metric("Protein",  f"{protein_target:.0f} g")
        col3.metric("TDEE",     f"{tdee:.0f}")

        #  FOOD POOL & SCORING

        #Ignoring Food from Other Categories and High Calorie Foods to bring Variety to diet

        food_pool = df[
            (df["Food Category"] != "Other")
            & (df["Caloric Value"] <= 600)
            & (df["Caloric Value"] > 0)
        ].copy()


        food_pool["Protein_Density"] = (
            food_pool["Protein"] / food_pool["Caloric Value"]
        )

        food_pool["Final_Score"] = (
            0.7 * food_pool[score_col]
            + 0.3 * food_pool["Protein_Density"]
        )
        food_pool = food_pool.sort_values("Final_Score", ascending=False)

        #  TOP CATEGORIES

        st.subheader("Top Food Categories")

        category_scores = (
            food_pool.groupby("Food Category")["Final_Score"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )

        fig_cs, ax_cs = plt.subplots(figsize=(10, 6))
        sns.barplot(x=category_scores.values, y=category_scores.index, ax=ax_cs)
        st.pyplot(fig_cs)

        # ── TOP FOODS PER CATEGORY ─────────────────────────

        st.subheader("Top Foods Per Category")

        top3 = (
            food_pool
            .sort_values("Final_Score", ascending=False)
            .groupby("Food Category")
            .head(3)
        )

        result = (
            top3.groupby("Food Category")["food"]
            .apply(list)
            .apply(pd.Series)
        )

        st.dataframe(result)

        # ── BAR PLOT: top 3 per category by goal score ─────

        bar_plot_categories = [
            "Breakfast Foods", "Eggs", "Fruit Juices", "Fruits",
            "Grains & Cereals", "Legumes & Beans", "Nuts & Seeds",
            "Plant-Based Protein", "Poultry", "Red Meat",
            "Seafood", "Snacks", "Soups", "Vegetables",
        ]

        #Goal Labels to be used for Recommendation Engine depending on the user input for Goal
        goal_score_labels = {
            "Maintain Weight":   ("Balanced_Score",     "Balanced Score"),
            "Weight Loss":       ("weight_loss_score",  "Weight Loss Score"),
            "Muscle Gain":       ("muscle_gain_score",  "Muscle Gain Score"),
            "Heart Health":      ("Heart_Health_Score", "Heart Health Score"),
            "Diabetic Friendly": ("Diabetes_Score",     "Diabetes Score"),
        }

        bar_score_col, bar_score_label = goal_score_labels[goal]

        #Using the goal score based on the selected user input for goal
        plot_df = (
            top3[top3["Food Category"].isin(bar_plot_categories)]
            .sort_values(["Food Category", bar_score_col],
                         ascending=[True, False])
        )

        #Displaying Top Food Categories for the selected goal
        if not plot_df.empty:
            fig_bar, ax_bar = plt.subplots(figsize=(16, 6))
            sns.barplot(
                data=plot_df,
                x="food",
                y=bar_score_col,
                hue="Food Category",
                ax=ax_bar
            )
            ax_bar.set_xticklabels(
                ax_bar.get_xticklabels(), rotation=90, ha="right"
            )
            ax_bar.set_title(f"Top 3 Foods per Category — {goal}")
            ax_bar.set_xlabel("Food")
            ax_bar.set_ylabel(bar_score_label)
            ax_bar.legend(
                title="Food Category",
                bbox_to_anchor=(1.01, 1),
                loc="upper left",
                fontsize=8
            )
            plt.tight_layout()
            st.pyplot(fig_bar)

        # TOP 10 OVERALL Foods for the Goal irrespective of the Categories

        st.subheader("Top 10 Foods Overall")
        st.dataframe(
            food_pool[["food", "Food Category", "Caloric Value",
                        "Protein", "Final_Score"]].head(10)
        )

        # MEAL TARGETS

        #Each meal gets a fixed share of the day's calorie and protein budget
        #Meal = (fraction of calorie budget, fraction of protein budget)
        meal_splits = {
            "Breakfast": (0.25, 0.25),
            "Lunch":     (0.35, 0.35), #Lunch is the biggest meal with 35% fraction of both budgets
            "Snack":     (0.10, 0.10),
            "Dinner":    (0.30, 0.30),
        }

        #Getting exact targets from the above frations
        meal_calorie_targets = {
            m: calorie_target * s[0] for m, s in meal_splits.items()
        }
        meal_protein_targets = {
            m: protein_target * s[1] for m, s in meal_splits.items()
        }

        # GOAL-SPECIFIC CATEGORY POOLS

        #Fixing Food categories to be used for each fitness goal and meal type.
        #Only foods from these pools will be used based on user input for fitness goal and Meal plan being displayed
        goal_categories = {
            "Weight Loss": {
                "Breakfast": ["Breakfast Foods", "Fruits",
                              "Nuts & Seeds", "Fruit Juices"],
                "Lunch":     ["Legumes & Beans", "Vegetables", "Seafood",
                              "Poultry", "Plant-Based Protein"],
                "Snack":     ["Snacks", "Fruits", "Nuts & Seeds"],
                "Dinner":    ["Vegetables", "Legumes & Beans", "Seafood",
                              "Soups", "Plant-Based Protein"],
            },
            "Muscle Gain": {
                "Breakfast": ["Breakfast Foods", "Eggs",
                              "Protein Supplements", "Dairy & Cheese"],
                "Lunch":     ["Poultry", "Red Meat", "Seafood",
                              "Legumes & Beans", "Grains & Cereals"],
                "Snack":     ["Protein Supplements", "Nuts & Seeds", "Snacks"],
                "Dinner":    ["Poultry", "Seafood", "Red Meat",
                              "Legumes & Beans", "Plant-Based Protein"],
            },
            "Heart Health": {
                "Breakfast": ["Breakfast Foods", "Fruits", "Nuts & Seeds"],
                "Lunch":     ["Legumes & Beans", "Vegetables",
                              "Plant-Based Protein", "Seafood"],
                "Snack":     ["Fruits", "Nuts & Seeds", "Snacks"],
                "Dinner":    ["Vegetables", "Legumes & Beans",
                              "Soups", "Seafood"],
            },
            "Diabetic Friendly": {
                "Breakfast": ["Breakfast Foods", "Eggs", "Nuts & Seeds"],
                "Lunch":     ["Legumes & Beans", "Vegetables",
                              "Seafood", "Poultry"],
                "Snack":     ["Nuts & Seeds", "Snacks"],
                "Dinner":    ["Vegetables", "Seafood",
                              "Legumes & Beans", "Soups"],
            },
            "Maintain Weight": {
                "Breakfast": ["Breakfast Foods", "Fruits", "Eggs"],
                "Lunch":     ["Legumes & Beans", "Poultry", "Seafood",
                              "Vegetables", "Grains & Cereals"],
                "Snack":     ["Snacks", "Fruits", "Nuts & Seeds"],
                "Dinner":    ["Seafood", "Poultry", "Vegetables",
                              "Soups", "Legumes & Beans"],
            },
        }

        #Filtering the food pool to only the categories allowed for this meal from above pre-determined categories for each goal
        def build_pool(categories):
            pool = food_pool[
                food_pool["Food Category"].isin(categories)
            ].sort_values("Final_Score", ascending=False)
            return pool if not pool.empty else food_pool #Safety fallbavk incase the filtering produces an empty pool

        #Used the above defined function and passed appropriate goal category to get a sorted pool based on Final_Score for each meal type
        pools = {
            meal: build_pool(cats)
            for meal, cats in goal_categories[goal].items()
        }

        # MEAL BUILDER

    
        def get_meal(
            target_calories,
            target_protein,
            pool,                     #Sorted foods in the pool by final score
            food_last_used_day,       #dictionary: food-name; day a food was last used (attempting to keep atleast 1 to 2 days between same foods)
            food_use_count,           #dictionary: food_name: how many times food was used this week
            used_today,               #set: foods already placed in an earlier meal today
            day_idx,                  #day_idx: id for day (0=monday...6=Sunday)
            max_foods=5,              #max_foods: To cap the number of items that can be generated for a single meal
            cal_tolerance=0.10,       #Since exact match with calorie target is sometimes impossible, a 10% tolerance is being allowed to stop generating the meal
            pro_tolerance=0.10,       #Similar for Protein Target
            min_gap=2,                #gap between repetition of foods (fixed as 2) to allow for variety in diet
            max_weekly_uses=2,        #Maximum number of times the food can appear in the whole week's plan: fixed at 2
        ):
            meal            = []      #To store foods chosen for this week
            cal_acc         = 0.0     #to keep track of calories based on foods added so far
            pro_acc         = 0.0     #to keep track of protein based on foodds added
            seen_in_meal    = set()   #foods already added to the meal (to avoid duplicates)
            categories_used = set()   #food categories already used for the meal (to keep variety)

            #Upper limit on protein in order to never exceed the target by more than 10%
            pro_ceiling = target_protein * (1 + pro_tolerance)

            # ── Shared eligibility check, reused by both the main pass
            #    and the calorie top-up pass below
            def is_eligible(food_name, food_cat):
                if food_name in seen_in_meal:
                    return False                                  #Constraint 1
                if food_name in used_today:
                    return False                                  #Constraint 2
                last_day = food_last_used_day.get(food_name, -999)
                if day_idx - last_day < min_gap:
                    return False                                  #Constraint 3
                if food_use_count.get(food_name, 0) >= max_weekly_uses:
                    return False                                  #Constraint 4
                if food_cat in categories_used:
                    return False                                  #Constraint 5 (one food per category)
                return True

            def commit_food(row, food_name, food_cat, scaled_cal, scaled_pro, portion):
                """Adds a food to the meal and updates all shared tracking state."""
                nonlocal cal_acc, pro_acc

                row_copy                  = row.copy()
                row_copy["Caloric Value"] = round(scaled_cal, 1)
                row_copy["Protein"]       = round(scaled_pro, 1)
                row_copy["Portion"]       = round(portion, 2)

                meal.append(row_copy)
                seen_in_meal.add(food_name)
                used_today.add(food_name)
                food_last_used_day[food_name] = day_idx
                food_use_count[food_name]     = food_use_count.get(food_name, 0) + 1
                categories_used.add(food_cat)

                cal_acc += scaled_cal
                pro_acc += scaled_pro

            # PASS 1 — Main selection loop (calorie + protein balanced)
            for _, row in pool.iterrows():

                food_name = row["food"]
                food_cat  = row["Food Category"]

                if not is_eligible(food_name, food_cat):
                    continue

                food_cal = row["Caloric Value"]
                food_pro = row["Protein"]

                if food_cal <= 0:
                    continue

                #Calculating how many calories are still needed to hit the target
                #if calorie accounted so far crosses target_calories, remaining calories is set to zero
                remaining_cal = max(0.0, target_calories - cal_acc)

                #This is determine how many servings would be needed to close the calorie gap
                cal_mult      = remaining_cal / food_cal

                #Fixing the portion to a realistic range: never less than 1/2 a serving and never more than 2.5x the serving
                cal_portion   = max(0.5, min(2.5, cal_mult))

                scaled_cal = food_cal * cal_portion
                scaled_pro = food_pro * cal_portion

                # Starting with the calorie-driven portion as our working portion
                portion = cal_portion

                #Protein Ceiling
                #If adding this food would push the total protein past the allowed ceiling,
                #only the PROTEIN contribution is reduced — calories stay at scaled_cal.
                if pro_acc + scaled_pro > pro_ceiling:
                    allowed_pro = pro_ceiling - pro_acc

                    #If there's no protein budget left, skip this food and go to the next ranked food item
                    if allowed_pro <= 0:
                        continue

                    if food_pro > 0:
                        
                        pro_driven_portion = allowed_pro / food_pro
                        scaled_pro = food_pro * min(cal_portion, pro_driven_portion)
                        portion    = min(cal_portion, pro_driven_portion) 

                #Add this food to the meal and update all tracking variables
                commit_food(row, food_name, food_cat, scaled_cal, scaled_pro, portion)

                #In case the meal reaches the maximum number of items (5), break the iteration
                if len(meal) >= max_foods:
                    break

                #To stop early if both calorie and protein targets are close enough (within their set tolerance)
                cal_ok = cal_acc >= target_calories * (1 - cal_tolerance)
                pro_ok = pro_acc >= target_protein  * (1 - pro_tolerance)
                if cal_ok and pro_ok:
                    break

            #For Calorie top-up pass.
            #If the main loop above finished (hit max_foods, ran out of
            #eligible pool items, or both targets were satisfied) but
            #calories are STILL below target, keep adding more foods
            #purely to close the calorie gap. Protein is intentionally
            #NOT re-checked against the ceiling here, since by this point
            #protein is already at/above its tolerance band — we only
            #need more calories, not more protein.
    
           
            cal_shortfall = target_calories * (1 - cal_tolerance) - cal_acc

            if cal_shortfall > 0 and len(meal) < max_foods:
                for _, row in pool.iterrows():

                    if len(meal) >= max_foods:
                        break

                    food_name = row["food"]
                    food_cat  = row["Food Category"]

                    if not is_eligible(food_name, food_cat):
                        continue

                    food_cal = row["Caloric Value"]
                    food_pro = row["Protein"]

                    if food_cal <= 0:
                        continue

                    remaining_cal = max(0.0, target_calories - cal_acc)
                    cal_mult      = remaining_cal / food_cal
                    portion       = max(0.5, min(2.5, cal_mult))

                    scaled_cal = food_cal * portion
                    scaled_pro = food_pro * portion

                    commit_food(row, food_name, food_cat, scaled_cal, scaled_pro, portion)

                    if cal_acc >= target_calories * (1 - cal_tolerance):
                        break

            #Returning the finished meal as a Dataframe along with the foods used_today so that next day's caller will have the set of food items to not include
            return pd.DataFrame(meal), used_today

        # WEEKLY PLAN GENERATION

        days = ["Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday"]

        weekly_plan        = {}   #For final output: {day: {meal_name: DataFrame}}
        food_last_used_day = {}   #Tracks last used food
        food_use_count     = {}   #Tracks weekly cap

        for day_idx, day in enumerate(days):

            #resets every day: to prevent a food appearing twice in the same day
            used_today = set()

            #To shuffle each meal's food pool differently per day so the plan doesn't always pick the exact same top foods every
            day_pools = {}
            for meal_name, pool in pools.items():
                shuffled = pool.sample(
                    frac=1,
                    random_state=day_idx * 10 + list(pools).index(meal_name)
                ).sort_values("Final_Score", ascending=False)
                day_pools[meal_name] = shuffled

            
            #Building all 4 meals for this day, in order
            #used_today is carried forward from one meal to the next so later meals know what's already been used earlier in same day
            meals = {}
            for meal_name in ["Breakfast", "Lunch", "Snack", "Dinner"]:
                meal_df, used_today = get_meal(
                    target_calories    = meal_calorie_targets[meal_name],
                    target_protein     = meal_protein_targets[meal_name],
                    pool               = day_pools[meal_name],
                    food_last_used_day = food_last_used_day,   #Shared across whole week
                    food_use_count     = food_use_count,       #Shared across whole week
                    used_today         = used_today,
                    day_idx            = day_idx,
                )
                meals[meal_name] = meal_df #each meal in each day is stored as a dataframe

            #Each day's plan is stored in weekly_plan dictionary
            weekly_plan[day] = meals

        # DISPLAY Final Weekly Plan

        st.header("Weekly Meal Plan")

        for day, meals in weekly_plan.items():

            st.subheader(day)

            for meal_name, meal_df in meals.items():
                st.markdown(f"### {meal_name}")
                display_cols = ["food", "Food Category",
                                "Caloric Value", "Protein", "Portion"]
                show_cols = [c for c in display_cols if c in meal_df.columns]
                st.dataframe(meal_df[show_cols])

            day_cal = sum(m["Caloric Value"].sum() for m in meals.values())
            day_pro = sum(m["Protein"].sum()       for m in meals.values())

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Calories",  f"{day_cal:.0f}")
            c2.metric("Target Calories", f"{calorie_target:.0f}")
            c3.metric("Total Protein",   f"{day_pro:.1f} g")
            c4.metric("Target Protein",  f"{protein_target:.0f} g")

            st.divider()


# Tab 6: KEY INSIGHTS & RECOMMENDATIONS

elif page == NAV_PAGES[4]:

    st.header("📌 Key Insights & Recommendations")

    st.markdown("---")

    col_ins, col_rec = st.columns(2, gap="large")

    # LEFT COLUMN: Major Insights
    with col_ins:

        st.markdown("""
<div style="
    background: rgba(255,255,255,0.82);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    border-left: 5px solid #2196F3;
    backdrop-filter: blur(6px);
">
<h3 style="color:#1565C0; margin-top:0;">🔍 Major Insights</h3>

<p><b>1.</b> Most foods contain low-to-moderate levels of calories, protein, fat, carbohydrates, fiber, and nutrition density — only a small number of foods show extremely high values.</p>

<p><b>2.</b> Protein and fat are the strongest contributors to caloric value, while carbohydrates and sugars show a much weaker relationship with calories.</p>

<p><b>3.</b> <b>Legumes & Beans</b> emerged as the most nutritionally balanced category — ranking high in protein and dietary fiber along with a wide range of minerals, while ranking low in fats and sugars.</p>

<p><b>4.</b> <b>Dairy & Cheese</b> and <b>Plant-Based Protein</b> have the highest nutrition density, while <b>Red Meat</b> is among the most mineral-rich categories, supplying significant iron, potassium, phosphorus, and zinc.</p>

<p><b>5.</b> <b>Fast Foods, Snacks, and Desserts</b> are nutritionally less desirable — high in calories, fats, carbohydrates, and sugars, resulting in a less balanced nutritional profile.</p>

<p><b>6.</b> <b>Red Meat, Seafood, and Poultry</b> are among the richest protein sources, with Seafood particularly rich in Potassium and both Red Meat and Poultry providing substantial Iron and Potassium.</p>

<p><b>7.</b> <b>Fruits, Fruit Juices, and Vegetables</b> are the strongest sources of Vitamin C. No single category dominates all vitamin groups — reinforcing the need for a diverse diet.</p>

<p><b>8.</b> <b>Eggs</b> emerged as one of the most balanced food categories, exhibiting a favorable balance of protein, carbohydrates, and fats compared to many other food groups.</p>

<p><b>9.</b> <b>Protein Supplements</b> rank highest for B-Complex Vitamins — among the richest sources of Vitamins B1, B2, B3, B11, and B12 in the entire dataset.</p>

<p><b>10.</b> <b>Nuts & Seeds</b> stand out as a healthy fat source — combining high fat content with moderate protein while also being the richest sources of Vitamin E.</p>
</div>
""", unsafe_allow_html=True)

    # RIGHT COLUMN: Top 10 Recommendations
    with col_rec:

        st.markdown("""
<div style="
    background: rgba(255,255,255,0.82);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    border-left: 5px solid #4CAF50;
    backdrop-filter: blur(6px);
">
<h3 style="color:#2E7D32; margin-top:0;">✅ Top 10 Recommendations</h3>

<p><b>1.</b> For specific goals: <b>Acerola Cherry Juice</b> tops weight loss, <b>Pork Arm Picnic Cooked</b> ranks highest for muscle gain, <b>Chokeberries</b> lead for heart health, and <b>Shrimp Cooked</b> is best for diabetes-friendly diets.</p>

<p><b>2.</b> <b>Turkey Breast Roasted</b> stands out as the most versatile food — ranking highly for protein, weight loss, muscle gain, and lean muscle gain simultaneously.</p>

<p><b>3.</b> For <b>diabetes-friendly</b> diets, prioritize: Sour Cherries, Golden Delicious Apples, Lychees, Pink Beans, White Beans, and Parsley (dried) — all consistently top performers.</p>

<p><b>4.</b> For better <b>diabetes management</b>, reduce intake of Alcoholic Beverages and fruit juices such as Fruit Cocktail Canned, Müller Thurgau White Wine, Apricot Juice, and Citrus Fruit Juice — these performed poorly in diabetes-friendly rankings.</p>

<p><b>5.</b> For <b>heart health</b>, prioritize Chokecherries, Mammy Apple, Navy Beans, Yellow Beans, Wheat Bran, and Rice Bran — all among the highest-performing options in the heart-health analysis.</p>

<p><b>6.</b> <b>Reduce</b> frequent consumption of high-fat red meats, oils & fats, and Dairy & Cheese for heart health — these categories performed less favorably compared to legumes, fruits, vegetables, and whole grains.</p>

<p><b>7.</b> For <b>high-protein, lower-carbohydrate</b> diets, prioritize <b>Seafood</b> — it has the highest proportion of protein, one of the lowest proportions of carbohydrates, and a moderate fat content.</p>

<p><b>8.</b> <b>Limit</b> Alcoholic Beverages and Oils & Fats for balanced macronutrient intake — Alcoholic Beverages derive most calories from carbohydrates, while Oils & Fats are almost entirely fat, both contributing minimal protein.</p>

<p><b>9.</b> Prioritize <b>Plant-Based Proteins, Seafood, Poultry, and Legumes</b> — these categories consistently ranked highly across multiple vitamin and mineral analyses rather than excelling in only a single nutrient.</p>

<p><b>10.</b> Build <b>balanced meals</b> around: <b>Legumes & Beans</b> (fiber & minerals) · <b>Seafood, Poultry, or Red Meat</b> (protein) · <b>Protein Supplements & Meat</b> (B-complex vitamins) · <b>Fruit Juices</b> (Vitamin C) · <b>Nuts & Seeds</b> (healthy fats).</p>
</div>
""", unsafe_allow_html=True)