# /// script
# dependencies = [
# "pandas>=2.0.0",
# "openpyxl>=3.1.0",
# "streamlit>=1.25.0",
# "plotly>=5.15.0",
# "pytest>=7.4.0",
# "streamlit",
# ]
# ///

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import os
from datetime import datetime

def get_latest_file(directory: Path, pattern: str) -> Path:
    """Get the most recent file matching the pattern from directory."""
    files = list(directory.glob(pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def get_value_transformations():
    """Define value transformations for various fields."""
    return {
        'Moat': {
            'Wide': 'Wid',
            'Narrow': 'Nar',
            '': ''
        },
        'RecDiv': {
            'Increased': 'Inc',
            'Maintained': 'Mnt',
            'Cut': 'Cut',
            '': ''
        },
        'Val': {
            'May be undervalued': 'Und',
            'Looks reasonably valued': 'Rea',
            'Could be overvalued': 'Ovr',
            '': ''
        },
        'FVU': {
            'Very High': 'VHi',
            'High': 'Hig',
            'Medium': 'Med',
            'Low': 'Low',
            'Very Low': 'VLo',
            '': ''
        }
    }

def apply_value_transformations(df):
    """Apply value transformations to specified columns."""
    transformations = get_value_transformations()
    
    for col, mapping in transformations.items():
        if col in df.columns:
            # Fill NaN with empty string first
            df[col] = df[col].fillna('')
            # Apply transformation
            df[col] = df[col].map(lambda x: mapping.get(x, mapping.get('', '')))
    
    return df

def load_data():
    """Load and merge data from the latest xlsx and csv files."""
    downloads_dir = Path.home() / "Downloads"
    
    # Get latest files
    xlsx_file = get_latest_file(downloads_dir, "for-divs*.xlsx")
    csv_file = get_latest_file(downloads_dir, "ForDivs*.csv")
    
    if not xlsx_file or not csv_file:
        st.error("Could not find both Excel and CSV files in Downloads directory")
        return None, None
    
    try:
        # Read files
        df_xlsx = pd.read_excel(xlsx_file, engine='openpyxl')
        df_csv = pd.read_csv(csv_file)
        
        # Validate "Ticker" column presence
        if "Ticker" not in df_xlsx.columns or "Ticker" not in df_csv.columns:
            st.error("Both files must contain a 'Ticker' column")
            return None, None
        
        # Clean ticker symbols (remove any whitespace and make uppercase)
        df_xlsx['Ticker'] = df_xlsx['Ticker'].str.strip().str.upper()
        df_csv['Ticker'] = df_csv['Ticker'].str.strip().str.upper()
        
        # Define column renames
        column_renames = {
            'Economic Moat': 'Moat',
            'Dividend Yield (Forward)': 'YldFwd',
            'Dividend Yield (Trailing)': 'YldTrl',
            'Dividend Safety': 'DivSafe',
            'Price/Earnings (Forward)': 'PEFwd',
            'Price/Cash Flow': 'PCF',
            'Expected Price': 'ExpPrc',
            '% From Expected Price': 'FromExp',
            '5 Year Average Dividend Yield': 'YldAvg5',
            '% Above 5 Year Average Dividend Yield': 'Yld5Dif',
            '5 Year Average P/E Ratio': 'PEAvg5',
            'Dividend Growth (Latest)': 'DivGrw',
            '5-Year Dividend Growth': 'DivGrw5',
            '10-Year Dividend Growth': 'DivGrw10',
            'Dividend Growth Streak (Years)': 'StrDiv',
            'Uninterrupted Dividend Streak (Years)': 'StrUnt',
            'Payment Frequency': 'PayFrq',
            'Payment Schedule': 'PaySch',
            'Dividend Taxation': 'DivTax',
            'Recession Dividend': 'RecDiv',
            'Recession Return': 'RecRet',
            'Ex Dividend Date': 'ExDiv',
            'Years Of Positive Free Cash Flow In Last 10': 'FCFYrs',
            'Payout Ratio': 'Payout',
            'Net Debt To Capital': 'DebtCap',
            'Net Debt To EBITDA': 'DebtEBT',
            'P/E Ratio': 'PE',
            'Capital Allocation': 'CapAll',
            'Morningstar Rating for Stocks': 'MSRate',
            'Fair Value Uncertainty': 'FVU',
            'Price/Fair Value': 'PFV',
            'Valuation': 'Val',
            'Interest Coverage': 'IntCov'
        }
        
        # Rename columns in both dataframes
        df_xlsx = df_xlsx.rename(columns=column_renames)
        df_csv = df_csv.rename(columns=column_renames)
        
        # Convert numeric columns in Excel file
        numeric_columns = [
            'Beta', 'Payout', 'DebtEBT',
            'DivGrw', 'DivGrw5', 'DivGrw10',
            'YldFwd', 'YldTrl',
            'IntCov'
        ]
        
        for col in numeric_columns:
            if col in df_xlsx.columns:
                df_xlsx[col] = pd.to_numeric(df_xlsx[col], errors='coerce')
                if col != 'StrDiv' and col != 'StrUnt':  # Don't round streak values
                    df_xlsx[col] = df_xlsx[col].round(2)
        

        # Standardize sector names in both dataframes before merging
        sector_mapping = {
            'Consumer Defensive': 'Consumer Staples',
            'Consumer Cyclical': 'Consumer Discretionary',
            'Communication Services': 'Communications',
            'Basic Materials': 'Materials',
            'Financial Services': 'Financials'
        }
        
        if 'Sector' in df_xlsx.columns:
            df_xlsx['Sector'] = df_xlsx['Sector'].replace(sector_mapping)
        if 'Sector' in df_csv.columns:
            df_csv['Sector'] = df_csv['Sector'].replace(sector_mapping)
        
        # Merge datasets with outer join and indicator
        df_merged = pd.merge(df_xlsx, df_csv, on="Ticker", how="outer", suffixes=('_xlsx', '_csv'))
        
        # Create Dividend Yield column with precedence:
        # 1. Forward Yield from Excel
        # 2. Trailing Yield from Excel
        # 3. Dividend Yield from CSV
        df_merged['Yield'] = df_merged['YldFwd']
        df_merged.loc[df_merged['Yield'].isna(), 'Yield'] = df_merged['YldTrl']
        df_merged.loc[df_merged['Yield'].isna(), 'Yield'] = df_merged['Dividend Yield']
        
        # Define columns to remove
        columns_to_remove = [
            'YldFwd', 'YldTrl', 'Dividend Yield',  # Yield columns (already handled)
            'PEFwd', 'PCF',                        # Price ratios
            'MktCap', 'ExpPrc', 'FromExp',         # Price and market related
            'YldAvg5', 'Yld5Dif',                  # Yield averages
            'PE', 'PEAvg5',                        # P/E ratios
            'PayFrq', 'DebtCap', 'PaySch',         # Payment and debt
            'Market Cap (Millions)', 'RecRet', 'Sub Sector', 'Industry', 'CapAll'
        ]
        
        # Drop all specified columns
        df_merged = df_merged.drop(columns=[col for col in columns_to_remove if col in df_merged.columns])
        
        # Handle sector columns first
        if 'Sector_xlsx' in df_merged.columns and 'Sector_csv' in df_merged.columns:
            df_merged['Sector'] = df_merged['Sector_xlsx'].fillna(df_merged['Sector_csv'])
            df_merged = df_merged.drop(['Sector_xlsx', 'Sector_csv'], axis=1)
        elif 'Sector_xlsx' in df_merged.columns:
            df_merged['Sector'] = df_merged['Sector_xlsx']
            df_merged = df_merged.drop(['Sector_xlsx'], axis=1)
        elif 'Sector_csv' in df_merged.columns:
            df_merged['Sector'] = df_merged['Sector_csv']
            df_merged = df_merged.drop(['Sector_csv'], axis=1)

        # Handle other duplicate columns
        duplicate_cols = [col for col in df_merged.columns if col.endswith('_xlsx') or col.endswith('_csv')]
        if duplicate_cols:
            for col in duplicate_cols:
                base_col = col[:-5] if col.endswith('_xlsx') else col[:-4]  # Remove suffix
                if col.endswith('_csv') and base_col != 'Sector':  # Skip Sector as it's already handled
                    if f"{base_col}_xlsx" in df_merged.columns:
                        df_merged[base_col] = df_merged[col].fillna(df_merged[f"{base_col}_xlsx"])
                        df_merged.drop([col, f"{base_col}_xlsx"], axis=1, inplace=True)
        
        # Fill missing values with reasonable defaults
        default_values = {
            'DivSafe': float('nan'),
            'Beta': float('nan'),
            'Payout': float('nan'),
            'DebtEBT': float('nan'),
            'DivGrw': float('nan'),
            'DivGrw5': float('nan'),
            'DivGrw10': float('nan'),
            'StrDiv': float('nan'),
            'StrUnt': float('nan'),
            'Sector': '',
            'DivTax': '',
            'Moat': '',
            'CapAll': '',
            'Style': '',
            'MSRate': float('nan'),
            'IntCov': float('nan'),
            'PFV': float('nan'),
            'FVU': ''
        }
        
        for col, default_val in default_values.items():
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].fillna(default_val)
        
        # Convert numeric columns
        numeric_columns = [
            'Yield', 'DivSafe',
            'Beta', 'Payout', 'DebtEBT',
            'DivGrw', 'DivGrw5', 'DivGrw10',
            'StrDiv', 'StrUnt', 'IntCov',
            'PFV'
        ]
        
        for col in numeric_columns:
            if col in df_merged.columns:
                df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')
        
        # Round numeric fields to two decimal places
        numeric_fields_to_round = ['Yield', 'DivSafe', 'Beta', 'Payout',
                                 'DebtEBT', 'DivGrw', 'DivGrw5', 'DivGrw10',
                                 'StrDiv', 'StrUnt', 'IntCov',
                                 'PFV']
        
        for col in numeric_fields_to_round:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].round(2)
        
        # Standardize sector names again in case any were missed
        if 'Sector' in df_merged.columns:
            df_merged['Sector'] = df_merged['Sector'].replace(sector_mapping)
        
        # Ensure categorical columns are strings
        categorical_columns = ['Sector', 'DivTax', 'FVU']
        for col in categorical_columns:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].fillna('')
        
        # Convert MSRate to stars
        def number_to_stars(n):
            if pd.isna(n):
                return ''
            return '★' * int(n)
        
        if 'MSRate' in df_merged.columns:
            df_merged['MSRate'] = df_merged['MSRate'].apply(number_to_stars)
        
        # Apply value transformations
        df_merged = apply_value_transformations(df_merged)
        
        return df_merged, None
        
    except Exception as e:
        return None, str(e)

def create_moat_scatter(df, x_col, x_title, title, jitter_x_scale, hover_x_label, hover_value_col, hover_value_label, hover_extra_col=None, hover_extra_label=None):
    """Create a scatter plot with moat-based coloring and styling."""
    if "Yield" not in df.columns or x_col not in df.columns:
        return None
        
    fig = go.Figure()
    
    # Create a trace for each moat category
    for moat in sorted(df['Moat'].unique()):
        mask = df['Moat'] == moat
        df_moat = df[mask]
        
        # Add small random jitter to positions to reduce overlap
        jitter_x = np.random.normal(0, jitter_x_scale, len(df_moat))
        jitter_y = np.random.normal(0, 0.1, len(df_moat))
        
        # Prepare customdata columns
        customdata_cols = ['Name', hover_value_col, 'Sector', 'DivSafe']
        if hover_extra_col:
            customdata_cols.append(hover_extra_col)
        
        fig.add_trace(go.Scatter(
            x=df_moat[x_col] + jitter_x,
            y=df_moat["Yield"] + jitter_y,
            mode='text',
            text="<b>" + df_moat["Ticker"] + "</b>",
            textfont=dict(
                size=10,
                color='#0066cc' if moat == 'Wid' else ('#99ccff' if moat == 'Nar' else '#e6e6e6')
            ),
            name=moat if moat != '' else 'Unknown',
            hovertemplate=(
                "<b>%{text}</b><br>" +
                f"{hover_x_label}: %{{x:.2f}}<br>" +
                "Yield: %{y:.2f}%<br>" +
                f"Moat: {moat}<br>" +
                "Name: %{customdata[0]}<br>" +
                f"{hover_value_label}: %{{customdata[1]}}<br>" +
                "Div Safety: %{customdata[3]:.2f}<br>" +
                "Sector: %{customdata[2]}<br>" +
                (f"{hover_extra_label}: %{{customdata[4]}}<br>" if hover_extra_col else "") +
                "<extra></extra>"
            ),
            customdata=df_moat[customdata_cols].values
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title="Yield (%)",
        showlegend=True,
        legend_title="Economic Moat",
        height=700  # Increase height to reduce vertical overlap
    )
    
    return fig

def create_yield_safety_scatter(df):
    """Create scatter plot of Yield vs Safety."""
    return create_moat_scatter(
        df=df,
        x_col="DivSafe",
        x_title="Safety Score",
        title="Yield vs Safety",
        jitter_x_scale=0.1,
        hover_x_label="Safety Score",
        hover_value_col="Payout",
        hover_value_label="Payout"
    )

def create_yield_pfv_scatter(df):
    """Create scatter plot of Yield vs Price/Fair Value."""
    return create_moat_scatter(
        df=df,
        x_col="PFV",
        x_title="Price/Fair Value",
        title="Yield vs Price/Fair Value",
        jitter_x_scale=0.02,
        hover_x_label="Price/Fair Value",
        hover_value_col="Val",
        hover_value_label="Valuation",
        hover_extra_col="FVU",
        hover_extra_label="Fair Value Uncertainty"
    )

def create_yield_growth_scatter(df):
    """Create scatter plot of Yield vs Dividend Growth."""
    # Check if required columns exist
    if "Grw5Y" not in df.columns:
        # Fall back to using DivGrw for both x-axis and hover data if Grw5Y is not available
        hover_value_col = "DivGrw"
        hover_value_label = "Dividend Growth"
    else:
        hover_value_col = "Grw5Y"
        hover_value_label = "5Y Dividend Growth"
        
    # Check if StrDiv exists
    hover_extra_col = "StrDiv" if "StrDiv" in df.columns else None
    hover_extra_label = "Dividend Streak (Years)" if hover_extra_col else None
    
    return create_moat_scatter(
        df=df,
        x_col="DivGrw", 
        x_title="Dividend Growth (%)",
        title="Yield vs Dividend Growth",
        jitter_x_scale=0.1,
        hover_x_label="Dividend Growth (%)",
        hover_value_col=hover_value_col,
        hover_value_label=hover_value_label,
        hover_extra_col=hover_extra_col,
        hover_extra_label=hover_extra_label
    )

def create_top_yield_bar(df):
    """Create bar chart of top yields."""
    if "Yield" not in df.columns:
        return None
        
    top_10 = df.nlargest(10, "Yield")
    fig = px.bar(
        top_10,
        x="Ticker",
        y="Yield",
        color="Sector",
        hover_data=["Name", "DivSafe"],
        title="Top 10 by Yield",
        labels={"Yield": "Yield (%)"}
    )
    return fig

import logging
for name, l in logging.root.manager.loggerDict.items():
    if "streamlit" in name:
        l.disabled = True

st.set_page_config(page_title="Screener", layout="wide")
st.title("Screener")

# Load data and display source information
df, error = load_data()

if error:
    st.error(f"Error loading data: {error}")
    exit(1)
if df is None:
    exit(1)
    
# Sidebar filters
st.sidebar.markdown("<h1 style='font-size: 32px;'>Filters</h1>", unsafe_allow_html=True)

# Dividend Safety categorical filter
if "DivSafe" in df.columns:
    safety_options = ["Very Safe (>80)", "Safe (60-80)", "None"]
    selected_safety = st.sidebar.multiselect("Safety Rating", safety_options, default=["Very Safe (>80)"])
    if selected_safety:
        safety_conditions = []
        for option in selected_safety:
            if option == "Very Safe (>80)":
                safety_conditions.append(df["DivSafe"] > 80)
            elif option == "Safe (60-80)":
                safety_conditions.append(df["DivSafe"].between(60, 80))
            elif option == "None":
                safety_conditions.append(df["DivSafe"].isna())
        df = df[pd.concat(safety_conditions, axis=1).any(axis=1)]

# Sector filter
if "Sector" in df.columns:
    sector_options = sorted([x for x in df["Sector"].unique() if x != ""])
    selected_sectors = st.sidebar.multiselect("Sectors", sector_options)
    if selected_sectors:
        df = df[df["Sector"].isin(selected_sectors)]

# Moat Rating filter
if "Moat" in df.columns:
    moat_options = sorted([x for x in df["Moat"].unique() if x != ""]) + ["Unknown"]
    selected_moats = st.sidebar.multiselect("Moats", moat_options, default=["Wid", "Nar"])
    if selected_moats:
        if "Unknown" in selected_moats:
            selected_moats.remove("Unknown")
            df = df[df["Moat"].isin(selected_moats) | (df["Moat"] == "")]
        else:
            df = df[df["Moat"].isin(selected_moats)]

# Dividend Taxation filter
if "DivTax" in df.columns:
    taxation_options = sorted([x for x in df["DivTax"].unique() if x != ""])
    selected_taxations = st.sidebar.multiselect("Div Taxes", taxation_options, default=["Qualified"])
    if selected_taxations:
        df = df[df["DivTax"].isin(selected_taxations)]

# Morningstar Rating filter
if "MSRate" in df.columns:
    star_options = ['★', '★★', '★★★', '★★★★', '★★★★★']
    selected_stars = st.sidebar.multiselect("Morningstar Rating", star_options, default=['★★★★', '★★★★★'])
    if selected_stars:
        df = df[df["MSRate"].isin(selected_stars)]

# Dividend Yield filter (only show if there are stocks with yield data)
if "Yield" in df.columns and df["Yield"].notna().any():
    min_yield = round(float(df["Yield"].min()), 2)
    max_yield = round(float(df["Yield"].max()), 2)
    selected_min, selected_max = st.sidebar.slider(
        "Yield Range (%)",
        min_yield, max_yield,
        value=(min_yield, max_yield),
        step=0.01
    )
    # Only filter stocks that have a yield value
    yield_mask = df["Yield"].notna()
    df = df[~yield_mask | (yield_mask & df["Yield"].between(selected_min, selected_max))]

# Display data source information
if df is not None:
    xlsx_file = get_latest_file(Path.home() / "Downloads", "*.xlsx")
    csv_file = get_latest_file(Path.home() / "Downloads", "*.csv")
    if xlsx_file and csv_file:
        st.caption(f"Data sources: {xlsx_file.name} and {csv_file.name}")
    st.header(f"Stocks ({len(df)})")
else:
    st.header("Stocks")

# Reorder columns to show most important first
important_cols = ['Ticker', 'Name', 'Sector', 'Yield', 
                    'DivSafe', 
                    'Moat',
                    'Beta',
                    'StrUnt',
                    'StrDiv',
                    'MSRate',
                    'Val',
                    'PFV',
                    'FVU',
                    'DivGrw',
                    'DivGrw5',
                    'DivGrw10',
                    'RecDiv',
                    'IntCov',
                    ]
cols = [col for col in important_cols if col in df.columns]
other_cols = [col for col in df.columns if col not in important_cols]
df = df[cols + other_cols]

# Replace NaN with empty string for display
display_df = df.copy()
display_df = display_df.fillna('')

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    height=400
)

# Display charts in tabs
st.header("Charts")

tab1, tab2, tab3, tab4 = st.tabs(["Yield vs Safety", "Yield vs Price/Fair Value", "Top Yields", "Yield vs Growth"])

with tab1:
    yield_safety_fig = create_yield_safety_scatter(df)
    if yield_safety_fig:
        st.plotly_chart(yield_safety_fig, use_container_width=True)
    else:
        st.warning("Missing required columns for Yield vs Safety chart")

with tab2:
    yield_pfv_fig = create_yield_pfv_scatter(df)
    if yield_pfv_fig:
        st.plotly_chart(yield_pfv_fig, use_container_width=True)
    else:
        st.warning("Missing required columns for Yield vs Price/Fair Value chart")

with tab3:
    top_yield_fig = create_top_yield_bar(df)
    if top_yield_fig:
        st.plotly_chart(top_yield_fig, use_container_width=True)
    else:
        st.warning("Missing required columns for Top Yields chart")

with tab4:
    yield_growth_fig = create_yield_growth_scatter(df)
    if yield_growth_fig:
        st.plotly_chart(yield_growth_fig, use_container_width=True)
    else:
        st.warning("Missing required columns for Yield vs Dividend Growth chart")

# Export functionality
if st.button("Export Filtered Data"):
    csv = df.to_csv(index=False)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dividend_analysis_{timestamp}.csv"
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )

if __name__ == "__main__":
    if "__streamlitmagic__" not in locals():
        import streamlit.web.bootstrap

        streamlit.web.bootstrap.run(__file__, False, [], {})
