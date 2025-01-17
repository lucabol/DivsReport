import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
from datetime import datetime

def get_latest_file(directory: Path, pattern: str) -> Path:
    """Get the most recent file matching the pattern from directory."""
    files = list(directory.glob(pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def load_data():
    """Load and merge data from the latest xlsx and csv files."""
    downloads_dir = Path.home() / "Downloads"
    
    # Get latest files
    xlsx_file = get_latest_file(downloads_dir, "*.xlsx")
    csv_file = get_latest_file(downloads_dir, "*.csv")
    
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
            'Market Cap': 'Mkt Cap (M)',
            'Industry': 'Sub Sec',
            'Dividend Yield (Forward)': 'Fwd Yld',
            'Dividend Yield (Trailing)': 'Trail Yld',
            'Dividend Safety': 'Div Safety',
            'Price/Earnings (Forward)': 'P/E Fwd',
            'Price/Cash Flow': 'P/CF',
            'Sub Sector': 'Sub Sec',
            'Market Cap (Millions)': 'Mkt Cap (M)',
            'Expected Price': 'Exp Price',
            '% From Expected Price': '% From Exp',
            '5 Year Average Dividend Yield': '5Y Avg Yld',
            '% Above 5 Year Average Dividend Yield': '% Above 5Y Avg',
            '5 Year Average P/E Ratio': '5Y Avg P/E',
            'Dividend Growth (Latest)': 'Div Growth',
            '5 Year Dividend Growth': '5Y Div Growth',
            '20 Year Dividend Growth': '20Y Div Growth',
            'Dividend Growth Streak (Years)': 'Div Streak',
            'Uninterrupted Dividend Streak (Years)': 'Unint Div Streak',
            'Payment Frequency': 'Pay Freq',
            'Payment Schedule': 'Pay Sched',
            'Dividend Taxation': 'Div Tax',
            'Recession Dividend': 'Rec Div',
            'Recession Return': 'Rec Ret',
            'Ex Dividend Date': 'Ex Div Date',
            'Years Of Positive Free Cash Flow In Last 10': 'FCF Yrs',
            'Payout Ratio': 'Payout',
            'Net Debt To Capital': 'Debt/Cap',
            'Net Debt To EBITDA': 'Debt/EBITDA',
            'P/E Ratio': 'P/E',
            'Capital Allocation': 'Cap Alloc',
            'Morningstar Rating for Stocks': 'MS Rating',
            'Fair Value Uncertainty': 'FV Uncert',
            'Price/Fair Value': 'P/FV',
            'Valuation': 'Val'
        }
        
        # Rename columns in both dataframes
        df_xlsx = df_xlsx.rename(columns=column_renames)
        df_csv = df_csv.rename(columns=column_renames)
        
        # Convert Market Cap to millions if needed (assuming it's in full numbers)
        if 'Mkt Cap (M)' in df_xlsx.columns:
            df_xlsx['Mkt Cap (M)'] = df_xlsx['Mkt Cap (M)'].astype(float) / 1_000_000
        
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
        df_merged['Yield'] = df_merged['Fwd Yld']
        df_merged.loc[df_merged['Yield'].isna(), 'Yield'] = df_merged['Trail Yld']
        df_merged.loc[df_merged['Yield'].isna(), 'Yield'] = df_merged['Dividend Yield']
        
        # Drop other yield columns after creating the main Yield column
        yield_columns = ['Fwd Yld', 'Trail Yld', 'Dividend Yield']
        df_merged = df_merged.drop(columns=[col for col in yield_columns if col in df_merged.columns])
        
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
            'P/E': float('nan'),
            'Div Safety': float('nan'),
            'Mkt Cap (M)': float('nan'),
            'Beta': float('nan'),
            'Payout': float('nan'),
            'Debt/Cap': float('nan'),
            'Debt/EBITDA': float('nan'),
            'Div Growth': float('nan'),
            '5Y Div Growth': float('nan'),
            '20Y Div Growth': float('nan'),
            'Div Streak': float('nan'),
            'Unint Div Streak': float('nan'),
            'Sector': 'Unknown',
            'Sub Sec': 'Unknown',
            'Pay Sched': 'Unknown',
            'Div Tax': 'Unknown',
            'Moat': 'Unknown',
            'Cap Alloc': 'Unknown',
            'Style Box': 'Unknown',
            'MS Rating': float('nan')
        }
        
        for col, default_val in default_values.items():
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].fillna(default_val)
        
        # Convert numeric columns
        numeric_columns = [
            'Yield', 'P/E', 'Div Safety', 'Mkt Cap (M)',
            'Beta', 'Payout', 'Debt/Cap', 'Debt/EBITDA',
            'Div Growth', '5Y Div Growth', '20Y Div Growth',
            'Div Streak', 'Unint Div Streak'
        ]
        
        for col in numeric_columns:
            if col in df_merged.columns:
                df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')
        
        # Round Yield to two decimal places
        if 'Yield' in df_merged.columns:
            df_merged['Yield'] = df_merged['Yield'].round(2)
        
        # Standardize sector names again in case any were missed
        if 'Sector' in df_merged.columns:
            df_merged['Sector'] = df_merged['Sector'].replace(sector_mapping)
        
        # Ensure categorical columns are strings
        categorical_columns = ['Sector', 'Sub Sec', 'Pay Sched', 'Div Tax']
        for col in categorical_columns:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].fillna('')
        
        return df_merged, None
        
    except Exception as e:
        return None, str(e)

def create_yield_safety_scatter(df):
    """Create scatter plot of Yield vs Safety."""
    if "Yield" not in df.columns or "Div Safety" not in df.columns:
        return None
        
    fig = px.scatter(
        df,
        x="Div Safety",
        y="Yield",
        color="Sector",
        hover_data=["Ticker", "Name", "Payout"],
        title="Yield vs Safety",
        labels={
            "Div Safety": "Safety Score",
            "Yield": "Yield (%)"
        }
    )
    return fig

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
        hover_data=["Name", "Div Safety"],
        title="Top 10 by Yield",
        labels={"Yield": "Yield (%)"}
    )
    return fig

def main():
    st.set_page_config(page_title="Dividend Investor Data Analysis", layout="wide")
    st.title("Dividend Investor Data Analysis")
    
    # Load data and display source information
    df, error = load_data()

    if error:
        st.error(f"Error loading data: {error}")
        return
    if df is None:
        return
        
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Add filters
    st.sidebar.subheader("Numeric Filters")
    
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
    
    # Dividend Safety filter (only show if there are stocks with safety data)
    if "Div Safety" in df.columns and df["Div Safety"].notna().any():
        min_safety = float(df["Div Safety"].min())
        max_safety = float(df["Div Safety"].max())
        selected_min_safety, selected_max_safety = st.sidebar.slider(
            "Safety Score",
            min_safety, max_safety,
            (min_safety, max_safety)
        )
        # Only filter stocks that have a Safety value
        safety_mask = df["Div Safety"].notna()
        df = df[~safety_mask | (safety_mask & df["Div Safety"].between(selected_min_safety, selected_max_safety))]
    
    st.sidebar.subheader("Categorical Filters")
    
    # Sector filter
    if "Sector" in df.columns:
        sector_options = ["All"] + sorted([x for x in df["Sector"].unique() if x not in ["", "Unknown"]])
        selected_sector = st.sidebar.selectbox("Sector", sector_options)
        if selected_sector != "All":
            df = df[df["Sector"] == selected_sector]
    
    # Moat Rating filter
    if "Moat" in df.columns:
        moat_options = ["All"] + sorted([x for x in df["Moat"].unique() if x not in ["", "Unknown"]])
        selected_moat = st.sidebar.selectbox("Moat", moat_options)
        if selected_moat != "All":
            df = df[df["Moat"] == selected_moat]
    
    # Dividend Taxation filter
    if "Div Tax" in df.columns:
        taxation_options = ["All"] + sorted([x for x in df["Div Tax"].unique() if x not in ["", "Unknown"]])
        selected_taxation = st.sidebar.selectbox("Div Tax", taxation_options)
        if selected_taxation != "All":
            df = df[df["Div Tax"] == selected_taxation]
    
    # Display data source information
    if df is not None:
        xlsx_file = get_latest_file(Path.home() / "Downloads", "*.xlsx")
        csv_file = get_latest_file(Path.home() / "Downloads", "*.csv")
        if xlsx_file and csv_file:
            st.caption(f"Data sources: {xlsx_file.name} and {csv_file.name}")
        st.header(f"Stock Data Table ({len(df)} stocks)")
    else:
        st.header("Stock Data Table")
    
    # Reorder columns to show most important first
    important_cols = ['Ticker', 'Name', 'Sector', 'Yield', 
                     'Div Safety', 
                     'Moat',
                     'Beta', 'Rec Div',
                     'Unint Div Streak',
                     'Debt/EBITDA',
                     'MS Rating',
                     'Val'
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
    st.header("Analysis Charts")
    
    tab1, tab2 = st.tabs(["Yield vs Safety", "Top Yields"])
    
    with tab1:
        yield_safety_fig = create_yield_safety_scatter(df)
        if yield_safety_fig:
            st.plotly_chart(yield_safety_fig, use_container_width=True)
        else:
            st.warning("Missing required columns for Yield vs Safety chart")
    
    with tab2:
        top_yield_fig = create_top_yield_bar(df)
        if top_yield_fig:
            st.plotly_chart(top_yield_fig, use_container_width=True)
        else:
            st.warning("Missing required columns for Top Yields chart")
    
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
    main()
