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
        
        # Rename Excel columns
        df_xlsx = df_xlsx.rename(columns={
            'Economic Moat': 'Moat Rating',
            'Market Cap': 'Market Cap (Millions)',
            'Industry': 'Sub Sector',
            'Dividend Yield (Forward)': 'Forward Yield',
            'Dividend Yield (Trailing)': 'Trailing Yield'
        })
        
        # Convert Market Cap to millions if needed (assuming it's in full numbers)
        if 'Market Cap (Millions)' in df_xlsx.columns:
            df_xlsx['Market Cap (Millions)'] = df_xlsx['Market Cap (Millions)'].astype(float) / 1_000_000
        
        # Standardize sector names in both dataframes before merging
        sector_mapping = {
            'Consumer Defensive': 'Consumer Stable',
            'Consumer Cyclical': 'Consumer Discretionary',
            'Communication Services': 'Communications',
            'Basic Materials': 'Materials'
        }
        
        if 'Sector' in df_xlsx.columns:
            df_xlsx['Sector'] = df_xlsx['Sector'].replace(sector_mapping)
        if 'Sector' in df_csv.columns:
            df_csv['Sector'] = df_csv['Sector'].replace(sector_mapping)
        
        # Merge datasets with outer join and indicator
        df_merged = pd.merge(df_xlsx, df_csv, on="Ticker", how="outer", suffixes=('_xlsx', '_csv'))
        
        # st.write("Merged", sorted(list(df_merged.columns)))
        # Create Dividend Yield column with precedence:
        # 1. Forward Yield from Excel
        # 2. Trailing Yield from Excel
        # 3. Dividend Yield from CSV
        df_merged['Yield'] = df_merged['Forward Yield']
        df_merged.loc[df_merged['Yield'].isna(), 'Yield'] = df_merged['Trailing Yield']
        df_merged.loc[df_merged['Yield'].isna(), 'Yield'] = df_merged['Dividend Yield']
        
        # Drop other yield columns after creating the main Yield column
        yield_columns = ['Forward Yield', 'Trailing Yield', 'Dividend Yield']
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
            'P/E Ratio': float('nan'),
            'Dividend Safety': float('nan'),
            'Market Cap (Millions)': float('nan'),
            'Beta': float('nan'),
            'Payout Ratio': float('nan'),
            'Net Debt To Capital': float('nan'),
            'Net Debt To EBITDA': float('nan'),
            'Dividend Growth (Latest)': float('nan'),
            '5 Year Dividend Growth': float('nan'),
            '20 Year Dividend Growth': float('nan'),
            'Dividend Growth Streak (Years)': float('nan'),
            'Uninterrupted Dividend Streak (Years)': float('nan'),
            'Sector': 'Unknown',
            'Sub Sector': 'Unknown',
            'Payment Schedule': 'Unknown',
            'Dividend Taxation': 'Unknown',
            'Moat Rating': 'Unknown',
            'Capital Allocation': 'Unknown',
            'Stock Style Box': 'Unknown',
            'Morningstar Rating for Stocks': float('nan')
        }
        
        for col, default_val in default_values.items():
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].fillna(default_val)
        
        # Convert numeric columns
        numeric_columns = [
            'Yield', 'P/E Ratio', 'Dividend Safety', 'Market Cap (Millions)',
            'Beta', 'Payout Ratio', 'Net Debt To Capital', 'Net Debt To EBITDA',
            'Dividend Growth (Latest)', '5 Year Dividend Growth', '20 Year Dividend Growth',
            'Dividend Growth Streak (Years)', 'Uninterrupted Dividend Streak (Years)'
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
        categorical_columns = ['Sector', 'Sub Sector', 'Payment Schedule', 'Dividend Taxation']
        for col in categorical_columns:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].fillna('')
        
        return df_merged, None
        
    except Exception as e:
        return None, str(e)

def create_yield_safety_scatter(df):
    """Create scatter plot of Yield vs Dividend Safety."""
    if "Yield" not in df.columns or "Dividend Safety" not in df.columns:
        return None
        
    fig = px.scatter(
        df,
        x="Dividend Safety",
        y="Yield",
        color="Sector",
        hover_data=["Ticker", "Name", "Payout Ratio"],
        title="Dividend Yield vs Safety",
        labels={
            "Dividend Safety": "Dividend Safety Score",
            "Yield": "Dividend Yield (%)"
        }
    )
    return fig

def create_yield_pe_scatter(df):
    """Create scatter plot of Yield vs P/E Ratio."""
    if "Yield" not in df.columns or "P/E Ratio" not in df.columns:
        return None
        
    fig = px.scatter(
        df,
        x="P/E Ratio",
        y="Yield",
        color="Sector",
        hover_data=["Ticker", "Name", "Market Cap (Millions)"],
        title="Dividend Yield vs P/E Ratio",
        labels={
            "P/E Ratio": "P/E Ratio",
            "Yield": "Dividend Yield (%)"
        }
    )
    return fig

def create_top_yield_bar(df):
    """Create bar chart of top 10 dividend yields."""
    if "Yield" not in df.columns:
        return None
        
    top_10 = df.nlargest(10, "Yield")
    fig = px.bar(
        top_10,
        x="Ticker",
        y="Yield",
        color="Sector",
        hover_data=["Name", "Dividend Safety"],
        title="Top 10 Stocks by Dividend Yield",
        labels={"Yield": "Dividend Yield (%)"}
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
            "Dividend Yield Range (%)",
            min_yield, max_yield,
            value=(min_yield, max_yield),
            step=0.01
        )
        # Only filter stocks that have a yield value
        yield_mask = df["Yield"].notna()
        df = df[~yield_mask | (yield_mask & df["Yield"].between(selected_min, selected_max))]
    
    # P/E Ratio filter (only show if there are stocks with P/E data)
    if "P/E Ratio" in df.columns and df["P/E Ratio"].notna().any():
        min_pe = float(df["P/E Ratio"].min())
        max_pe = float(df["P/E Ratio"].max())
        selected_min_pe, selected_max_pe = st.sidebar.slider(
            "P/E Ratio Range",
            min_pe, max_pe,
            (min_pe, max_pe)
        )
        # Only filter stocks that have a P/E value
        pe_mask = df["P/E Ratio"].notna()
        df = df[~pe_mask | (pe_mask & df["P/E Ratio"].between(selected_min_pe, selected_max_pe))]
    
    # Dividend Safety filter (only show if there are stocks with safety data)
    if "Dividend Safety" in df.columns and df["Dividend Safety"].notna().any():
        min_safety = float(df["Dividend Safety"].min())
        max_safety = float(df["Dividend Safety"].max())
        selected_min_safety, selected_max_safety = st.sidebar.slider(
            "Dividend Safety Score",
            min_safety, max_safety,
            (min_safety, max_safety)
        )
        # Only filter stocks that have a Safety value
        safety_mask = df["Dividend Safety"].notna()
        df = df[~safety_mask | (safety_mask & df["Dividend Safety"].between(selected_min_safety, selected_max_safety))]
    
    st.sidebar.subheader("Categorical Filters")
    
    # Sector filter
    if "Sector" in df.columns:
        sector_options = ["All"] + sorted([x for x in df["Sector"].unique() if x not in ["", "Unknown"]])
        selected_sector = st.sidebar.selectbox("Sector", sector_options)
        if selected_sector != "All":
            df = df[df["Sector"] == selected_sector]
    
    # Sub Sector filter
    if "Sub Sector" in df.columns:
        subsector_options = ["All"] + sorted([x for x in df["Sub Sector"].unique() if x not in ["", "Unknown"]])
        selected_subsector = st.sidebar.selectbox("Sub Sector", subsector_options)
        if selected_subsector != "All":
            df = df[df["Sub Sector"] == selected_subsector]
    
    # Payment Schedule filter
    if "Payment Schedule" in df.columns:
        schedule_options = ["All"] + sorted([x for x in df["Payment Schedule"].unique() if x not in ["", "Unknown"]])
        selected_schedule = st.sidebar.selectbox("Payment Schedule", schedule_options)
        if selected_schedule != "All":
            df = df[df["Payment Schedule"] == selected_schedule]
    
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
    important_cols = ['Ticker', 'Name', 'Sector', 'Sub Sector', 'Yield', 
                     'P/E Ratio', 'Dividend Safety', 
                     'Market Cap (Millions)', 'Payout Ratio', 'Dividend Growth Streak (Years)',
                     'Payment Schedule', 'Dividend Taxation']
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
    
    tab1, tab2, tab3 = st.tabs(["Yield vs Safety", "Yield vs P/E", "Top Yields"])
    
    with tab1:
        yield_safety_fig = create_yield_safety_scatter(df)
        if yield_safety_fig:
            st.plotly_chart(yield_safety_fig, use_container_width=True)
        else:
            st.warning("Missing required columns for Yield vs Safety chart")
    
    with tab2:
        yield_pe_fig = create_yield_pe_scatter(df)
        if yield_pe_fig:
            st.plotly_chart(yield_pe_fig, use_container_width=True)
        else:
            st.warning("Missing required columns for Yield vs P/E chart")
    
    with tab3:
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
