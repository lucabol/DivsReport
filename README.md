# Dividend Investor Data Analysis Tool

A Streamlit web application for analyzing dividend stock data by merging and visualizing information from Excel and CSV files.

## Features

- Automatically loads and merges the latest `for-divs*.xlsx` and `ForDivs*.csv` files from your Downloads directory
- Interactive data table with:
  - Automatic column renaming for clarity
  - Smart data type conversion and formatting
  - Standardized sector names
  - Morningstar ratings displayed as stars
  - Abbreviated values for better readability
- Dynamic visualizations:
  - Yield vs Safety scatter plot with moat-based coloring
  - Yield vs Price/Fair Value scatter plot
  - Top 10 stocks by dividend yield bar chart
- Comprehensive filtering options:
  - Safety Rating (Very Safe, Safe)
  - Sectors
  - Economic Moat (Wide, Narrow)
  - Dividend Taxation
  - Morningstar Rating
  - Dividend Yield range
- Export filtered data to timestamped CSV

## Running the Application

From the project directory, run:
```bash
uv run .\src\app.py
```

The application will open in your default web browser.

## Data File Requirements

The application expects two files in your Downloads directory:
- An Excel file (*.xlsx) with filename starting with "for-divs"
- A CSV file (*.csv) with filename starting with "ForDivs"

### Required Columns
- Both files must have a "Ticker" column for merging

### Key Columns Used
- Dividend Yield (Forward/Trailing)
- Dividend Safety
- Economic Moat
- Price/Fair Value
- Fair Value Uncertainty
- Morningstar Rating
- Sector
- Beta
- Payout Ratio
- Debt to EBITDA
- Interest Coverage
- Dividend Growth (Latest, 5Y, 20Y)
- Dividend Streaks (Growth, Uninterrupted)
- Recession Dividend Performance
- Dividend Taxation

## Data Processing Features

### Automatic Data Cleaning
- Standardizes sector names across files
- Cleans ticker symbols (removes whitespace, converts to uppercase)
- Rounds numeric values for consistency
- Converts Morningstar ratings to star symbols
- Abbreviates categorical values for better display

### Smart Data Merging
- Merges Excel and CSV data using ticker symbols as keys
- Handles duplicate columns intelligently
- Prioritizes forward yield over trailing yield
- Standardizes column names for clarity

### Error Handling
- Validates file existence in Downloads directory
- Checks for required columns
- Handles missing or invalid data gracefully
- Provides clear error messages for common issues

## Export Functionality
- Exports filtered dataset to CSV
- Includes timestamp in filename
- Preserves all applied filters and data transformations
