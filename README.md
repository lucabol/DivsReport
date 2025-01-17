# Dividend Investor Data Analysis Tool

An interactive web application for analyzing dividend stock data by merging and visualizing information from Excel and CSV files.

## Features

- Automatically loads and merges the latest `.xlsx` and `.csv` files from your Downloads directory
- Interactive data table with sorting and filtering capabilities
- Dynamic visualizations:
  - Dividend Yield vs Safety scatter plot
  - Dividend Yield vs P/E Ratio scatter plot
  - Top 10 stocks by dividend yield bar chart
- Filtering options:
  - Dividend yield range
  - Moat rating selection
- Export filtered data to CSV

## Installation

1. Ensure you have Python 3.8+ installed
2. Clone this repository
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

From the project directory, run:
```bash
streamlit run src/app.py
```

The application will open in your default web browser.

## Using Sample Data

Sample data files are provided in the `data/` directory:
- `sample_dividend_data.csv`: Contains dividend yields, P/E ratios, moat ratings, etc.
- `sample_financial_metrics.xlsx`: Contains market cap, growth rates, and other metrics

To test the application with sample data:
1. Copy both sample files from the `data/` directory
2. Paste them into your Downloads folder
3. Run the application

## Data File Requirements

Your data files must include:

### Required Columns
- Both files must have a "Ticker" column for merging

### Recommended Columns for Full Functionality
- Dividend Yield
- P/E Ratio
- Moat Rating
- Credit Rating
- Dividend Safety
- Market Cap
- Revenue Growth
- Payout Ratio
- Years of Dividend Growth
- Sector

## Features in Detail

### 1. Data Loading
- Automatically detects the most recent `.xlsx` and `.csv` files
- Validates file structure and required columns
- Merges datasets using stock tickers as keys

### 2. Interactive Table
- Sort any column by clicking headers
- Filter data using the sidebar controls
- View all available metrics in a clean, organized format

### 3. Visualizations
- **Yield vs Safety Scatter Plot**: Helps identify stable, high-yield investments
- **Yield vs P/E Scatter Plot**: Identifies potentially undervalued high-yield stocks
- **Top Yield Bar Chart**: Quick view of highest-yielding stocks

### 4. Data Export
- Export filtered dataset to CSV
- Includes timestamp in filename
- Preserves all applied filters

## Error Handling

The application includes robust error handling for:
- Missing files in Downloads directory
- Missing required columns
- Invalid data formats
- File access issues

Clear error messages guide users to resolve any issues that arise.
