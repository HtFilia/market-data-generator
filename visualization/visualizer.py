import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


def load_data() -> pd.DataFrame:
    """Load and prepare data from simulation results CSV file."""
    data_path = Path("data/simulation_results.csv")
    if not data_path.exists():
        st.error(f"Simulation results file not found at {data_path}")
        st.stop()
    
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def main():
    st.set_page_config(page_title="Market Data Visualizer", layout="wide")
    st.title("Market Data Visualizer")

    # Load data
    df = load_data()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Sector filter
    sectors = ["All"] + sorted(df['sector'].unique().tolist())
    selected_sector = st.sidebar.selectbox("Sector", sectors)
    
    # Geography filter
    geographies = ["All"] + sorted(df['geography'].unique().tolist())
    selected_geo = st.sidebar.selectbox("Geography", geographies)
    
    # Asset filter
    assets = ["All"] + sorted(df['asset_id'].unique().tolist())
    selected_asset = st.sidebar.selectbox("Asset", assets)
    
    # Date range filter
    min_date = df['date'].min()
    max_date = df['date'].max()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_sector != "All":
        filtered_df = filtered_df[filtered_df['sector'] == selected_sector]
    
    if selected_geo != "All":
        filtered_df = filtered_df[filtered_df['geography'] == selected_geo]
    
    if selected_asset != "All":
        filtered_df = filtered_df[filtered_df['asset_id'] == selected_asset]
    
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['date'].dt.date >= start_date) &
            (filtered_df['date'].dt.date <= end_date)
        ]
    
    # Create plot
    fig = px.line(
        filtered_df,
        x='date',
        y='close',
        color='asset_id',
        title="Asset Prices Over Time",
        labels={'date': 'Date', 'close': 'Price', 'asset_id': 'Asset'}
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode='x unified',
        showlegend=True,
        legend_title="Assets"
    )
    
    # Display plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Display statistics
    st.subheader("Statistics")
    stats_df = filtered_df.groupby('asset_id')['close'].agg(['mean', 'std', 'min', 'max']).round(2)
    st.dataframe(stats_df)


if __name__ == "__main__":
    main() 