import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import random
import re

file_path = "ccc-phase3-public.csv"
US_POPULATION = 340_100_000

@st.cache_data
def load_data():
    df = pd.read_csv(file_path, encoding='latin1', low_memory=False)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['size_mean'] = pd.to_numeric(df['size_mean'], errors='coerce')
    df['participants_numeric'] = df['size_mean']
    df['targets'] = df['targets'].astype(str).str.lower()
    df['organizations'] = df['organizations'].astype(str).str.lower()
    df['state'] = df['state'].astype('category')
    df['targets'] = df['targets'].astype('category')
    df['organizations'] = df['organizations'].astype('category')
    for col in [
        'participant_injuries', 'police_injuries', 'arrests',
        'participant_deaths', 'police_deaths'
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def jitter_coords(df, lat_col='lat', lon_col='lon', jitter_amount=0.05):
    df = df.copy().reset_index(drop=True)
    coords = df[[lat_col, lon_col]].astype(str).agg('_'.join, axis=1)
    counts = coords.value_counts()
    dup_coords = counts[counts > 1].index
    for coord in dup_coords:
        idxs = df.index[coords == coord].tolist()
        for offset, i in enumerate(idxs):
            if offset == 0:
                continue
            angle = random.uniform(0, 2 * np.pi)
            radius = random.uniform(0.0005, jitter_amount)
            df.at[i, lat_col] += np.cos(angle) * radius
            df.at[i, lon_col] += np.sin(angle) * radius
    return df

def filter_data(
    df, start_date, end_date, size_filter, trump_filter, org_search, state_filter,
    any_outcomes_filter
):
    dff = df
    mask = pd.Series(True, index=dff.index)
    if start_date and end_date:
        mask &= (dff['date'] >= start_date) & (dff['date'] <= end_date)
    if size_filter == 'has':
        mask &= dff['size_mean'].notna()
    elif size_filter == 'no':
        mask &= dff['size_mean'].isna()
    if trump_filter:
        mask &= dff['targets'].str.contains("trump", na=False)
    if org_search:
        org_terms = [o.strip().lower() for o in org_search.split(',') if o.strip()]
        if org_terms:
            pattern = '|'.join([re.escape(org) for org in org_terms])
            mask &= dff['organizations'].str.contains(pattern, na=False, regex=True)
    if state_filter:
        mask &= dff['state'].isin(state_filter)
    for outcome in any_outcomes_filter:
        if outcome == 'arrests_any':
            mask &= dff['arrests'].notna() & (dff['arrests'].astype(str).str.lower() != 'unspecified') & (pd.to_numeric(dff['arrests'], errors='coerce').fillna(0) > 0)
        elif outcome == 'participant_injuries_any':
            mask &= dff['participant_injuries'].notna() & (dff['participant_injuries'].astype(str).str.lower() != 'unspecified') & (pd.to_numeric(dff['participant_injuries'], errors='coerce').fillna(0) > 0)
        elif outcome == 'police_injuries_any':
            mask &= dff['police_injuries'].notna() & (pd.to_numeric(dff['police_injuries'], errors='coerce').fillna(0) > 0)
        elif outcome == 'property_damage_any':
            mask &= dff['property_damage'].notna() & (dff['property_damage'].astype(str).str.strip() != "")
        elif outcome == 'participant_deaths_any':
            mask &= dff['participant_deaths'].notna() & (pd.to_numeric(dff['participant_deaths'], errors='coerce').fillna(0) > 0)
        elif outcome == 'police_deaths_any':
            mask &= dff['police_deaths'].notna() & (pd.to_numeric(dff['police_deaths'], errors='coerce').fillna(0) > 0)
    dff = dff.loc[mask]
    return dff

def aggregate_events_for_map(dff_map):
    df_map = dff_map.copy()
    def best_location(row):
        loc = str(row.get('location', '')).strip()
        if loc and loc.lower() != 'nan':
            return loc
        loc2 = str(row.get('locality', '')).strip()
        if loc2 and loc2.lower() != 'nan':
            return loc2
        state = row.get('state', 'Unknown')
        date = row['date'].date() if pd.notnull(row.get('date')) else ''
        return f"{state}, {date}"
    df_map['location_label'] = df_map.apply(best_location, axis=1)
    df_map['location_label'] = df_map['location_label'].replace('', np.nan).fillna('Unknown')
    df_map['event_label'] = df_map.apply(
        lambda row: f"{row.get('title', 'No Title')} ({row['date'].date() if pd.notnull(row['date']) else ''})<br>Org: {row.get('organizations', 'Unknown')}", axis=1
    )
    df_map = df_map.dropna(subset=['lat', 'lon'])
    agg = df_map.groupby('location_label').agg(
        lat=('lat', 'first'),
        lon=('lon', 'first'),
        count=('title', 'size'),
        event_list=('event_label', lambda x: "<br><br>".join(x)),
        title=('title', lambda x: "; ".join(x.astype(str))),
        size_mean=('size_mean', lambda x: x.mean() if x.notna().any() else np.nan)
    ).reset_index()
    def format_mean_size(val):
        return f"{int(val):,}" if pd.notnull(val) else "NA"

    agg['hover'] = agg.apply(
        lambda row: (
            f"<b>{row['location_label']}</b><br>"
            f"Events at this site: {row['count']}<br><br>"
            f"<b>Events:</b><br>{row['event_list']}<br><br>"
            f"<b>Mean Size:</b> {format_mean_size(row['size_mean'])}"
        ), axis=1
    )
    agg['text'] = agg['location_label']
    return agg

def build_hover(row):
    def safe(val):
        return "NA" if pd.isnull(val) else val
    details = [
        f"<b>{safe(row.get('location_label',''))}</b>",
        f"Title: {safe(row.get('title',''))}",
        f"Date: {safe(row.get('date',''))}",
        f"Locality: {safe(row.get('locality',''))}",
        f"Location: {safe(row.get('location',''))}",
        f"Organizations: {safe(row.get('organizations',''))}",
        f"Participants: {safe(row.get('size_mean','')):.0f}" if pd.notnull(row.get('size_mean','')) else "Participants: NA",
        f"Notables: {safe(row.get('notables',''))}",
        f"Targets: {safe(row.get('targets',''))}",
        f"Claims Summary: {safe(row.get('claims_summary',''))}",
        f"Participant Measures: {safe(row.get('participant_measures',''))}",
        f"Police Measures: {safe(row.get('police_measures',''))}",
        f"Participant Injuries: {safe(row.get('participant_injuries',''))}",
        f"Police Injuries: {safe(row.get('police_injuries',''))}",
        f"Arrests: {safe(row.get('arrests',''))}",
        f"Property Damage: {safe(row.get('property_damage',''))}",
        f"Notes: {safe(row.get('notes',''))}",
    ]
    return "<br>".join(details)

st.set_page_config(layout="wide", page_title="Protest Dashboard", page_icon="🗺️")

# --- Custom CSS for Wix-like Modern Look ---
st.markdown("""
    <style>
    html, body, .block-container {
        background-color: #F8F7F9 !important;
        color: #222 !important;
        font-family: 'helvetica-w01-roman', Helvetica, Arial, sans-serif !important;
    }
    .big-metric {
        font-size: 2.5rem;
        font-weight: 700;
        color: #AC3C3D;
        margin-bottom: 0.2rem;
        font-family: 'helvetica-w01-roman', Helvetica, Arial, sans-serif !important;
        letter-spacing: 0.05em;
    }
    .metric-label {
        font-size: 1.1rem;
        color: #244CC4;
        margin-bottom: 1.5rem;
        font-family: 'helvetica-w01-roman', Helvetica, Arial, sans-serif !important;
        letter-spacing: 0.08em;
    }
    .stButton>button, .stDownloadButton>button {
        background-color: #F8F7F9;
        color: #AC3C3D;
        font-family: 'helvetica-w01-roman', Helvetica, Arial, sans-serif !important;
        font-size: 22px;
        font-weight: 400;
        border-radius: 8px;
        border: 1px solid #AC3C3D;
        padding: 0.5rem 1.5rem;
        margin-top: 0.5rem;
        letter-spacing: 0.1em;
        transition: all 0.2s ease;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background: #AC3C3D;
        color: #F8F7F9;
        border: 1px solid #F8F7F9;
    }
    .stButton>button:disabled, .stDownloadButton>button:disabled {
        background: #E2E2E2 !important;
        color: #8F8F8F !important;
        border: none !important;
    }
    .stDataFrame {
        background-color: #fff;
        border-radius: 8px;
        border: 1px solid #E2E2E2;
        font-family: 'helvetica-w01-roman', Helvetica, Arial, sans-serif !important;
    }
    .st-expander {
        background: #fff !important;
        border-radius: 8px !important;
        border: 1px solid #E2E2E2 !important;
    }
    .st-expanderHeader {
        color: #244CC4 !important;
        font-family: 'helvetica-w01-roman', Helvetica, Arial, sans-serif !important;
        font-size: 20px !important;
        font-weight: 400 !important;
        letter-spacing: 0.08em;
    }
    .stTextInput>div>input, .stSelectbox>div>div>div>input, .stMultiSelect>div>div>div>input {
        font-family: 'helvetica-w01-roman', Helvetica, Arial, sans-serif !important;
        font-size: 16px !important;
        color: #222 !important;
    }
    </style>
""", unsafe_allow_html=True)

df = load_data()

# --- SIDEBAR ---
with st.sidebar:
    st.title("Filters")
    date_min = df['date'].min()
    date_max = df['date'].max()
    start_date, end_date = st.date_input("Date Range", [date_min, date_max])
    size_filter = st.radio("Participant Size Filter", ['all', 'has', 'no'], format_func=lambda x: {
        'all': 'All events', 'has': 'Has participant size', 'no': 'No participant size'
    }[x])
    trump_filter = st.checkbox("Only anti-Trump events")
    org_search = st.text_input("Organization Search (comma separated)")
    state_filter = st.multiselect("State/Territory", sorted(df['state'].dropna().unique()))
    any_outcomes_options = {
        'arrests_any': 'Any Arrested Protesters',
        'participant_injuries_any': 'Any Participant Injuries',
        'police_injuries_any': 'Any Police Injuries',
        'property_damage_any': 'Any Property Damage',
        'participant_deaths_any': 'Any Participant Deaths',
        'police_deaths_any': 'Any Police Deaths',
    }
    any_outcomes_filter = st.multiselect(
        "Event Outcomes",
        options=list(any_outcomes_options.keys()),
        format_func=lambda x: any_outcomes_options[x]
    )
    download_choice = st.selectbox("Download Data", ['filtered', 'full'], format_func=lambda x: {
        'filtered': 'Filtered View Only', 'full': 'Full Dataset'
    }[x])
    show_definitions = st.toggle("Show Data Definitions & Sources", value=False)

# --- MAIN LAYOUT ---
if show_definitions:
    st.markdown("""
    ### Data Definitions & Sources

    **Data Source:** Crowd Counting Consortium (CCC) Phase 3. [Original data and metadata available here.](https://github.com/crowdcountingconsortium/public)

    - **Location:** Based on city-level geocoding. If multiple events occurred in the same city on the same day, their locations are jittered for visualization. Exact event locations may not be available; city centroids or modified city coordinates are used.
    - **Anti-Trump events:** Events where the 'targets' field includes the substring 'trump' (case-insensitive).
    - **Participant Size:** The 'size_mean' field is an estimate of crowd size, as reported or inferred. Some events may have missing or uncertain size estimates.
    - **Organizations:** Organizations are listed as a semicolon-separated string. Organization search matches any substring in this field.
    - **State/Territory:** Includes U.S. states and territories as reported in the original data.
    - **Date:** Date of the event (YYYY-MM-DD).
    - **Cumulative Total Events:** Number of events after all filters are applied.
    - **Size Mean as % of US Population:** Calculated as the largest single-day sum of 'size_mean' in the filtered data, divided by the 2024 U.S. population estimate (340,100,000).
    - **Download:** You can download either the filtered view or the full dataset as CSV.
    - **More info:** See the CCC [Harvard Dataverse](https://dataverse.harvard.edu/dataverse/ccc) for full metadata and documentation.
    """)
else:
    # --- FILTER DATA ---
    dff = filter_data(
        df,
        pd.to_datetime(start_date),
        pd.to_datetime(end_date),
        size_filter,
        trump_filter,
        org_search,
        state_filter,
        any_outcomes_filter
    )

    # Ensure location_label exists for event details
    if 'location_label' not in dff.columns:
        def best_location(row):
            loc = str(row.get('location', '')).strip()
            if loc and loc.lower() != 'nan':
                return loc
            loc2 = str(row.get('locality', '')).strip()
            if loc2 and loc2.lower() != 'nan':
                return loc2
            state = row.get('state', 'Unknown')
            date = row['date'].date() if pd.notnull(row.get('date')) else ''
            return f"{state}, {date}"
        dff['location_label'] = dff.apply(best_location, axis=1)

    # Ensure all event detail columns exist in dff
    detail_columns = [
        'locality', 'location', 'organizations', 'size_mean', 'notables', 'targets',
        'claims_summary', 'participant_measures', 'police_measures',
        'participant_injuries', 'police_injuries', 'arrests', 'property_damage', 'notes'
    ]
    for col in detail_columns:
        if col not in dff.columns:
            dff[col] = np.nan

    # --- KPIs ---
    total_events = len(dff)
    total_participants = dff['size_mean'].sum() if 'size_mean' in dff.columns else 0
    peak_day = dff.groupby('date')['size_mean'].sum().max() if 'size_mean' in dff.columns else 0
    percent_us_pop = (peak_day / US_POPULATION) * 100 if peak_day else 0
    mean_size = dff['size_mean'].mean() if 'size_mean' in dff.columns else 0
    percent_no_size = 0
    if total_events > 0 and 'size_mean' in dff.columns:
        percent_no_size = 100 * dff['size_mean'].isna().sum() / total_events

    # --- KPIs with modern style ---
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"<div class='big-metric'>{total_events:,}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Total Events in Range</div>", unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"<div class='big-metric'>{percent_us_pop:.4f}%</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Size Mean as % of US Population</div>", unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"<div class='big-metric'>{mean_size:,.0f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Mean Protest Size</div>", unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"<div class='big-metric'>{percent_no_size:.1f}%</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Events Missing Size</div>", unsafe_allow_html=True)

    # --- MAP with bold colors and light theme ---
    selected_location = None
    if 'lat' in dff.columns and 'lon' in dff.columns and not dff['lat'].isnull().all() and not dff['lon'].isnull().all():
        dff_map = dff.dropna(subset=['lat', 'lon'])
        dff_map = jitter_coords(dff_map, lat_col='lat', lon_col='lon', jitter_amount=0.03)
        agg_map = aggregate_events_for_map(dff_map)
        has_size = agg_map[agg_map['size_mean'].notna()].copy()
        no_size = agg_map[agg_map['size_mean'].isna()].copy()
        fig_map = go.Figure()
        # Calculate sizeref for consistent marker sizing
        if not has_size.empty:
            max_size = has_size['size_mean'].max()
            sizeref = 2.0 * max_size / (50.0 ** 2) if max_size > 0 else 1
        else:
            sizeref = 1
        # Assign detailed hover to both groups
        if not has_size.empty:
            has_size['detailed_hover'] = has_size.apply(build_hover, axis=1)
            fig_map.add_trace(go.Scattermapbox(
                lat=has_size['lat'],
                lon=has_size['lon'],
                mode='markers',
                marker=dict(
                    size=has_size['size_mean'],
                    color='#fbbf24',
                    opacity=0.85,
                    sizemode='area',
                    sizeref=sizeref,
                    sizemin=6
                ),
                text=has_size['location_label'],
                hovertext=has_size['detailed_hover'],
                name="Has Size"
            ))
        if not no_size.empty:
            no_size['detailed_hover'] = no_size.apply(build_hover, axis=1)
            fig_map.add_trace(go.Scattermapbox(
                lat=no_size['lat'],
                lon=no_size['lon'],
                mode='markers',
                marker=dict(
                    size=14,
                    color='#ef4444',
                    opacity=0.85,
                    sizemode='area',
                    sizeref=sizeref,
                    sizemin=6
                ),
                text=no_size['location_label'],
                hovertext=no_size['detailed_hover'],
                name="Missing Size"
            ))
        fig_map.update_layout(
            mapbox_style="carto-positron",  # <-- Light map style
            mapbox_zoom=3,
            mapbox_center={"lat": 39.8283, "lon": -98.5795},
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            height=420,
            showlegend=False,
            paper_bgcolor="#fff",
            plot_bgcolor="#fff",
            font=dict(color="#222", family="Inter, Segoe UI, Arial, sans-serif")
        )
        st.plotly_chart(fig_map, use_container_width=True, key="mainmap", click_events=True)
    else:
        st.warning("No events with valid location data for the selected filters.")

    # --- MOMENTUM GRAPH ---
    dff_momentum = dff[['date', 'participants_numeric']].dropna()
    if not dff_momentum.empty:
        dff_momentum = dff_momentum.set_index('date').resample('D').agg(['sum', 'count'])
        dff_momentum.columns = ['sum', 'count']
        dff_momentum['momentum'] = dff_momentum['sum'] * dff_momentum['count']
        dff_momentum['alt_momentum'] = dff_momentum['sum'].rolling(7).sum()
        dff_momentum = dff_momentum.reset_index()
        fig_momentum = go.Figure()
        fig_momentum.add_trace(go.Scatter(
            x=dff_momentum['date'],
            y=dff_momentum['momentum'],
            mode='lines',
            name='Original Momentum'
        ))
        fig_momentum.add_trace(go.Scatter(
            x=dff_momentum['date'],
            y=dff_momentum['alt_momentum'],
            mode='lines',
            name='7-Day Rolling Total'
        ))
        # Trendline
        if len(dff_momentum) > 1:
            trend_df = dff_momentum.dropna(subset=['momentum'])
            if len(trend_df) > 1:
                x_ord = trend_df['date'].map(pd.Timestamp.toordinal)
                y = trend_df['momentum']
                coeffs = np.polyfit(x_ord, y, 1)
                trendline = np.polyval(coeffs, x_ord)
                fig_momentum.add_trace(go.Scatter(
                    x=trend_df['date'],
                    y=trendline,
                    mode='lines',
                    name='Trendline',
                    line=dict(dash='dash', color='black')
                ))
        fig_momentum.update_layout(
            template="plotly_white",
            title="Momentum of Dissent",
            height=270,
            margin=dict(t=30, b=10, l=10, r=10),
            paper_bgcolor="#fff",
            plot_bgcolor="#fff",
            font=dict(color="#222", family="Inter, Segoe UI, Arial, sans-serif")
        )
        st.plotly_chart(fig_momentum, use_container_width=True)

    # --- DAILY EVENT COUNT ---
    if 'date' in dff.columns and not dff['date'].isnull().all():
        dff_daily = dff.set_index('date').resample('D').size().reset_index(name='count')
        if not dff_daily.empty:
            fig_daily = px.bar(
                dff_daily,
                x='date',
                y='count',
                title="Daily Event Count",
                height=270,
                template="plotly_white"
            )
            fig_daily.update_layout(
                margin=dict(t=30, b=10, l=10, r=10),
                paper_bgcolor="#fff",
                plot_bgcolor="#fff",
                font=dict(color="#222", family="Inter, Segoe UI, Arial, sans-serif")
            )
            st.plotly_chart(fig_daily, use_container_width=True)
        dff_cum = dff.set_index('date').resample('D').size().reset_index(name='count')
        if not dff_cum.empty:
            dff_cum['cumulative'] = dff_cum['count'].cumsum()
            fig_cumulative = px.line(
                dff_cum,
                x='date',
                y='cumulative',
                title="Cumulative Total Events",
                height=250,
                template="plotly_white"
            )
            fig_cumulative.update_layout(
                margin=dict(t=30, b=10, l=10, r=10),
                paper_bgcolor="#fff",
                plot_bgcolor="#fff",
                font=dict(color="#222", family="Inter, Segoe UI, Arial, sans-serif")
            )
            st.plotly_chart(fig_cumulative, use_container_width=True)
        dff_participants = dff.set_index('date').resample('D')['size_mean'].sum().reset_index(name='participants')
        if not dff_participants.empty and not dff_participants['participants'].isnull().all():
            fig_daily_participants = px.bar(
                dff_participants,
                x='date',
                y='participants',
                title="Daily Participant Count",
                height=250,
                template="plotly_white"
            )
            fig_daily_participants.update_layout(
                margin=dict(t=30, b=10, l=10, r=10),
                paper_bgcolor="#fff",
                plot_bgcolor="#fff",
                font=dict(color="#222", family="Inter, Segoe UI, Arial, sans-serif")
            )
            st.plotly_chart(fig_daily_participants, use_container_width=True)


    # --- DOWNLOAD BUTTON ---
    if download_choice == 'full':
        export_df = df.copy()
    else:
        export_df = dff.copy()
    st.download_button(
        "Download CSV",
        export_df.to_csv(index=False).encode('utf-8'),
        "protest_data.csv",
        "text/csv"
    )

    # --- Data Table ---
    with st.expander("Show Filtered Data Table"):
        st.dataframe(dff, use_container_width=True, height=400)