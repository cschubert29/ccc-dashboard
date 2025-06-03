import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update, dash_table
from flask_caching import Cache
import os
import random
import re
import time
import io

file_path = "ccc-phase3-public.csv"
US_POPULATION = 340_100_000

# Load data
df = pd.read_csv(file_path, encoding='latin1', low_memory=False)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['size_mean'] = pd.to_numeric(df['size_mean'], errors='coerce')
df['participants_numeric'] = df['size_mean']
df['targets'] = df['targets'].astype(str).str.lower()
df['organizations'] = df['organizations'].astype(str).str.lower()
df['state'] = df['state'].astype('category')
df['targets'] = df['targets'].astype('category')
df['organizations'] = df['organizations'].astype('category')

# Ensure numeric columns are actually numeric for filtering
for col in [
    'participant_injuries', 'police_injuries', 'arrests',
    'participant_deaths', 'police_deaths'
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.title = "Protest Dashboard"

# Configure caching
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})

if not os.path.exists('cache-directory'):
    os.makedirs('cache-directory')

# --- DESIGN CONSTANTS ---
FONT_FAMILY = "helvetica,Arial,sans-serif" 
PRIMARY_BLUE = "#244CC4"
PRIMARY_RED = "#AC3C3D"
BG_LIGHT = "#F8F7F9"
KPI_BG_BLUE = "#244CC4"
KPI_BG_RED = "#AC3C3D"
KPI_TEXT = "#fff"

# Sidebar content as separate components
filter_panel = html.Div([
    html.H2("Filters", style={'marginBottom': '20px', 'fontFamily': FONT_FAMILY, 'color': PRIMARY_BLUE}),
    html.Label("Date Range", style={'fontFamily': FONT_FAMILY}),
    dcc.DatePickerRange(
        id='date-range',
        start_date=df['date'].min(),
        end_date=df['date'].max(),
        display_format='YYYY-MM-DD',
        style={'marginBottom': '20px', 'width': '100%'}
    ),
    html.Label("Participant Size Filter", style={'fontFamily': FONT_FAMILY}),
    dcc.RadioItems(
        id='size-filter',
        options=[
            {'label': 'Has participant size', 'value': 'has'},
            {'label': 'No participant size', 'value': 'no'},
            {'label': 'All events', 'value': 'all'}
        ],
        value='all',
        labelStyle={'display': 'block', 'marginBottom': '6px', 'fontFamily': FONT_FAMILY},
        style={'marginBottom': '20px'}
    ),
    dcc.Checklist(
        id='trump-filter',
        options=[{'label': 'Only anti-Trump events', 'value': 'trump'}],
        value=[],
        style={'marginBottom': '20px', 'fontFamily': FONT_FAMILY}
    ),
    html.Label("Organization Search", style={'fontFamily': FONT_FAMILY}),
    dcc.Input(
        id='org-search',
        type='text',
        placeholder="Type organizations, separated by commas",
        style={'width': '100%', 'marginBottom': '5px', 'borderRadius': '8px', 'border': '1px solid #ccc', 'padding': '8px', 'fontFamily': FONT_FAMILY}
    ),
    html.Div("↩ Separate multiple organizations with commas", style={'fontSize': '0.8em', 'color': '#666', 'marginBottom': '15px'}),
    html.Label("State/Territory", style={'fontFamily': FONT_FAMILY}),
    dcc.Dropdown(
        id='state-filter',
        options=[
            {'label': s, 'value': s}
            for s in sorted(df['state'].dropna().unique())
        ],
        value=[],
        multi=True,
        placeholder="Select state(s) or territory(ies)",
        clearable=True,
        style={'marginBottom': '20px', 'borderRadius': '8px', 'fontFamily': FONT_FAMILY}
    ),
    html.Label("Event Outcomes", style={'fontFamily': FONT_FAMILY}),
    dcc.Checklist(
        id='any-outcomes-filter',
        options=[
            {'label': 'Any Arrested Protesters', 'value': 'arrests_any'},
            {'label': 'Any Participant Injuries', 'value': 'participant_injuries_any'},
            {'label': 'Any Police Injuries', 'value': 'police_injuries_any'},
            {'label': 'Any Property Damage', 'value': 'property_damage_any'},
            {'label': 'Any Participant Deaths', 'value': 'participant_deaths_any'},
            {'label': 'Any Police Deaths', 'value': 'police_deaths_any'},
        ],
        value=[],
        style={'marginBottom': '20px', 'fontFamily': FONT_FAMILY}
    ),
    html.Label("Download Data", style={'fontFamily': FONT_FAMILY}),
    dcc.Dropdown(
        id='download-choice',
        options=[
            {'label': 'Filtered View Only', 'value': 'filtered'},
            {'label': 'Full Dataset', 'value': 'full'}
        ],
        value='filtered',
        clearable=False,
        style={'marginBottom': '10px', 'borderRadius': '8px', 'fontFamily': FONT_FAMILY}
    ),
    html.Button(
        "Download CSV",
        id="download-btn",
        style={
            'marginBottom': '20px',
            'width': '100%',
            'backgroundColor': PRIMARY_BLUE,
            'color': '#fff',
            'fontWeight': 'bold',
            'fontFamily': FONT_FAMILY,
            'borderRadius': '12px',
            'border': 'none',
            'fontSize': '1.1em',
            'padding': '12px 0',
            'boxShadow': '0 2px 8px rgba(36,76,196,0.08)',
            'transition': 'background 0.2s'
        }
    ),
    dcc.Download(id="download-data"),
], id='filter-panel', style={
    'padding': '24px',
    'backgroundColor': BG_LIGHT,
    'borderRadius': '16px',
    'boxShadow': '0 2px 8px rgba(0,0,0,0.07)',
    'marginBottom': '24px',
    'fontFamily': FONT_FAMILY
})

definitions_panel = html.Div([
    html.H3("Data Definitions & Sources", style={'marginTop': '20px', 'fontFamily': FONT_FAMILY, 'color': PRIMARY_BLUE}),
    html.Div([
        html.P([
            html.B("Data Source: "),
            "Crowd Counting Consortium (CCC) Phase 3. ",
            html.A("Original data and metadata available here.",
                   href="https://github.com/crowdcountingconsortium/public", target="_blank")
        ]),
        html.Ul([
            html.Li([
                html.B("Location: "),
                "Based on city-level geocoding. If multiple events occurred in the same city on the same day, their locations are jittered for visualization. "
                "Exact event locations may not be available; city centroids or modified city coordinates are used."
            ]),
            html.Li([
                html.B("Anti-Trump events: "),
                "Events where the 'targets' field includes the substring 'trump' (case-insensitive)."
            ]),
            html.Li([
                html.B("Participant Size: "),
                "The 'size_mean' field is an estimate of crowd size, as reported or inferred. Some events may have missing or uncertain size estimates."
            ]),
            html.Li([
                html.B("Organizations: "),
                "Organizations are listed as a semicolon-separated string. Organization search matches any substring in this field."
            ]),
            html.Li([
                html.B("State/Territory: "),
                "Includes U.S. states and territories as reported in the original data."
            ]),
            html.Li([
                html.B("Date: "),
                "Date of the event (YYYY-MM-DD)."
            ]),
            html.Li([
                html.B("Cumulative Total Events: "),
                "Number of events after all filters are applied."
            ]),
            html.Li([
                html.B("Size Mean as % of US Population: "),
                "Calculated as the largest single-day sum of 'size_mean' in the filtered data, divided by the 2024 U.S. population estimate (340,100,000)."
            ]),
            html.Li([
                html.B("Download: "),
                "You can download either the filtered view or the full dataset as CSV."
            ]),
            html.Li([
                html.B("More info: "),
                "See the CCC ",
                html.A("Harvard Dataverse", href="https://dataverse.harvard.edu/dataverse/ccc", target="_blank"),
                " for full metadata and documentation."
            ])
        ])
    ], style={'fontSize': '0.95em', 'color': '#333', 'marginTop': '10px'})
], id='definitions-panel', style={
    'padding': '24px',
    'backgroundColor': BG_LIGHT,
    'borderRadius': '12px',
    'boxShadow': '0 2px 8px rgba(0,0,0,0.05)',
    'marginBottom': '24px',
    'fontFamily': FONT_FAMILY
})

# Sidebar with toggle button and content container
sidebar = html.Div([
    html.Button(
        id='toggle-sidebar',
        n_clicks=0,
        children="Show Data Definitions & Sources",
        style={'marginBottom': '18px', 'width': '100%', 'fontWeight': 'bold', 'fontFamily': FONT_FAMILY, 'backgroundColor': PRIMARY_BLUE, 'color': '#fff'}
    ),
    html.Div(id='sidebar-content')
], style={
    'width': '320px',
    'padding': '32px 16px 48px 16px',
    'boxSizing': 'border-box',
    'backgroundColor': BG_LIGHT,
    'borderRight': '1px solid #ccc',
    'flexShrink': '0',
    'flexGrow': '1',
    'overflowY': 'auto',
    'boxShadow': '2px 0 8px rgba(0,0,0,0.07)',
    'borderRadius': '0 12px 12px 0',
    'position': 'relative',
    'zIndex': 1050,
    'fontFamily': FONT_FAMILY
})

dashboard_layout = html.Div([
    sidebar,
    html.Div([
        html.Div(id='no-data-message', style={'marginBottom': '24px'}),
        html.Div(id='kpi-cards', style={
            'display': 'flex',
            'justifyContent': 'space-between',
            'margin': '24px 0 24px 0',
            'gap': '18px'
        }),
        html.Div([
            dcc.Graph(id='map-graph', style={
                'height': '340px',
                'marginBottom': '8px',
                'borderRadius': '16px',
                'backgroundColor': '#fff',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.07)'
            }),
            # Only show attribution ONCE, not inside the event details panel
            html.Div(
                "Map tiles by Carto, data © OpenStreetMap contributors",
                style={
                    'fontSize': '0.95em',
                    'color': '#888',
                    'textAlign': 'right',
                    'margin': '0 6px 10px 0',
                    'fontFamily': FONT_FAMILY
                }
            ),
            html.Div(id='event-details-panel',
                style={
                    'margin': '0 0 28px 0',
                    'padding': '18px',
                    'backgroundColor': '#fff',
                    'borderRadius': '12px',
                    'boxShadow': '0 2px 8px rgba(0,0,0,0.05)',
                    'minHeight': '56px',
                    'display': 'block',
                    'fontFamily': FONT_FAMILY
                }
            ),
            dcc.Graph(id='momentum-graph', style={
                'height': '260px',
                'marginBottom': '28px',
                'borderRadius': '16px',
                'backgroundColor': '#fff',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.07)'
            }),
            dcc.Graph(id='daily-graph', style={
                'height': '260px',
                'marginBottom': '28px',
                'borderRadius': '16px',
                'backgroundColor': '#fff',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.07)'
            }),
            dcc.Graph(id='cumulative-graph', style={
                'height': '260px',
                'marginBottom': '28px',
                'borderRadius': '16px',
                'backgroundColor': '#fff',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.07)'
            }),
            dcc.Graph(id='daily-participant-graph', style={
                'height': '260px',
                'marginBottom': '28px',
                'borderRadius': '16px',
                'backgroundColor': '#fff',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.07)'
            })
        ], style={'minWidth': '0'})
    ], style={
        'width': 'calc(100% - 320px)',
        'padding': '32px',
        'boxSizing': 'border-box',
        'flexGrow': '1',
        'overflow': 'auto',
        'backgroundColor': '#f5f6fa',
        'borderRadius': '12px 0 0 12px',
        'fontFamily': FONT_FAMILY
    })
], style={
    'display': 'flex',
    'flexDirection': 'row',
    'flexWrap': 'nowrap',
    'height': '100vh',
    'overflow': 'hidden',
    'fontFamily': FONT_FAMILY,
    'backgroundColor': '#e9ecef'
})

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='filtered-data'),
    html.Div(id='page-content')
], style={'height': '100vh', 'margin': 0, 'padding': 0, 'fontFamily': FONT_FAMILY})

def jitter_coords(df, lat_col='lat', lon_col='lon', jitter_amount=0.05):
    # Add random jitter to duplicate lat/lon pairs in a DataFrame
    df = df.copy().reset_index(drop=True)
    coords = df[[lat_col, lon_col]].astype(str).agg('_'.join, axis=1)
    counts = coords.value_counts()
    dup_coords = counts[counts > 1].index
    for coord in dup_coords:
        idxs = df.index[coords == coord].tolist()
        for offset, i in enumerate(idxs):
            if offset == 0:
                continue  # Leave the first as is so they don't all move
            angle = random.uniform(0, 2 * np.pi)
            radius = random.uniform(0.0005, jitter_amount)
            df.at[i, lat_col] += np.cos(angle) * radius
            df.at[i, lon_col] += np.sin(angle) * radius
    return df

def safe_numeric(series):
    return pd.to_numeric(series, errors='coerce')

# --- SPEED OPTIMIZATION SECTION ---

# 1. Only copy DataFrame when necessary (avoid df.copy() in filter_data)
# 2. Remove unnecessary print statements
# 3. Limit columns stored in dcc.Store (filtered-data)
# 4. Use .loc for filtering to avoid chained assignment warnings
# 5. Avoid unnecessary .apply in filter_data

@cache.memoize(timeout=120)
def filter_data(
    start_date, end_date, size_filter, trump_filter, org_search, state_filter,
    any_outcomes_filter
):
    # Use view, not copy, unless modifying
    dff = df

    mask = pd.Series(True, index=dff.index)

    # Apply filters (all vectorized)
    if start_date and end_date:
        mask &= (dff['date'] >= start_date) & (dff['date'] <= end_date)
    if size_filter == 'has':
        mask &= dff['size_mean'].notna()
    elif size_filter == 'no':
        mask &= dff['size_mean'].isna()
    if 'trump' in trump_filter:
        mask &= dff['targets'].str.contains("trump", na=False)
    if org_search:
        org_terms = [o.strip().lower() for o in org_search.split(',') if o.strip()]
        if org_terms:
            pattern = '|'.join([re.escape(org) for org in org_terms])
            mask &= dff['organizations'].str.contains(pattern, na=False, regex=True)
    if state_filter:
        mask &= dff['state'].isin(state_filter)

    # Outcomes logic (all vectorized)
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

    # Filter only once, then add columns as needed
    dff = dff.loc[mask]

    # Only add columns if missing (should not happen in production)
    required_cols = ['lat', 'lon', 'date', 'size_mean', 'participants_numeric', 'title', 'organizations']
    for col in required_cols:
        if col not in dff.columns:
            dff[col] = np.nan

    # Add location_label (vectorized, not apply)
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
    # Use .apply only for this, as it's not easily vectorizable
    dff = dff.copy()  # Only copy here if needed for new columns
    dff['location_label'] = dff.apply(best_location, axis=1)

    # Remove fallback logic (not needed for speed, but simplifies)
    return dff

@cache.memoize(timeout=120)
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
    agg['hover'] = agg.apply(
        lambda row: (
            f"<b>{row['location_label']}</b><br>"
            f"Events at this site: {row['count']}<br><br>"
            f"<b>Events:</b><br>{row['event_list']}"
        ), axis=1
    )
    agg['text'] = agg['location_label']
    return agg

@app.callback(
    [Output('map-graph', 'figure'),
     Output('momentum-graph', 'figure'),
     Output('daily-graph', 'figure'),
     Output('kpi-cards', 'children'),
     Output('filtered-data', 'data'),
     Output('cumulative-graph', 'figure'),
     Output('daily-participant-graph', 'figure'),
     Output('no-data-message', 'children')],
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('size-filter', 'value'),
     Input('trump-filter', 'value'),
     Input('org-search', 'value'),
     Input('state-filter', 'value'),
     Input('any-outcomes-filter', 'value')]
)
def update_all(
    start_date, end_date, size_filter, trump_filter, org_search, state_filter,
    any_outcomes_filter
):
    t0 = time.time()
    dff = filter_data(
        start_date, end_date, size_filter, trump_filter, org_search, state_filter,
        any_outcomes_filter
    )
    t1 = time.time()
    dff_map = dff.dropna(subset=['lat', 'lon'])
    dff_map = jitter_coords(dff_map, lat_col='lat', lon_col='lon', jitter_amount=0.03)
    agg_map = aggregate_events_for_map(dff_map)
    t2 = time.time()

    total_events = len(dff)
    total_participants = dff['size_mean'].sum() if 'size_mean' in dff.columns else 0
    peak_day = dff.groupby('date')['size_mean'].sum().max() if 'size_mean' in dff.columns else 0
    percent_us_pop = (peak_day / US_POPULATION) * 100 if peak_day else 0
    mean_size = dff['size_mean'].mean() if 'size_mean' in dff.columns else 0
    percent_no_size = 0
    if total_events > 0 and 'size_mean' in dff.columns:
        percent_no_size = 100 * dff['size_mean'].isna().sum() / total_events

    # Alternate KPI box colors: red, blue, red, blue, all white text
    kpis = [
        html.Div([
            html.Div(f"{total_events:,}", style={
                'fontSize': '2.2rem',
                'fontWeight': '700',
                'color': KPI_TEXT,
                'letterSpacing': '0.05em',
                'fontFamily': FONT_FAMILY
            }),
            html.Div("Total Events in Range", style={
                'fontSize': '1.1rem',
                'color': KPI_TEXT,
                'letterSpacing': '0.08em',
                'fontFamily': FONT_FAMILY
            })
        ], style={
            'width': '24%',
            'textAlign': 'center',
            'padding': '18px 0 10px 0',
            'borderRadius': '16px',
            'backgroundColor': KPI_BG_RED,
            'boxShadow': '0 2px 8px rgba(36,76,196,0.08)'
        }),
        html.Div([
            html.Div(f"{percent_us_pop:.4f}%", style={
                'fontSize': '2.2rem',
                'fontWeight': '700',
                'color': KPI_TEXT,
                'letterSpacing': '0.05em',
                'fontFamily': FONT_FAMILY
            }),
            html.Div("Size Mean as % of US Population", style={
                'fontSize': '1.1rem',
                'color': KPI_TEXT,
                'letterSpacing': '0.08em',
                'fontFamily': FONT_FAMILY
            })
        ], style={
            'width': '24%',
            'textAlign': 'center',
            'padding': '18px 0 10px 0',
            'borderRadius': '16px',
            'backgroundColor': KPI_BG_BLUE,
            'boxShadow': '0 2px 8px rgba(36,76,196,0.08)'
        }),
        html.Div([
            html.Div(f"{mean_size:,.0f}", style={
                'fontSize': '2.2rem',
                'fontWeight': '700',
                'color': KPI_TEXT,
                'letterSpacing': '0.05em',
                'fontFamily': FONT_FAMILY
            }),
            html.Div("Mean Protest Size", style={
                'fontSize': '1.1rem',
                'color': KPI_TEXT,
                'letterSpacing': '0.08em',
                'fontFamily': FONT_FAMILY
            })
        ], style={
            'width': '24%',
            'textAlign': 'center',
            'padding': '18px 0 10px 0',
            'borderRadius': '16px',
            'backgroundColor': KPI_BG_RED,
            'boxShadow': '0 2px 8px rgba(36,76,196,0.08)'
        }),
        html.Div([
            html.Div(f"{percent_no_size:.1f}%", style={
                'fontSize': '2.2rem',
                'fontWeight': '700',
                'color': KPI_TEXT,
                'letterSpacing': '0.05em',
                'fontFamily': FONT_FAMILY
            }),
            html.Div("Events Missing Size", style={
                'fontSize': '1.1rem',
                'color': KPI_TEXT,
                'letterSpacing': '0.08em',
                'fontFamily': FONT_FAMILY
            })
        ], style={
            'width': '24%',
            'textAlign': 'center',
            'padding': '18px 0 10px 0',
            'borderRadius': '16px',
            'backgroundColor': KPI_BG_BLUE,
            'boxShadow': '0 2px 8px rgba(36,76,196,0.08)'
        }),
    ]

    # Defensive: Ensure 'lat' and 'lon' columns exist and are not all missing
    if 'lat' not in dff.columns or 'lon' not in dff.columns or dff['lat'].isnull().all() or dff['lon'].isnull().all():
        empty_fig = go.Figure()
        return (
            empty_fig, empty_fig, empty_fig, kpis,
            dff.to_json(date_format='iso', orient='split'),
            empty_fig, empty_fig,
            html.Div(
                "No events with valid location data for the selected filters.",
                style={'color': 'red', 'fontWeight': 'bold', 'fontSize': '1.2em', 'margin': '20px 0'}
            )
        )

    # Separate locations with and without participant size
    has_size = agg_map[agg_map['size_mean'].notna()]
    no_size = agg_map[agg_map['size_mean'].isna()]
    fig_map = go.Figure()

    # Plot locations with participant size (blue)
    if not has_size.empty:
        max_size = has_size['size_mean'].max()
        sizeref = 2.0 * max_size / (50.0 ** 2) if max_size > 0 else 1
        fig_map.add_trace(go.Scattermapbox(
            lat=has_size['lat'],
            lon=has_size['lon'],
            mode='markers',
            marker=dict(
                size=has_size['size_mean'],
                color='blue',
                sizemode='area',
                sizeref=sizeref,
                sizemin=5
            ),
            text=has_size['text'],
            hovertext=has_size['hover'],
            name="Has Size"
        ))

    # Plot locations without participant size (red, fixed size)
    if not no_size.empty:
        fig_map.add_trace(go.Scattermapbox(
            lat=no_size['lat'],
            lon=no_size['lon'],
            mode='markers',
            marker=dict(
                size=12,
                color='red',
                sizemode='area',
                sizeref=1,
                sizemin=5
            ),
            text=no_size['text'],
            hovertext=no_size['hover'],
            name="Missing Size"
        ))

    # Restore old fixed map zoom/center for consistent US view
    fig_map.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=3,
        mapbox_center={"lat": 39.8283, "lon": -98.5795},
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=320,
        showlegend=True
    )

    dff_momentum = dff[['date', 'participants_numeric']].dropna()
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

    # Add linear trendline to the momentum plot
    if len(dff_momentum) > 1:
        # Remove Na for fitting
        trend_df = dff_momentum.dropna(subset=['momentum'])
        if len(trend_df) > 1:
            # Convert dates to ordinal for fitting
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
        title="Momentum of Dissent",
        height=270,
        margin=dict(t=30, b=10, l=10, r=10)
    )

    # Defensive: Ensure 'date' is datetime and not all-NaT before resampling
    if (
        'date' not in dff.columns or
        dff['date'].isnull().all() or
        not np.issubdtype(dff['date'].dtype, np.datetime64)
    ):
        fig_daily = go.Figure()
        fig_cumulative = go.Figure()
        fig_daily_participants = go.Figure()
    else:
        # Daily event count
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
                fig_daily.update_layout(margin=dict(t=30, b=10, l=10, r=10))
            else:
                fig_daily = go.Figure()
        else:
            fig_daily = go.Figure()

        # Cumulative total events by date
        dff_cum = dff.set_index('date').resample('D').size().reset_index(name='count')
        if dff_cum.empty:
            fig_cumulative = go.Figure()
        else:
            dff_cum['cumulative'] = dff_cum['count'].cumsum()
            fig_cumulative = px.line(
                dff_cum,
                x='date',
                y='cumulative',
                title="Cumulative Total Events",
                height=250,
                template="plotly_white"
            )
            fig_cumulative.update_layout(margin=dict(t=30, b=10, l=10, r=10))

        # Daily participant count (sum of size_mean per day)
        dff_participants = dff.set_index('date').resample('D')['size_mean'].sum().reset_index(name='participants')
        if dff_participants.empty or dff_participants['participants'].isnull().all():
            fig_daily_participants = go.Figure()
        else:
            fig_daily_participants = px.bar(
                dff_participants,
                x='date',
                y='participants',
                title="Daily Participant Count",
                height=250,
                template="plotly_white"
            )
            fig_daily_participants.update_layout(margin=dict(t=30, b=10, l=10, r=10))
    
    # Only store necessary columns in dcc.Store
    store_cols = [
        'lat', 'lon', 'date', 'size_mean', 'participants_numeric', 'title',
        'organizations', 'location_label', 'locality', 'location', 'notables',
        'targets', 'claims_summary', 'participant_measures', 'police_measures',
        'participant_injuries', 'police_injuries', 'arrests', 'property_damage', 'notes'
    ]
    dff_store = dff[store_cols].copy() if all(col in dff.columns for col in store_cols) else dff.copy()
    return fig_map, fig_momentum, fig_daily, kpis, dff_store.to_json(date_format='iso', orient='split'), fig_cumulative, fig_daily_participants, None

@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    return dashboard_layout

@app.callback(
    Output('event-details-panel', 'children'),
    [Input('map-graph', 'clickData'),
     Input('filtered-data', 'data')]
)
def show_event_details(clickData, filtered_json):
    # Always show the sticky message if nothing is selected
    if not clickData or not filtered_json:
        return html.Div(
            "Click a map marker to see event details.",
            style={
                'color': '#555',
                'fontSize': '1.1em',
                'fontStyle': 'italic',
                'textAlign': 'center',
                'padding': '16px 0'
            }
        )

    dff = pd.read_json(io.StringIO(filtered_json), orient='split')
    point = clickData['points'][0]
    location_label = point.get('text') or point.get('hovertext')
    if not location_label:
        return html.Div("No details available for this location.", style={'color': '#555', 'margin': '12px 0'})

    # Prevent KeyError
    if 'location_label' not in dff.columns:
        return html.Div("No event details found for this marker.", style={'color': '#555', 'margin': '12px 0'})

    events = dff[dff['location_label'] == location_label]

    if events.empty:
        return html.Div("No event details found for this marker.", style={'color': '#555', 'margin': '12px 0'})

    # Only show non-null/non-empty values in event details
    detail_fields = [
        ('Title', 'title'),
        ('Locality', 'locality'),
        ('Date', 'date'),
        ('Location', 'location'),
        ('Organizations', 'organizations'),
        ('Participants', 'size_mean'),
        ('Notables', 'notables'),
        ('Targets', 'targets'),
        ('Claims Summary', 'claims_summary'),
        ('Participant Measures', 'participant_measures'),
        ('Police Measures', 'police_measures'),
        ('Participant Injuries', 'participant_injuries'),
        ('Police Injuries', 'police_injuries'),
        ('Arrests', 'arrests'),
        ('Property Damage', 'property_damage'),
        ('Notes', 'notes')
    ]

    details = []
    for idx, (_, row) in enumerate(events.iterrows()):
        event_detail = []
        for label, col in detail_fields:
            val = row.get(col, None)
            # Only show if not null/empty/NaN
            if pd.isnull(val) or val is None or (isinstance(val, float) and np.isnan(val)) or (isinstance(val, str) and not val.strip()):
                continue
            # Format date
            if col == 'date':
                try:
                    val = pd.to_datetime(val).strftime('%Y-%m-%d')
                except Exception:
                    pass
            event_detail.append(html.P(f"{label}: {val}", style={'margin': '0 0 4px 0'}))
        # Compose summary as "Title - Date"
        title_val = row.get('title', 'No Title')
        date_val = row.get('date', None)
        if pd.isnull(date_val) or date_val is None or (isinstance(date_val, float) and np.isnan(date_val)):
            date_str = "NA"
        else:
            try:
                date_str = pd.to_datetime(date_val).strftime('%Y-%m-%d')
            except Exception:
                date_str = str(date_val)
        summary_text = f"{title_val} - {date_str}"
        details.append(
            html.Details([
                html.Summary(
                    summary_text,
                    style={'fontWeight': 'bold', 'fontSize': '1.1em'}
                ),
                html.Div(event_detail, style={'marginLeft': '12px'})
            ], open=False, style={'marginBottom': '16px'})
        )
    return details



@app.callback(
    Output("download-data", "data"),
    Input("download-btn", "n_clicks"),
    State("download-choice", "value"),
    State('date-range', 'start_date'),
    State('date-range', 'end_date'),
    State('size-filter', 'value'),
    State('trump-filter', 'value'),
    State('org-search', 'value'),
    State('state-filter', 'value'),
    State('any-outcomes-filter', 'value'),
    prevent_initial_call=True
)
def download_csv(n_clicks, choice, start_date, end_date, size_filter, trump_filter, org_search, state_filter, any_outcomes_filter):

    if not n_clicks:
        return no_update

    if choice == 'full':
        export_df = df.copy()
    else:
        export_df = filter_data(
            start_date, end_date, size_filter, trump_filter, org_search, state_filter, any_outcomes_filter
        )

    if 'fallback' in export_df.columns:
        export_df = export_df.drop(columns=['fallback'])

    return dcc.send_data_frame(export_df.to_csv, "protest_data.csv", index=False)

@app.callback(
    Output('sidebar-content', 'children'),
    Input('toggle-sidebar', 'n_clicks'),
    prevent_initial_call=False
)
def toggle_sidebar_content(n_clicks):
    # Show definitions on odd clicks, filters on even (or 0)
    if n_clicks and n_clicks % 2 == 1:
        return definitions_panel
    return filter_panel

# if __name__ == '__main__':
#     app.run(debug=True)