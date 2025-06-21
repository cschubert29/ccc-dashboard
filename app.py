import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, no_update, dash_table
from flask_caching import Cache
import os
import random
import re
import io

# Data file path
file_path = "ccc_anti_trump.csv"  
US_POPULATION = 340_100_000

# Check if a preprocessed file exists
processed_file = "processed_data.parquet"
if os.path.exists(processed_file):
    df = pd.read_parquet(processed_file)
else:
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
    if 'trump_stance' in df.columns:
        df['trump_stance'] = df['trump_stance'].astype(str).str.lower()

    # Ensure numeric columns are actually numeric for filtering
    for col in [
        'participant_injuries', 'police_injuries', 'arrests',
        'participant_deaths', 'police_deaths'
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.to_parquet(processed_file)  # Save the processed DataFrame

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

# Design constants
FONT_FAMILY = "helvetica,Arial,sans-serif" 
PRIMARY_BLUE = "#244CC4"
PRIMARY_RED = "#AC3C3D"
PRIMARY_WHITE = "#FFFFFF"

# Standardized margin for all graphs
standard_margin = dict(t=30, b=20, l=18, r=18)

# Sidebar content as separate components
filter_panel = html.Div([
    html.H2("Filters", style={'marginBottom': '20px', 'fontFamily': FONT_FAMILY, 'color': PRIMARY_BLUE}),
    html.Div([
        html.Label(
            "Date Range",
            title="The Crowd Counting Consortium (CCC) only releases new data once a month, during the first week. Recent events may not appear until after the next monthly update.",
            style={'fontFamily': FONT_FAMILY, 'marginRight': '6px', 'cursor': 'help'}
        )
    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '2px'}),
    dcc.DatePickerRange(
        id='date-range',
        start_date=df['date'].min(),
        end_date=df['date'].max(),
        display_format='YYYY-MM-DD',
        style={'marginBottom': '4px', 'width': '100%'}
    ),
    html.Div(
        "Click the date values to select dates to filter on. Note data is on a monthly release schedule, so recent events may not appear until the next update.",
        style={
            'fontSize': '0.8em',
            'color': PRIMARY_BLUE,
            'marginBottom': '16px',
            'marginTop': '-2px',
            'fontFamily': FONT_FAMILY
        }
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
    html.Label("Organization Search", style={'fontFamily': FONT_FAMILY}),
    dcc.Input(
        id='org-search',
        type='text',
        placeholder="Type organizations, separated by commas",
        style={'width': '100%', 'marginBottom': '5px', 'borderRadius': '8px', 'border': f'1px solid {PRIMARY_BLUE}', 'padding': '8px', 'fontFamily': FONT_FAMILY}
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
        "Download Dataset",
        id="download-btn",
        style={
            'marginBottom': '20px',
            'width': '100%',
            'backgroundColor': PRIMARY_BLUE,
            'color': PRIMARY_WHITE,
            'fontWeight': 'bold',
            'fontFamily': FONT_FAMILY,
            'borderRadius': '12px',
            'border': 'none',
            'fontSize': '1.1em',
            'padding': '12px 0',
            'boxShadow': '0 2px 8px rgba(36,76,196,0.08)',
            'transition': 'all 0.2s ease-in-out',
            'cursor': 'pointer'
        },
        className="hover-button"
    ),
    dcc.Download(id="download-data"),
    html.Div([
        html.A(
            "Is your data/event missing? Click here to learn how to fix it!",
            href="https://bit.ly/m/WeCount",
            target="_blank",
            style={
                'display': 'inline-block',
                'padding': '12px 24px',
                'backgroundColor': PRIMARY_BLUE,
                'color': PRIMARY_WHITE,
                'fontWeight': 'bold',
                'textDecoration': 'none',
                'borderRadius': '8px',
                'boxShadow': '0 2px 8px rgba(36,76,196,0.08)',
                'marginTop': '20px',
                'textAlign': 'center'
            }
        )
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),
], id='filter-panel', style={
    'padding': '24px',
    'backgroundColor': PRIMARY_WHITE,
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
            "Crowd Counting Consortium (CCC) Phase 3, subset of anti-Trump events only. ",
            html.A("Original data and metadata available here.",
                   href="https://github.com/crowdcountingconsortium/public", target="_blank")
        ]),
        html.P([
            "The data is coded based on the claims included in the dataset. If Trump isn't mentioned in the claims, the protest won't be included. ",
            "If something seems off or you want to log your protest data, visit ",
            html.A("this link", href="https://bit.ly/m/WeCount", target="_blank"),
            "."
        ], style={'marginTop': '10px'}),
        html.Ul([
            html.Li([
                html.B("Location: "),
                "Based on city-level geocoding. If multiple events occurred in the same city on the same day, their locations are jittered for visualization. "
                "Exact event locations may not be available; city centroids or modified city coordinates are used."
            ]),
            html.Li([
                html.B("Anti-Trump events: "),
                "This dashboard uses a dataset filtered to include only anti-Trump events. Events that may be against him but do not mention him explicitly may not be included here."
            ]),
            html.Li([
                html.B("Participant Size: "),
                "The 'size_mean' field is an average of the upper and lower range estimates of crowd size, as reported. "
                "This provides a standardized estimate of participant size for each event. Some events may have missing or uncertain size estimates."
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
                html.B("Largest Daily Participant Count as % of US population: "),
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
    'backgroundColor': PRIMARY_WHITE,
    'borderRadius': '12px',
    'boxShadow': '0 2px 8px rgba(0,0,0,0.05)',
    'marginBottom': '24px',
    'fontFamily': FONT_FAMILY
})

# Sidebar with content container and toggle button for definitions
sidebar = html.Div([
    # Content container for the filter panel or definitions panel
    html.Div(id='sidebar-content', children=[
        filter_panel
    ], style={
        'display': 'block',
        'width': '100%',
        'transition': 'all 0.3s ease-in-out'
    }),
    # Bottom button to toggle between filters and definitions
    html.Button(
        id='toggle-definitions',
        n_clicks=0,
        children="Show Data Definitions & Sources",
        style={
            'marginTop': 'auto',
            'width': '100%',
            'fontWeight': '600',
            'fontFamily': FONT_FAMILY,
            'backgroundColor': '#fff',
            'color': PRIMARY_BLUE,
            'border': f'2px solid {PRIMARY_BLUE}',
            'borderRadius': '24px',
            'fontSize': '1.08em',
            'padding': '12px 0',
            'boxShadow': '0 2px 8px rgba(36,76,196,0.06)',
            'cursor': 'pointer',
            'transition': 'all 0.2s ease-in-out',
            'position': 'relative',
            'zIndex': 1100
        },
        className="hover-button"
    )
], id='sidebar', style={
    'width': '320px',
    'padding': '18px 16px 48px 16px',
    'boxSizing': 'border-box',
    'background': 'linear-gradient(135deg, #f5f6fa, #e9ecef)',
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
    html.Div(id='main-content', children=[
        html.Div(
            "Anti-Trump Events - 2025",
            style={
                'fontFamily': FONT_FAMILY,
                'fontWeight': 'bold',
                'fontSize': '2.5rem',
                'color': PRIMARY_BLUE,
                'textAlign': 'center',
                'marginBottom': '24px'
            }
        ),
        dcc.Tabs(
            id='dashboard-tabs',
            value='map',
            children=[
                dcc.Tab(label='Map', value='map'),
                dcc.Tab(label='Metrics', value='metrics'),
                dcc.Tab(label='Graphs', value='graphs'),
                dcc.Tab(label='Table', value='table')
            ],
            style={'marginBottom': '16px'}
        ),
        html.Div([
            html.Div([
                dcc.Graph(id='map-graph'),
                html.Div(id='event-details-panel'),
                html.Div(id='no-data-message')
            ], id='map-tab', style={'display': 'block'}),
            html.Div([
                html.Div(id='kpi-cards', style={
                    'display': 'flex',
                    'justifyContent': 'space-between',
                    'gap': '12px',
                    'marginBottom': '24px'
                }),
                html.Div([
                    html.Div(id='largest-event-kpi', style={
                        'flex': '1',
                        'textAlign': 'center',
                        'padding': '12px',
                        'borderRadius': '12px',
                        'backgroundColor': PRIMARY_BLUE,
                        'color': PRIMARY_WHITE,
                        'fontWeight': 'bold',
                        'boxShadow': '0 2px 8px rgba(36,76,196,0.08)',
                        'margin': '0 6px'
                    }),
                    html.Div(id='largest-day-kpi', style={
                        'flex': '1',
                        'textAlign': 'center',
                        'padding': '12px',
                        'borderRadius': '12px',
                        'backgroundColor': PRIMARY_RED,
                        'color': PRIMARY_WHITE,
                        'fontWeight': 'bold',
                        'boxShadow': '0 2px 8px rgba(36,76,196,0.08)',
                        'margin': '0 6px'
                    })
                ], style={
                    'display': 'flex',
                    'gap': '12px',
                    'marginBottom': '24px'
                }),
                html.Div([
                    html.Div(id='no-injuries-kpi', style={
                        'flex': '1',
                        'textAlign': 'center',
                        'padding': '12px',
                        'borderRadius': '12px',
                        'backgroundColor': PRIMARY_BLUE,
                        'color': PRIMARY_WHITE,
                        'fontWeight': 'bold',
                        'boxShadow': '0 2px 8px rgba(36,76,196,0.08)',
                        'margin': '0 6px'
                    }),
                    html.Div(id='no-arrests-kpi', style={
                        'flex': '1',
                        'textAlign': 'center',
                        'padding': '12px',
                        'borderRadius': '12px',
                        'backgroundColor': PRIMARY_BLUE,
                        'color': PRIMARY_WHITE,
                        'fontWeight': 'bold',
                        'boxShadow': '0 2px 8px rgba(36,76,196,0.08)',
                        'margin': '0 6px'
                    }),
                    html.Div(id='no-damage-kpi', style={
                        'flex': '1',
                        'textAlign': 'center',
                        'padding': '12px',
                        'borderRadius': '12px',
                        'backgroundColor': PRIMARY_BLUE,
                        'color': PRIMARY_WHITE,
                        'fontWeight': 'bold',
                        'boxShadow': '0 2px 8px rgba(36,76,196,0.08)',
                        'margin': '0 6px'
                    })
                ], style={
                    'display': 'flex',
                    'gap': '12px'
                })
            ], id='metrics-tab', style={'display': 'none'}),
            html.Div([
                dcc.Graph(id='momentum-graph'),
                dcc.Graph(id='daily-graph'),
                dcc.Graph(id='cumulative-graph'),
                dcc.Graph(id='daily-participant-graph')
            ], id='graphs-tab', style={'display': 'none'}),
        html.Div([
            dash_table.DataTable(
                id='filtered-table',
                columns=[],  # Columns will be set dynamically in the callback
                style_table={'overflowY': 'auto', 'maxHeight': '500px', 'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '10px', 'fontFamily': FONT_FAMILY, 'fontSize': '14px'},
                style_header={'fontWeight': 'bold', 'backgroundColor': PRIMARY_BLUE, 'color': PRIMARY_WHITE},
                virtualization=True,
                fixed_rows={'headers': True}
            )
        ], id='table-tab', style={'display': 'none'})
        ])
    ], style={
        'width': 'calc(100% - 320px)',
        'padding': '32px',
        'boxSizing': 'border-box',
        'flexGrow': '1',
        'overflow': 'auto',
        'backgroundColor': '#f5f6fa',
        'borderRadius': '12px 0 0 12px',
        'fontFamily': FONT_FAMILY,
        'transition': 'width 0.3s ease-in-out'
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


@app.callback(
    [
        Output('map-tab', 'style'),
        Output('metrics-tab', 'style'),
        Output('graphs-tab', 'style'),
        Output('table-tab', 'style')  # New table tab
    ],
    Input('dashboard-tabs', 'value')
)
def update_tab_display(tab):
    return [
        {'display': 'block'} if tab == 'map' else {'display': 'none'},
        {'display': 'block'} if tab == 'metrics' else {'display': 'none'},
        {'display': 'block'} if tab == 'graphs' else {'display': 'none'},
        {'display': 'block'} if tab == 'table' else {'display': 'none'}  # Handle table tab visibility
    ]

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='filtered-data'),
    dashboard_layout
])

def jitter_coords(df, lat_col='lat', lon_col='lon', jitter_amount=0.05):
    
    # Add random jitter to duplicate lat/lon pairs in a DataFrame.
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


# --- SPEED OPTIMIZATION SECTION ---

# 1. Only copy DataFrame when necessary (avoid df.copy() in filter_data)
# 2. Remove unnecessary print statements
# 3. Limit columns stored in dcc.Store (filtered-data)
# 4. Use .loc for filtering to avoid chained assignment warnings
# 5. Avoid unnecessary .apply in filter_data

@cache.memoize(timeout=120)
def filter_data(
    start_date, end_date, size_filter, org_search, state_filter,
    any_outcomes_filter
):
    dff = df.copy()
    mask = pd.Series(True, index=dff.index)

    # Apply filters
    if start_date and end_date:
        mask &= (dff['date'] >= start_date) & (dff['date'] <= end_date)
    if size_filter == 'has':
        mask &= dff['size_mean'].notna()
    elif size_filter == 'no':
        mask &= dff['size_mean'].isna()
    if org_search:
        pattern = '|'.join(map(re.escape, org_search.lower().split(',')))
        mask &= dff['organizations'].str.contains(pattern, na=False, regex=True)
    if state_filter:
        mask &= dff['state'].isin(state_filter)

    # Apply outcomes filters
    for outcome in any_outcomes_filter:
        if outcome == 'arrests_any':
            mask &= dff['arrests'].notna() & (dff['arrests'] > 0)
        elif outcome == 'participant_injuries_any':
            mask &= dff['participant_injuries'].notna() & (dff['participant_injuries'] > 0)

    return dff[mask]

@cache.memoize(timeout=120)
def aggregate_events_for_map(dff_map):
    df_map = dff_map.copy()

    # Helper function to determine the best location label
    def best_location(row):
        loc = str(row.get('location', 'Unknown')).strip()
        if loc and loc.lower() != 'nan':
            return loc
        loc2 = str(row.get('locality', 'Unknown')).strip()
        if loc2 and loc2.lower() != 'nan':
            return loc2
        state = row.get('state', 'Unknown')
        date = row['date'].date() if pd.notnull(row.get('date')) else 'Unknown'
        return f"{state}, {date}"

    # Apply the best location logic
    df_map['location_label'] = df_map.apply(best_location, axis=1)
    df_map['location_label'] = df_map['location_label'].replace('', 'Unknown').fillna('Unknown')

    # Create event labels for hover text
    df_map['event_label'] = df_map.apply(
        lambda row: (
            f"<b>{row.get('title', 'Unknown')}</b><br>"
            f"Date: {row['date'].date() if pd.notnull(row['date']) else 'Unknown'}<br>"
            f"Organizations: {row.get('organizations', 'Unknown')}<br>"
            f"Participants: {row.get('size_mean', 'Unknown')}"
        ),
        axis=1
    )

    # Drop rows without valid latitude and longitude
    df_map = df_map.dropna(subset=['lat', 'lon'])

    # Aggregate data for the map
    agg = df_map.groupby('location_label').agg(
        lat=('lat', 'first'),
        lon=('lon', 'first'),
        count=('title', 'size'),
        event_list=('event_label', lambda x: "<br><br>".join(x)),
        title=('title', lambda x: "; ".join(x.fillna('Unknown').astype(str).replace('', 'Unknown'))),
        size_mean=('size_mean', lambda x: x.mean() if x.notna().any() else np.nan)
    ).reset_index()

    # Create hover text with 'Unknown' for missing values
    agg['hover'] = agg.apply(
        lambda row: (
            f"<b>{row['location_label']}</b><br>"
            f"Events at this site: {row['count']}<br><br>"
            f"<b>Events:</b><br>{row['event_list']}"
        ),
        axis=1
    )

    # Ensure text field is populated
    agg['text'] = agg['location_label']

    return agg

@app.callback(
    [
        Output('map-graph', 'figure'),
        Output('momentum-graph', 'figure'),
        Output('daily-graph', 'figure'),
        Output('kpi-cards', 'children'),
        Output('filtered-data', 'data'),
        Output('cumulative-graph', 'figure'),
        Output('daily-participant-graph', 'figure'),
        Output('largest-event-kpi', 'children'),
        Output('largest-day-kpi', 'children'),
        Output('no-injuries-kpi', 'children'),
        Output('no-arrests-kpi', 'children'),
        Output('no-damage-kpi', 'children')
    ],
    [
        Input('url', 'pathname'),
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date'),
        Input('size-filter', 'value'),
        Input('org-search', 'value'),
        Input('state-filter', 'value'),
        Input('any-outcomes-filter', 'value')
    ]
)
def update_all(
    pathname, start_date=None, end_date=None, size_filter='all', org_search=None,
    state_filter=None, any_outcomes_filter=None
):
    # Set default values if inputs are None
    start_date = start_date or df['date'].min()
    end_date = end_date or df['date'].max()
    size_filter = size_filter or 'all'
    org_search = org_search or ''
    state_filter = state_filter or []
    any_outcomes_filter = any_outcomes_filter or []

    dff = filter_data(
        start_date, end_date, size_filter, org_search, state_filter,
        any_outcomes_filter
    )

    # Handle empty dataset
    if dff.empty:
        empty_fig = go.Figure()
        return (
            empty_fig, empty_fig, empty_fig, [],
            dff.to_json(date_format='iso', orient='split'),
            empty_fig, empty_fig, [], [], [], [], []
        )

    # Calculate metrics
    total_events = len(dff)
    percent_no_injuries = 0
    percent_no_arrests = 0
    percent_no_damage = 0

    if total_events > 0:
        percent_no_injuries = 100 * (dff['participant_injuries'].isna().sum() / total_events)
        percent_no_arrests = 100 * (dff['arrests'].isna().sum() / total_events)
        percent_no_damage = 100 * (dff['property_damage'].isna().sum() / total_events)

    # Build KPIs
    no_injuries_kpi = [
        html.Div([
            "🩹 ",  # Injury icon
            html.Span(f"{percent_no_injuries:.1f}%", style={
                'fontSize': '1.2rem',
                'fontWeight': '700',
                'color': PRIMARY_WHITE,
                'fontFamily': FONT_FAMILY,
                'letterSpacing': '0.05em'
            })
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
        html.Div("% with No Injuries", style={
            'fontSize': '0.95rem',
            'color': PRIMARY_WHITE,
            'fontFamily': FONT_FAMILY,
            'letterSpacing': '0.08em'
        })
    ]
    no_arrests_kpi = [
        html.Div([
            "🚔 ",  # Arrest icon
            html.Span(f"{percent_no_arrests:.1f}%", style={
                'fontSize': '1.2rem',
                'fontWeight': '700',
                'color': PRIMARY_WHITE,
                'fontFamily': FONT_FAMILY,
                'letterSpacing': '0.05em'
            })
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
        html.Div("% with No Arrests", style={
            'fontSize': '0.95rem',
            'color': PRIMARY_WHITE,
            'fontFamily': FONT_FAMILY,
            'letterSpacing': '0.08em'
        })
    ]
    no_damage_kpi = [
        html.Div([
            "🏚️ ",  # Damage icon
            html.Span(f"{percent_no_damage:.1f}%", style={
                'fontSize': '1.2rem',
                'fontWeight': '700',
                'color': PRIMARY_WHITE,
                'fontFamily': FONT_FAMILY,
                'letterSpacing': '0.05em'
            })
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
        html.Div("% with No Property Damage", style={
            'fontSize': '0.95rem',
            'color': PRIMARY_WHITE,
            'fontFamily': FONT_FAMILY,
            'letterSpacing': '0.08em'
        })
    ]

    # Calculate metrics
    total_events = len(dff)
    total_participants = dff['size_mean'].sum() if 'size_mean' in dff.columns else 0
    peak_day = dff.groupby('date')['size_mean'].sum().max() if 'size_mean' in dff.columns else 0
    percent_us_pop = (peak_day / US_POPULATION) * 100 if peak_day else 0
    mean_size = dff['size_mean'].mean() if 'size_mean' in dff.columns else 0
    percent_no_size = 0
    if total_events > 0 and 'size_mean' in dff.columns:
        percent_no_size = 100 * dff['size_mean'].isna().sum() / total_events

    # Calculate largest event and largest day
    largest_event = dff['size_mean'].max() if 'size_mean' in dff.columns and not dff['size_mean'].isnull().all() else 0
    largest_day = dff.groupby('date')['size_mean'].sum().max() if 'size_mean' in dff.columns and not dff['size_mean'].isnull().all() else 0

    # KPI section
    kpis = [
        html.Div([
            html.Div(f"{total_events:,}", style={
                'fontSize': '2.2rem',
                'fontWeight': '700',
                'color': PRIMARY_WHITE,
                'letterSpacing': '0.05em',
                'fontFamily': FONT_FAMILY
            }),
            html.Div("Total Events", style={
                'fontSize': '1.1rem',
                'color': PRIMARY_WHITE,
                'letterSpacing': '0.08em',
                'fontFamily': FONT_FAMILY
            })
        ], style={
            'width': '24%',
            'textAlign': 'center',
            'padding': '18px 18px 10px 18px',
            'borderRadius': '16px',
            'backgroundColor': PRIMARY_BLUE,
            'boxShadow': '0 2px 8px rgba(36,76,196,0.08)'
        }),
        html.Div([
            html.Div(f"{percent_us_pop:.4f}%", style={
                'fontSize': '2.2rem',
                'fontWeight': '700',
                'color': PRIMARY_WHITE,
                'letterSpacing': '0.05em',
                'fontFamily': FONT_FAMILY
            }),
            html.Div("Largest Daily Participant Count as % of US Population", style={
                'fontSize': '1.1rem',
                'color': PRIMARY_WHITE,
                'letterSpacing': '0.08em',
                'fontFamily': FONT_FAMILY
            })
        ], style={
            'width': '24%',
            'textAlign': 'center',
            'padding': '18px 18px 10px 18px',
            'borderRadius': '16px',
            'backgroundColor': PRIMARY_RED,
            'boxShadow': '0 2px 8px rgba(36,76,196,0.08)'
        }),
        html.Div([
            html.Div(f"{percent_no_size:.1f}%", style={
                'fontSize': '2.2rem',
                'fontWeight': '700',
                'color': PRIMARY_WHITE,
                'letterSpacing': '0.05em',
                'fontFamily': FONT_FAMILY
            }),
            html.Div("Events Missing Size", style={
                'fontSize': '1.1rem',
                'color': PRIMARY_WHITE,
                'letterSpacing': '0.08em',
                'fontFamily': FONT_FAMILY
            })
        ], style={
            'width': '24%',
            'textAlign': 'center',
            'padding': '18px 18px 10px 18px',
            'borderRadius': '16px',
            'backgroundColor': PRIMARY_RED,
            'boxShadow': '0 2px 8px rgba(36,76,196,0.08)'
        }),
        html.Div([
            html.Div(f"{mean_size:,.0f}", style={
                'fontSize': '2.2rem',
                'fontWeight': '700',
                'color': PRIMARY_WHITE,
                'letterSpacing': '0.05em',
                'fontFamily': FONT_FAMILY
            }),
            html.Div("Average Participant Count", style={
                'fontSize': '1.1rem',
                'color': PRIMARY_WHITE,
                'letterSpacing': '0.08em',
                'fontFamily': FONT_FAMILY
            })
        ], style={
            'width': '24%',
            'textAlign': 'center',
            'padding': '18px 18px 10px 18px',
            'borderRadius': '16px',
            'backgroundColor': PRIMARY_BLUE,
            'boxShadow': '0 2px 8px rgba(36,76,196,0.08)'
        }),
    ]

    # Defensive: Ensure 'lat' and 'lon' columns exist and are not all missing
    if 'lat' not in dff.columns or 'lon' not in dff.columns or dff['lat'].isnull().all() or dff['lon'].isnull().all():
        empty_fig = go.Figure()
        # Define default values for largest_event_kpi and largest_day_kpi
        largest_event_kpi = []
        largest_day_kpi = []
        return (
            empty_fig, empty_fig, empty_fig, kpis,
            dff.to_json(date_format='iso', orient='split'),
            empty_fig, empty_fig,
            html.Div(
                "No events with valid location data for the selected filters.",
                style={'color': 'red', 'fontWeight': 'bold', 'fontSize': '1.2em', 'margin': '20px 0'}
            ),
            largest_event_kpi, largest_day_kpi
        )

    agg_map = aggregate_events_for_map(dff)

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
                color=PRIMARY_BLUE,
                sizemode='area',
                sizeref=sizeref,
                sizemin=5
            ),
            text=has_size['text'],
            hovertemplate=(
                "<b>%{text}</b><br><br>"
                "Events at this site: %{customdata[0]}<br>"
                "Participants: %{customdata[1]}<br>"
                "<extra></extra>" 
            ),
            customdata=has_size[['count', 'size_mean']].values,
            name="Has Size"
        ))

    # Plot locations without participant size (red)
    if not no_size.empty:
        fig_map.add_trace(go.Scattermapbox(
            lat=no_size['lat'],
            lon=no_size['lon'],
            mode='markers',
            marker=dict(
                size=12,
                color=PRIMARY_RED,
                sizemode='area',
                sizeref=1,
                sizemin=5
            ),
            text=no_size['text'],
            hovertemplate=(
                "<b>%{text}</b><br><br>"
                "Events at this site: %{customdata[0]}<br>"
                "<extra></extra>"
            ),
            customdata=no_size[['count']].values,  # Pass additional data for hovertemplate
            name="Missing Size"
        ))

    # Restore old fixed map zoom/center for consistent US view
    fig_map.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=3,
        mapbox_center={"lat": 39.8283, "lon": -98.5795},
        margin=standard_margin,
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.18,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        )
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
        name='Momentum'
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
        margin=standard_margin
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
                fig_daily.update_layout(margin=standard_margin)
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
            fig_cumulative.update_layout(margin=standard_margin)

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
            fig_daily_participants.update_layout(margin=standard_margin)
    def best_location(row):
        loc = str(row.get('location', 'Unknown')).strip()
        if loc and loc.lower() != 'nan':
            return loc
        loc2 = str(row.get('locality', 'Unknown')).strip()
        if loc2 and loc2.lower() != 'nan':
            return loc2
        state = row.get('state', 'Unknown')
        date = row['date'].date() if pd.notnull(row.get('date')) else 'Unknown'
        return f"{state}, {date}"

    dff['location_label'] = dff.apply(best_location, axis=1)
    store_cols = [
        'date', 'locality', 'state', 'resolved_locality', 'resolved_state', 'resolved_county',
        'fips_code', 'lat', 'lon', 'location', 'title', 'event_type', 'organizations',
        'participants', 'notables', 'targets', 'claims_summary', 'claims_verbatim',
        'issue_tags_summary', 'issue_tags_verbatim', 'issues', 'valence', 'conf',
        'macroevent', 'online', 'size_text', 'size_low', 'size_high', 'size_mean',
        'size_cat', 'participant_measures', 'police_measures', 'participant_injuries',
        'police_injuries', 'arrests', 'property_damage', 'arrests_any',
        'participant_casualties_any', 'police_casualties_any', 'property_damage_any',
        'participant_deaths', 'police_deaths', 'notes', 'source1', 'source2', 'source3',
        'source4', 'source5', 'source6', 'source7', 'source8', 'source9', 'source10',
        'source11', 'source12', 'source13', 'source14', 'source15', 'source16', 'source17',
        'source18', 'source19', 'source20', 'source21', 'source22', 'source23', 'source24',
        'source25', 'source26', 'source27', 'source28', 'source29', 'source30', 'location_label'
    ]

    existing_cols = [col for col in store_cols if col in dff.columns]
    dff_store = dff[existing_cols].copy()

    # Build the small KPI boxes
    largest_event_kpi = [
        html.Div(f"{largest_event:,.0f} participants", style={
            'fontSize': '1.2rem',
            'fontWeight': '700',
            'color': PRIMARY_WHITE,
            'fontFamily': FONT_FAMILY,
            'letterSpacing': '0.05em'
        }),
        html.Div("Largest Event", style={
            'fontSize': '0.95rem',
            'color': PRIMARY_WHITE,
            'fontFamily': FONT_FAMILY,
            'letterSpacing': '0.08em'
        })
    ]
    largest_day_kpi = [
        html.Div(f"{largest_day:,.0f} participants", style={
            'fontSize': '1.2rem',
            'fontWeight': '700',
            'color': PRIMARY_WHITE,
            'fontFamily': FONT_FAMILY,
            'letterSpacing': '0.05em'
        }),
        html.Div("Largest Day", style={
            'fontSize': '0.95rem',
            'color': PRIMARY_WHITE,
            'fontFamily': FONT_FAMILY,
            'letterSpacing': '0.08em'
        })
    ]
    return (
        fig_map, fig_momentum, fig_daily, kpis, 
        dff_store.to_json(date_format='iso', orient='split'),
        fig_cumulative, fig_daily_participants,
        largest_event_kpi, largest_day_kpi,
        no_injuries_kpi, no_arrests_kpi, no_damage_kpi
    )


@app.callback(
    Output('event-details-panel', 'children'),
    Input('map-graph', 'clickData'),
    State('filtered-data', 'data')
)
def update_event_details(click_data, filtered_data):
    if not click_data or not filtered_data:
        return html.Div(
            "Click a map marker to see event details.",
            style={'color': '#555', 'fontSize': '.9em', 'fontStyle': 'italic', 'textAlign': 'center', 'padding': '16px 0'}
        )

    try:
        dff = pd.read_json(io.StringIO(filtered_data), orient='split')
        point = click_data['points'][0]
        location_label = point.get('text') or point.get('hovertext')
        if not location_label:
            return html.Div("No details available for this location.", style={'color': '#555', 'margin': '12px 0'})

        # Normalize for robust matching
        def norm(x):
            return str(x).strip().lower() if pd.notnull(x) else ''

        norm_label = norm(location_label)
        dff['__norm_label'] = dff['location_label'].apply(norm)
        location_events = dff[dff['__norm_label'] == norm_label]

        # Fallback: substring match if exact match fails
        if location_events.empty:
            location_events = dff[dff['__norm_label'].str.contains(re.escape(norm_label))]
            if location_events.empty:
                return html.Div("No event details found for this marker.", style={'color': '#555', 'margin': '12px 0'})

        detail_fields = [
            ('Title', 'title'),
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
        for _, event in location_events.iterrows():
            event_detail = []
            for label, col in detail_fields:
                value = event.get(col, 'Unknown')
                if pd.isnull(value) or (isinstance(value, str) and (not value.strip() or value.strip().lower() == 'nan')):
                    value = 'Unknown'
                if col == 'date' and pd.notnull(value):
                    try:
                        value = pd.to_datetime(value).strftime('%Y-%m-%d')
                    except Exception:
                        value = 'Unknown'
                event_detail.append(html.P(f"{label}: {value}", style={'margin': '0 0 4px 0'}))

            title = event.get('title', 'Unknown')
            date = event.get('date', 'Unknown')
            if pd.notnull(date):
                try:
                    date = pd.to_datetime(date).strftime('%Y-%m-%d')
                except Exception:
                    date = 'Unknown'
            header = f"{title} - {date}"

            details.append(
                html.Details([
                    html.Summary(header, style={'fontWeight': 'bold', 'fontSize': '1.1em'}),
                    html.Div(event_detail, style={'marginLeft': '12px'})
                ], open=True, style={'marginBottom': '16px'})
            )

        return html.Div(details, style={'padding': '12px'})

    except Exception as e:
        return html.Div(
            f"An error occurred while loading event details: {str(e)}",
            style={'color': 'red', 'fontSize': '.9em', 'fontStyle': 'italic', 'textAlign': 'center', 'padding': '16px 0'}
        )


@app.callback(
    [Output('filtered-table', 'data'),
     Output('filtered-table', 'columns')],
    Input('filtered-data', 'data')
)
def update_table(data_json):
    if not data_json:
        return [], []
    dff = pd.read_json(io.StringIO(data_json), orient='split')
    columns = [
        {'name': col, 'id': col, 'type': 'text'} if col != 'date' else
        {'name': col, 'id': col, 'type': 'datetime', 'format': {'specifier': '%Y-%m-%d'}}
        for col in dff.columns
    ]
    return dff.to_dict('records'), columns


@app.callback(
    Output("download-data", "data"),  # Use the correct dcc.Download ID
    Input("download-btn", "n_clicks"),  # Triggered by the download button
    State("filtered-data", "data"),  # Use the filtered data
    State("download-choice", "value"),  # Check if the user wants filtered or full data
    prevent_initial_call=True
)
def download_filtered_table(n_clicks, filtered_data, download_choice):
    if not filtered_data:
        return no_update

    # Load the filtered data
    dff = pd.read_json(io.StringIO(filtered_data), orient='split')

    # If the user selects "Full Dataset," return the full dataframe
    if download_choice == "full":
        return dcc.send_data_frame(df.to_csv, filename="full_dataset.csv")

    # Otherwise, return the filtered dataset
    return dcc.send_data_frame(dff.to_csv, filename="filtered_dataset.csv")


@app.callback(
    [Output('sidebar-content', 'children'),
     Output('toggle-definitions', 'children')],
    Input('toggle-definitions', 'n_clicks'),
    prevent_initial_call=True
)
def toggle_sidebar_content(n_clicks):
    if n_clicks % 2 == 1:
        return definitions_panel, "Show Filters"
    else:
        return filter_panel, "Show Data Definitions & Sources"


# Uncomment the following 2 lines to run the app directly and test locally. Comment back out when deploying to production.
# if __name__ == '__main__':
#     app.run(debug=True)