import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, no_update, dash_table, ctx
from flask_caching import Cache
import os
import re
import io
import time

# -------------------------
# Data & constants
# -------------------------
file_path = "ccc_anti_trump.csv"
US_POPULATION = 340_100_000
PROCESSED = "processed_data.parquet"

# Ensure clean preprocess (mirrors original behavior)
if os.path.exists(PROCESSED):
    os.remove(PROCESSED)

if os.path.exists(PROCESSED):
    df = pd.read_parquet(PROCESSED)
else:
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

    for col in ['participant_injuries', 'police_injuries', 'arrests', 'participant_deaths', 'police_deaths']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'property_damage' in df.columns:
        df['property_damage_any'] = (
            df['property_damage'].notna() & (df['property_damage'].astype(str).str.strip() != "")
        ).astype(int)

    df.to_parquet(PROCESSED)

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.title = "Protest Dashboard"

cache = Cache(app.server, config={'CACHE_TYPE': 'filesystem', 'CACHE_DIR': 'cache-directory'})
os.makedirs('cache-directory', exist_ok=True)

# Plotly margins
standard_margin = dict(t=30, b=20, l=18, r=18)

# -------------------------
# Panels (structure only; styling via CSS)
# -------------------------
filter_panel = html.Div([
    html.H2("Filters", className="panel-title"),
    html.Div([
        html.Label(
            "Date Range",
            title=("The Crowd Counting Consortium (CCC) only releases new data once a month, during the first week. "
                   "Recent events may not appear until after the next monthly update."),
            className="label-help"
        )
    ], className="date-label-row"),
    dcc.DatePickerRange(
        id='date-range',
        start_date=df['date'].min(),
        end_date=df['date'].max(),
        display_format='YYYY-MM-DD',
        day_size=29,
        className="date-range"
    ),
    html.Div(
        "Click the date values to select dates to filter on. Note data is on a monthly release schedule, so recent events may not appear until the next update.",
        className="date-help"
    ),
    html.Label("National Day of Action", className="std-label"),
    dcc.Dropdown(
        id='day-of-action',
        options=[
            {'label': 'February 5', 'value': '2025-02-05'},
            {'label': 'February 17', 'value': '2025-02-17'},
            {'label': 'March 4', 'value': '2025-03-04'},
            {'label': 'April 5', 'value': '2025-04-05'},
            {'label': 'April 19', 'value': '2025-04-19'},
            {'label': 'May 1', 'value': '2025-05-01'},
            {'label': 'June 14', 'value': '2025-06-14'},
            {'label': 'July 4', 'value': '2025-07-04'},
            {'label': 'July 17', 'value': '2025-06-17'},
        ],
        placeholder="Select a date...",
        className="std-dropdown",
        clearable=True
    ),

    html.Label("Participant Count Filter", className="std-label"),
    dcc.RadioItems(
        id='size-filter',
        options=[
            {'label': 'Has participant count', 'value': 'has'},
            {'label': 'No participant count', 'value': 'no'},
            {'label': 'All events', 'value': 'all'}
        ],
        value='all',
        labelStyle={'display': 'block', 'marginBottom': '6px'},
        className="radio-block"
    ),

    html.Label("Organization Search", className="std-label"),
    dcc.Input(
        id='org-search',
        type='text',
        placeholder="Type organizations, separated by commas",
        className="text-input"
    ),
    html.Div("↩ Separate multiple organizations with commas", className="subtle-hint"),

    html.Label("State/Territory", className="std-label"),
    dcc.Dropdown(
        id='state-filter',
        options=[{'label': s, 'value': s} for s in sorted(df['state'].dropna().unique())],
        value=[],
        multi=True,
        placeholder="Select state(s) or territory(ies)",
        clearable=True,
        className="std-dropdown"
    ),

    html.Label("City", className="std-label"),
    dcc.Dropdown(
        id='city-filter',
        options=[],
        value=[],
        multi=True,
        placeholder="Select a state first for cities",
        clearable=True,
        className="std-dropdown"
    ),

    html.Label("Event Outcomes", className="std-label"),
    dcc.Checklist(
        id='any-outcomes-filter',
        options=[
            {'label': 'Any Arrested Protesters', 'value': 'arrests_any'},
            {'label': 'Any Participant Injuries', 'value': 'participant_injuries_any'},
            {'label': 'Any Police Injuries', 'value': 'police_injuries_any'},
            {'label': 'Any Property Damage', 'value': 'property_damage_any'},
        ],
        value=[],
        className="checklist"
    ),

    html.Label("Download Data", className="std-label"),
    dcc.Dropdown(
        id='download-choice',
        options=[
            {'label': 'Filtered View Only', 'value': 'filtered'},
            {'label': 'Full Dataset', 'value': 'full'}
        ],
        value='filtered',
        clearable=False,
        className="std-dropdown"
    ),
    html.Button("Download Dataset", id="download-btn", className="primary-button hover-button"),
    dcc.Download(id="download-data"),

    html.Div([
        html.A(
            "Is your data/event missing? Click here to learn how to fix it!",
            href="https://bit.ly/m/WeCount",
            target="_blank",
            className="primary-link-button"
        )
    ], className="centered")
], id='filter-panel', className="panel filter-panel")

definitions_panel = html.Div([
    html.H3("Data Definitions & Sources", className="panel-title"),
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
        ], className="top-margined"),
        html.Ul([
            html.Li([html.B("Location: "), "Based on city-level geocoding. If multiple events occurred in the same city on the same day, their locations are jittered for visualization. Exact event locations may not be available; city centroids or modified city coordinates are used."]),
            html.Li([html.B("Anti-Trump events: "), "This dashboard uses a dataset filtered to include only anti-Trump events. Events that may be against him but do not mention him explicitly may not be included here."]),
            html.Li([html.B("Participant Count: "), "The 'size_mean' field is an average of the upper and lower range estimates of crowd size, as reported. This provides a standardized estimate of participant size for each event. Some events may have missing or uncertain size estimates."]),
            html.Li([html.B("Momentum of Dissent: "), 
                "For each day, the sum of estimated participants is multiplied by the number of events that day. "
                "The 'Momentum of Dissent' shown in the dashboard is the sum of these daily values over the most recent 7 days (a rolling 7-day sum). "
                "This highlights periods of sustained, high-volume protest activity. The concept and approach for 'momentum' as a protest metric is inspired by the methodology described in: ",
                html.A("Chenoweth, E., Perkoski, E., & Kang, S. (2017). State Repression and Nonviolent Resistance. Research in Social Movements, Conflicts and Change, 41, 85–117.",
                       href="https://bura.brunel.ac.uk/bitstream/2438/19075/1/FullText.pdf", target="_blank"),
                ". In that work, 'movement momentum' is used to capture the intensity and persistence of protest activity over time."
            ]),
            html.Li([html.B("Organizations: "), "Organizations are listed as a semicolon-separated string. Organization search matches any substring in this field."]),
            html.Li([html.B("State/Territory: "), "Includes U.S. states and territories as reported in the original data."]),
            html.Li([html.B("Date: "), "Date of the event (YYYY-MM-DD)."]),
            html.Li([html.B("Cumulative Total Events: "), "Number of events after all filters are applied."]),
            html.Li([html.B("Largest Daily Participant Count as % of US population: "), "Calculated as the largest single-day sum of 'size_mean' in the filtered data, divided by the 2024 U.S. population estimate (340,100,000)."]),
            html.Li([html.B("Download: "), "You can download either the filtered view or the full dataset as CSV."]),
            html.Li([html.B("More info: "), "See the CCC ", html.A("Harvard Dataverse", href="https://dataverse.harvard.edu/dataverse/ccc", target="_blank"), " for full metadata and documentation."])
        ])
    ], className="definitions-body")
], id='definitions-panel', className="panel definitions-panel")

# -------------------------
# Sidebar (single definition; width 380px to match layout calc)
# -------------------------
def get_sidebar(is_open: bool):
    toggle_icon = "❮" if is_open else "❯"
    toggle_tab = html.Div(
        toggle_icon,
        id='sidebar-toggle-tab',
        n_clicks=0,
        className=f"sidebar-toggle-tab {'open' if is_open else 'closed'}"
    )

    content = html.Div([
        html.Div(filter_panel, id='filter-panel-container', className=f\"{'visible' if is_open else 'hidden'}\"),
        html.Div(definitions_panel, id='definitions-panel-container', className=\"hidden\")
    ], id='sidebar-content', className='sidebar-content')

    bottom_btn = html.Button(
        id='toggle-definitions',
        n_clicks=0,
        children="Show Data Definitions & Sources",
        className=f"toggle-definitions hover-button {'shown' if is_open else 'hidden'}"
    )
    return html.Div([
        html.Div([content, bottom_btn], id='sidebar', className=f"sidebar {'open' if is_open else 'closed'}"),
        toggle_tab
    ], className="sidebar-wrapper")

# -------------------------
# Layout
# -------------------------
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='filtered-data'),
    dcc.Store(id='sidebar-open', data=True),
    html.Div(id='sidebar-dynamic'),
    html.Div(id='main-content', children=[
        html.Div("Anti-Trump Events - 2025", className="title"),
        html.Div(
            f"Data current to {df['date'].max().strftime('%Y-%m-%d') if not df['date'].isna().all() else 'Unknown'}",
            className="subtitle"
        ),
        html.Div([
            html.Div([
                html.Div(id='total-events-kpi', className="kpi-box kpi-blue"),
                html.Div(id='mean-size-kpi', className="kpi-box kpi-red"),
                html.Div(id='no-injuries-kpi', className="kpi-box kpi-blue"),
                html.Div(id='no-arrests-kpi', className="kpi-box kpi-red"),
                html.Div(id='no-damage-kpi', className="kpi-box kpi-blue"),
            ], className="kpi-row"),
            html.Div([
                html.Div(id='total-participants-kpi', className="kpi-box kpi-red"),
                html.Div(id='largest-event-kpi', className="kpi-box kpi-blue"),
                html.Div(id='largest-day-kpi', className="kpi-box kpi-blue"),
                html.Div(id='percent-us-pop-kpi', className="kpi-box kpi-red"),
            ], className="kpi-row second"),
            html.Div(id='threshold-text', className="threshold-box")
        ], className="kpi-section"),
        dcc.Tabs(
            id='dashboard-tabs',
            value='map',
            children=[
                dcc.Tab(label='Map', value='map', children=[
                    dcc.Graph(id='map-graph', config={'displayModeBar': True, 'modeBarButtonsToRemove': ['select2d','lasso2d']}),
                    html.Div(id='event-details-panel')
                ]),
                dcc.Tab(label='Graphs', value='graphs', children=[
                    html.Div([
                        html.Div("Momentum of Dissent", className="graph-title"),
                        dcc.Graph(id='momentum-graph', config={'displayModeBar': True, 'modeBarButtonsToRemove': ['select2d','lasso2d']})
                    ]),
                    html.Div([
                        html.Div("Daily Event Count", className="graph-title"),
                        dcc.Graph(id='daily-graph', config={'displayModeBar': True, 'modeBarButtonsToRemove': ['select2d','lasso2d']})
                    ]),
                    html.Div([
                        html.Div("Cumulative Total Events", className="graph-title"),
                        dcc.Graph(id='cumulative-graph', config={'displayModeBar': True, 'modeBarButtonsToRemove': ['select2d','lasso2d']})
                    ]),
                    html.Div([
                        html.Div("Daily Participant Count", className="graph-title"),
                        dcc.Graph(id='daily-participant-graph', config={'displayModeBar': True, 'modeBarButtonsToRemove': ['select2d','lasso2d']})
                    ])
                ]),
                dcc.Tab(label='Table', value='table', children=[
                    dash_table.DataTable(
                        id='filtered-table',
                        columns=[],
                        style_table={'overflowY': 'auto', 'maxHeight': '500px', 'overflowX': 'auto', 'width': '100%', 'minWidth': '100%'},
                        style_cell={'textAlign': 'left', 'padding': '10px'},
                        virtualization=True,
                        fixed_rows={'headers': True}
                    )
                ])
            ],
            className="tabs-wrapper"
        ),
        html.Div(id='footer-message', className="footer-message")
    ], className="main-content")
], className="app-root")

# -------------------------
# Sidebar callbacks
# -------------------------
@app.callback(
    Output('sidebar-open', 'data'),
    Input('sidebar-toggle-tab', 'n_clicks'),
    State('sidebar-open', 'data'),
    prevent_initial_call=True
)
def toggle_sidebar(n, is_open):
    return not is_open

@app.callback(
    Output('sidebar-dynamic', 'children'),
    Output('main-content', 'style'),
    Input('sidebar-open', 'data'),
    prevent_initial_call=False
)
def render_sidebar(is_open):
    # Keep dynamic width via inline style to match the original responsive behavior exactly
    main_style = {
        'width': 'calc(100% - 380px)' if is_open else '100%'
    }
    return get_sidebar(is_open), main_style

# -------------------------
# Helpers
# -------------------------
def jitter_coords(dff, lat_col='lat', lon_col='lon', jitter_amount=0.01):
    dff = dff.copy().reset_index(drop=True)
    coords = dff[[lat_col, lon_col]].round(5).astype(str).agg('_'.join, axis=1)
    counts = coords.value_counts()
    dup_coords = counts[counts > 1].index
    for coord in dup_coords:
        idxs = dff.index[coords == coord].tolist()
        n = len(idxs)
        if n <= 1:
            continue
        center_lat = float(dff.at[idxs[0], lat_col])
        center_lon = float(dff.at[idxs[0], lon_col])
        for k, i in enumerate(idxs[1:], 1):
            angle = 2 * np.pi * (k - 1) / (n - 1)
            radius = jitter_amount
            dff.at[i, lat_col] = center_lat + np.cos(angle) * radius
            dff.at[i, lon_col] = center_lon + np.sin(angle) * radius
    return dff

# Speed optimizations preserved
@cache.memoize(timeout=120)
def filter_data(start_date, end_date, size_filter, org_search, state_filter, city_filter, any_outcomes_filter):
    dff = df
    mask = pd.Series(True, index=dff.index)

    for col in ['arrests', 'participant_injuries', 'police_injuries']:
        if col in dff.columns:
            dff[col] = pd.to_numeric(dff[col], errors='coerce')

    if start_date and end_date:
        mask &= (dff['date'] >= start_date) & (dff['date'] <= end_date)

    if size_filter == 'has':
        mask &= dff['size_mean'].notna()
    elif size_filter == 'no':
        mask &= dff['size_mean'].isna()

    if org_search and org_search.strip():
        orgs = [o.strip() for o in org_search.lower().split(',') if o.strip()]
        if orgs:
            pattern = '|'.join(map(re.escape, orgs))
            mask &= dff['organizations'].str.contains(pattern, na=False, regex=True)

    if state_filter and len(state_filter) > 0:
        mask &= dff['state'].isin(state_filter)

    if city_filter and len(city_filter) > 0:
        mask &= dff['resolved_locality'].isin(city_filter)

    for outcome in any_outcomes_filter or []:
        if outcome == 'arrests_any':
            mask &= dff['arrests'].notna() & (dff['arrests'] > 0)
        elif outcome == 'participant_injuries_any':
            mask &= dff['participant_injuries'].notna() & (dff['participant_injuries'] > 0)
        elif outcome == 'police_injuries_any':
            mask &= dff['police_injuries'].notna() & (dff['police_injuries'] > 0)
        elif outcome == 'property_damage_any':
            mask &= dff['property_damage_any'] == 1

    return dff.loc[mask].copy()

@cache.memoize(timeout=120)
def aggregate_events_for_map(dff_map):
    df_map = dff_map

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

    df_map['location_label'] = df_map.apply(best_location, axis=1)
    df_map['location_label'] = df_map['location_label'].replace('', 'Unknown').fillna('Unknown')

    df_map['event_label'] = df_map.apply(
        lambda row: (
            f"<b>{row.get('title', 'Unknown')}</b><br>"
            f"Date: {row['date'].date() if pd.notnull(row['date']) else 'Unknown'}<br>"
            f"Organizations: {row.get('organizations', 'Unknown')}<br>"
            f"Participants: {row.get('size_mean', 'Unknown')}"
        ),
        axis=1
    )

    df_map = df_map.dropna(subset=['lat', 'lon'])

    agg = df_map.groupby('location_label').agg(
        lat=('lat', 'first'),
        lon=('lon', 'first'),
        count=('title', 'size'),
        event_list=('event_label', lambda x: "<br><br>".join(x)),
        title=('title', lambda x: "; ".join(x.fillna('Unknown').astype(str).replace('', 'Unknown'))),
        size_mean=('size_mean', lambda x: x.mean() if x.notna().any() else np.nan)
    ).reset_index()

    agg['hover'] = agg.apply(
        lambda row: (
            f"<b>{row['location_label']}</b><br>"
            f"Events at this site: {row['count']}<br><br>"
            f"<b>Events:</b><br>{row['event_list']}"
        ),
        axis=1
    )
    agg['text'] = agg['location_label']
    return agg

# -------------------------
# Main callback
# -------------------------
@app.callback(
   [
       Output('map-graph', 'figure'),
       Output('momentum-graph', 'figure'),
       Output('daily-graph', 'figure'),
       Output('filtered-data', 'data'),
       Output('cumulative-graph', 'figure'),
       Output('daily-participant-graph', 'figure'),
       Output('total-events-kpi', 'children'),
       Output('largest-event-kpi', 'children'),
       Output('mean-size-kpi', 'children'),
       Output('largest-day-kpi', 'children'),
       Output('total-participants-kpi', 'children'),
       Output('no-injuries-kpi', 'children'),
       Output('no-arrests-kpi', 'children'),
       Output('no-damage-kpi', 'children'),
       Output('percent-us-pop-kpi', 'children'),
       Output('threshold-text', 'children'),
       Output('footer-message', 'children')
   ],
   [
       Input('date-range', 'start_date'),
       Input('date-range', 'end_date'),
       Input('day-of-action', 'value'),
       Input('size-filter', 'value'),
       Input('org-search', 'value'),
       Input('state-filter', 'value'),
       Input('city-filter', 'value'),
       Input('any-outcomes-filter', 'value'),
       Input('download-choice', 'value')
   ]
)
def update_all(start_date=None, end_date=None, day_of_action=None, size_filter=None, org_search=None,
               state_filter=None, city_filter=None, any_outcomes_filter=None, download_choice=None):

    if day_of_action:
        start_date = end_date = day_of_action

    dff = filter_data(start_date, end_date, size_filter, org_search, state_filter, city_filter, any_outcomes_filter)

    total_events = len(dff)
    total_participants = dff['size_mean'].sum() if 'size_mean' in dff.columns else 0
    mean_size = dff['size_mean'].mean() if 'size_mean' in dff.columns else 0
    largest_event = dff['size_mean'].max() if 'size_mean' in dff.columns and not dff['size_mean'].isnull().all() else 0
    largest_day = dff.groupby('date')['size_mean'].sum().max() if 'size_mean' in dff.columns and not dff['size_mean'].isnull().all() else 0
    percent_us_pop = (largest_day / US_POPULATION) * 100 if largest_day else 0
    percent_no_injuries = 100 * (dff['participant_injuries'].isna().sum() / total_events) if total_events > 0 else 0
    percent_no_arrests = 100 * (dff['arrests'].isna().sum() / total_events) if total_events > 0 else 0

    threshold_met = percent_us_pop >= 3.5
    threshold_text = html.Div([
        html.Span("3.5% threshold met?", className="threshold-label"),
        html.Span("✅ Yes" if threshold_met else "❌ No", className=f"{'yes' if threshold_met else 'no'}")
    ], className="threshold-inner")

    if 'property_damage_any' in dff.columns:
        percent_no_damage = 100 * (dff['property_damage_any'] == 0).sum() / total_events if total_events > 0 else 0
    else:
        percent_no_damage = 0

    def kpi_block(value_str, emoji, label):
        return [html.Div([
            html.Div(value_str, className="kpi-value"),
            html.Div(emoji, className="kpi-emoji"),
            html.Div(label, className="kpi-label")
        ])]

    total_events_kpi = kpi_block(f"{total_events:,}", "🗓️", "Total Events")
    largest_event_kpi = kpi_block(f"{largest_event:,.0f} participants", "🥇", "Largest Event")
    mean_size_kpi = kpi_block(f"{mean_size:,.0f}", "📊", "Average Participant Count")
    largest_day_kpi = kpi_block(f"{largest_day:,.0f} participants", "🥇", "Largest Day")
    total_participants_kpi = kpi_block(f"{total_participants:,.0f}", "🌟", "Total Participants")
    no_injuries_kpi = kpi_block(f"{percent_no_injuries:.1f}%", "🚑", "Events with No Injuries")
    no_arrests_kpi = kpi_block(f"{percent_no_arrests:.1f}%", "🚔", "Events with No Arrests")
    no_damage_kpi = kpi_block(f"{percent_no_damage:.1f}%", "🏚️", "Events with No Property Damage")

    # Guard for missing map coords
    if 'lat' not in dff.columns or 'lon' not in dff.columns or dff['lat'].isnull().all() or dff['lon'].isnull().all():
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="No matching data available for the selected filters.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=22, color="red"), align="center"
        )
        empty_fig.update_layout(mapbox_style="carto-positron", mapbox_zoom=3, mapbox_center={"lat": 39.8283, "lon": -98.5795},
                                margin=standard_margin, height=500, showlegend=False)
        empty_json = dff.to_json(date_format='iso', orient='split')
        dash_kpi = lambda label, icon="—": [html.Div([html.Div("-", className="kpi-value"), html.Div(icon, className="kpi-emoji"), html.Div(label, className="kpi-label")])]
        return (empty_fig, empty_fig, empty_fig, empty_json, empty_fig, empty_fig,
                dash_kpi("Total Events","🗓️"), dash_kpi("Largest Event","🥇"), dash_kpi("Average Participant Count","📊"),
                dash_kpi("Largest Day","🥇"), dash_kpi("Total Participants","🌟"),
                dash_kpi("Events with No Injuries","🚑"), dash_kpi("Events with No Arrests","🚔"),
                dash_kpi("Events with No Property Damage","🏚️"), dash_kpi("Most Daily Participants as % of USA","👥"),
                threshold_text, html.Div(className="footer-message"))

    # Build map
    dff_jittered = jitter_coords(dff, lat_col='lat', lon_col='lon', jitter_amount=0.01)
    agg_map = aggregate_events_for_map(dff_jittered)

    has_size = agg_map[agg_map['size_mean'].notna()]
    no_size = agg_map[agg_map['size_mean'].isna()]
    fig_map = go.Figure()

    if not has_size.empty:
        max_size = has_size['size_mean'].max()
        sizeref = 2.0 * max_size / (50.0 ** 2) if max_size > 0 else 1
        fig_map.add_trace(go.Scattermapbox(
            lat=has_size['lat'], lon=has_size['lon'], mode='markers',
            marker=dict(size=has_size['size_mean'], color="#244CC4", opacity=.5, sizemode='area', sizeref=sizeref, sizemin=5),
            text=has_size['text'], customdata=has_size[['count', 'size_mean']].values,
            hovertemplate="<b>%{text}</b><br><br>Events at this site: %{customdata[0]}<br>Participants: %{customdata[1]:,.0f}<br><extra></extra>",
            name="Has Participant Count", showlegend=False
        ))

    if not no_size.empty:
        fig_map.add_trace(go.Scattermapbox(
            lat=no_size['lat'], lon=no_size['lon'], mode='markers',
            marker=dict(size=12, color="#AC3C3D", opacity=.5, sizemode='area', sizeref=1, sizemin=5),
            text=no_size['text'], customdata=no_size[['count']].values,
            hovertemplate="<b>%{text}</b><br><br>Events at this site: %{customdata[0]}<br><extra></extra>",
            name="Missing Participant Count", showlegend=False
        ))

    fig_map.add_trace(go.Scattermapbox(lat=[None], lon=[None], mode='markers', marker=dict(size=16, color="#244CC4"),
                                       name="Has Participant Count", showlegend=True))
    fig_map.add_trace(go.Scattermapbox(lat=[None], lon=[None], mode='markers', marker=dict(size=16, color="#AC3C3D"),
                                       name="Missing Participant Count", showlegend=True))

    if not dff.empty and (city_filter and len(city_filter) > 0):
        center_lat = dff['lat'].mean(); center_lon = dff['lon'].mean(); zoom = 10 if len(city_filter) == 1 else 13
    elif not dff.empty and (state_filter and len(state_filter) > 0):
        center_lat = dff['lat'].mean(); center_lon = dff['lon'].mean(); zoom = 5
    else:
        center_lat = 39.8283; center_lon = -98.5795; zoom = 3

    fig_map.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=zoom,
        mapbox_center={"lat": center_lat, "lon": center_lon},
        margin=standard_margin, height=500, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5, font=dict(size=12))
    )

    # Momentum graph
    dff_momentum = dff[['date', 'participants_numeric']].dropna().set_index('date').resample('D').agg(['sum', 'count'])
    dff_momentum.columns = ['sum', 'count']
    dff_momentum['momentum'] = (dff_momentum['sum'] * dff_momentum['count']).rolling(7).sum()
    dff_momentum = dff_momentum.reset_index()

    fig_momentum = go.Figure()
    fig_momentum.add_trace(go.Scatter(
        x=dff_momentum['date'], y=dff_momentum['momentum'], mode='lines', name='Momentum',
        hovertemplate="<b>Momentum of Dissent</b>: %{y:,.0f}<br>Date: %{x|%Y-%m-%d}<br><span style='font-size:0.95em;'>Momentum of Dissent = (participants on a given day) × (number of events in the 7 days prior)</span><extra></extra>"
    ))
    valid = dff_momentum['momentum'].notna()
    if valid.sum() > 1:
        z = np.polyfit(pd.to_numeric(dff_momentum.loc[valid, 'date']), dff_momentum.loc[valid, 'momentum'], 1)
        p = np.poly1d(z)
        fig_momentum.add_trace(go.Scatter(x=dff_momentum.loc[valid, 'date'], y=p(pd.to_numeric(dff_momentum.loc[valid, 'date'])),
                                          mode='lines', name='Trendline of Momentum', line=dict(dash='dash', color='gray')))
    fig_momentum.update_layout(height=270, margin=standard_margin)

    # Daily event count
    dff_daily = dff.set_index('date').resample('D').size().reset_index(name='count')
    fig_daily = px.bar(dff_daily, x='date', y='count', height=270, template="plotly_white")
    fig_daily.update_layout(margin=standard_margin)

    # Cumulative total events
    dff_cum = dff.set_index('date').resample('D').size().reset_index(name='count')
    dff_cum['cumulative'] = dff_cum['count'].cumsum()
    fig_cumulative = px.line(dff_cum, x='date', y='cumulative', height=250, template="plotly_white")
    fig_cumulative.update_layout(margin=standard_margin)

    # Daily participant count
    dff_participants = dff.set_index('date').resample('D')['size_mean'].sum().reset_index(name='participants')
    fig_daily_participant_graph = px.bar(dff_participants, x='date', y='participants', height=250, template="plotly_white")
    fig_daily_participant_graph.update_layout(margin=standard_margin)

    # Ensure location_label is present for details panel
    def best_location(row):
        loc = str(row.get('location', 'Unknown')).strip()
        if loc and loc.lower() != 'nan':
            return loc
        loc2 = str(row.get('locality', 'Unknown')).strip()
        if loc2 and loc2.lower() != 'nan':
            return loc2
        state = row.get('state', 'Unknown')
        date = row['date'].date() if pd.notnull(row.get('date')) else 'Unknown'
        return f\"{state}, {date}\"

    if 'location_label' not in dff.columns:
        dff['location_label'] = dff.apply(best_location, axis=1)

    # Percent of Population KPI (state vs US)
    from state_pop import STATE_POP  # unchanged import

    selected_states = state_filter if isinstance(state_filter, list) else [state_filter]
    if size_filter == "no":
        percent_us_pop_kpi = "-"
    else:
        if len(selected_states) == 1 and selected_states[0] in STATE_POP:
            state_code = selected_states[0]
            population_base = STATE_POP[state_code]
            pop_label = f"% of {state_code} Population"
        else:
            population_base = US_POPULATION
            pop_label = "% of US Population"

        if total_participants > 0 and population_base > 0:
            percent_val = 100 * total_participants / population_base
            percent_us_pop_kpi = html.Div([
                html.Div(f\"{percent_val:.2f}%\", className=\"kpi-value\"),
                html.Div(\"👥\", className=\"kpi-emoji\"),
                html.Div(pop_label, className=\"kpi-label\")
            ])
        else:
            percent_us_pop_kpi = html.Div([
                html.Div(\"-\", className=\"kpi-value\"),
                html.Div(\"👥\", className=\"kpi-emoji\"),
                html.Div(pop_label, className=\"kpi-label\")
            ])

    # Footer message + link
    LINK_BUTTON_CLASS = "link-button"
    missing_count = dff['size_mean'].isna().sum()
    missing_pct = 100 * missing_count / total_events if total_events > 0 else 0

    if size_filter == "no":
        msg = html.Span(["There are ", html.Span(f\"{total_events:,}\", className=\"accent-blue bold\"),
                         " events in the database for your filter selection missing participant counts."])
        link_visible = False
    elif size_filter == "has":
        dff_no = filter_data(start_date, end_date, "no", org_search, state_filter, city_filter, any_outcomes_filter)
        missing_total = len(dff_no)
        if missing_total == 0:
            msg = html.Span(["There are ", html.Span(f\"{total_events:,}\", className=\"accent-blue bold\"),
                             " events in the database for your filter selections. All have participant counts."])
            link_visible = False
        else:
            msg = html.Span(["There are ", html.Span(f\"{total_events:,}\", className=\"accent-blue bold\"),
                             " events in the database for your filter selections, but ",
                             html.Span(f\"{missing_total:,}\", className=\"accent-red bold\"),
                             " additional events are missing participant counts. Participant counts are vital for tracking protest size and progress over time."])
            link_visible = True
    else:
        if missing_count > 0:
            msg = html.Span(["There are ", html.Span(f\"{total_events:,}\", className=\"accent-blue bold\"),
                             " events in the database for your filter selections, but ",
                             html.Span(f\"{missing_pct:.1f}%\", className=\"accent-red bold\"),
                             " (", html.Span(f\"{missing_count:,}\", className=\"accent-red bold\"), ") of those are missing vital information needed to track protest size and progress over time."])
            link_visible = True
        else:
            msg = html.Span(["There are ", html.Span(f\"{total_events:,}\", className=\"accent-blue bold\"),
                             " events in the database for your filter selections. All have participant counts."])
            link_visible = False

    missing_button = html.Span(
        "Click here to see only events missing participant counts",
        id='missing-link',
        n_clicks=0,
        className=LINK_BUTTON_CLASS,
        style={'display': 'inline' if link_visible and size_filter != "no" else 'none'}
    )

    footer_message = html.Div([msg, missing_button], className="footer-message-inner")

    if size_filter == "no":
        largest_event_kpi = "-"
        mean_size_kpi = "-"
        largest_day_kpi = "-"
        total_participants_kpi = "-"
        no_injuries_kpi = "-"
        no_arrests_kpi = "-"
        no_damage_kpi = "-"
        percent_us_pop_kpi = "-"

    return (
        fig_map,
        fig_momentum,
        fig_daily,
        dff.to_json(date_format='iso', orient='split'),
        fig_cumulative,
        fig_daily_participant_graph,
        total_events_kpi,
        largest_event_kpi,
        mean_size_kpi,
        largest_day_kpi,
        total_participants_kpi,
        no_injuries_kpi,
        no_arrests_kpi,
        no_damage_kpi,
        percent_us_pop_kpi,
        threshold_text,
        footer_message
    )

# -------------------------
# Event details panel
# -------------------------
@app.callback(
    Output('event-details-panel', 'children'),
    Input('map-graph', 'clickData'),
    State('filtered-data', 'data')
)
def update_event_details(click_data, filtered_data):
    box_class = "details-box"
    if not click_data or not filtered_data:
        return html.Div("Click a map marker to see event details.", className=f\"{box_class} hint\")

    try:
        dff = pd.read_json(io.StringIO(filtered_data), orient='split')
        point = click_data['points'][0]
        location_label = point.get('text')
        if not location_label:
            return html.Div("No details available for this location.", className=box_class)

        def norm(x): return str(x).strip().lower() if pd.notnull(x) else ''
        norm_label = norm(location_label)
        dff['__norm_label'] = dff['location_label'].apply(norm)
        location_events = dff[dff['__norm_label'] == norm_label]

        if location_events.empty:
            location_events = dff[dff['__norm_label'].str.contains(re.escape(norm_label))]
            if location_events.empty:
                return html.Div("No event details found for this marker.", className=box_class)

        always_fields = [
            ('Title', 'title'),
            ('Date', 'date'),
            ('Location', 'location'),
            ('City', 'resolved_locality'),
            ('State', 'resolved_state'),
            ('County', 'resolved_county'),
            ('Organizations', 'organizations'),
            ('Participants', 'size_mean'),
            ('Targets', 'targets'),
            ('Claims', 'claims_summary')
        ]
        optional_fields = [
            ('Notables', 'notables'),
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
            for label, col in always_fields:
                value = event.get(col, 'Unknown')
                if pd.isnull(value) or (isinstance(value, str) and (not value.strip() or value.strip().lower() == 'nan')):
                    value = 'Unknown'
                if col == 'date' and pd.notnull(value) and value != 'Unknown':
                    try:
                        value = pd.to_datetime(value).strftime('%Y-%m-%d')
                    except Exception:
                        value = 'Unknown'
                event_detail.append(html.P(f\"{label}: {value}\", className=\"detail-line\"))

            for label, col in optional_fields:
                value = event.get(col, 'Unknown')
                if pd.isnull(value) or (isinstance(value, str) and (not value.strip() or value.strip().lower() == 'nan')):
                    continue
                event_detail.append(html.P(f\"{label}: {value}\", className=\"detail-line\"))

            title = event.get('title', 'Unknown')
            date = event.get('date', 'Unknown')
            if pd.notnull(date) and date != 'Unknown':
                try:
                    date = pd.to_datetime(date).strftime('%Y-%m-%d')
                except Exception:
                    date = 'Unknown'
            header = f\"{title} - {date}\"

            details.append(html.Details([
                html.Summary(header, className=\"detail-header\"),
                html.Div(event_detail, className=\"detail-body\")
            ], open=True, className=\"detail-item\"))

        return html.Div(details, className=box_class)

    except Exception as e:
        return html.Div(f\"An error occurred while loading event details: {str(e)}\", className=f\"{box_class} error\")

# -------------------------
# Table + download
# -------------------------
@app.callback(
    [Output('filtered-table', 'data'),
     Output('filtered-table', 'columns')],
    Input('filtered-data', 'data')
)
def update_table(data_json):
    if not data_json:
        return [], []
    try:
        dff = pd.read_json(io.StringIO(data_json), orient='split')
        columns = [{'name': col, 'id': col} for col in dff.columns]
        return dff.to_dict('records'), columns
    except Exception:
        return [], []

@app.callback(
    Output("download-data", "data"),
    Input("download-btn", "n_clicks"),
    State("filtered-data", "data"),
    State("download-choice", "value"),
    prevent_initial_call=True
)
def download_filtered_table(n_clicks, filtered_data, download_choice):
    if not filtered_data:
        return no_update
    dff = pd.read_json(io.StringIO(filtered_data), orient='split')
    if download_choice == "full":
        return dcc.send_data_frame(df.to_csv, filename="full_dataset.csv")
    return dcc.send_data_frame(dff.to_csv, filename="filtered_dataset.csv")

# -------------------------
# Panel toggle
# -------------------------
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

# -------------------------
# City dropdown options
# -------------------------
@app.callback(
    Output('city-filter', 'options'),
    Output('city-filter', 'value'),
    Input('state-filter', 'value'),
    State('city-filter', 'value')
)
def update_city_options(selected_states, selected_cities):
    if not selected_states:
        return [], []
    filtered = df[df['state'].isin(selected_states)]
    cities = sorted(filtered['resolved_locality'].dropna().unique())
    options = [{'label': c, 'value': c} for c in cities]
    new_selected = [c for c in (selected_cities or []) if c in cities]
    return options, new_selected

# -------------------------
# Footer link -> filter + switch tab
# -------------------------
@app.callback(
    Output("size-filter", "value"),
    Output("dashboard-tabs", "value"),
    Output("missing-link", "style"),
    Input("missing-link", "n_clicks"),
    State("size-filter", "value"),
    prevent_initial_call=True
)
def click_missing(n_clicks, current_filter):
    if ctx.triggered_id == "missing-link" and n_clicks:
        return "no", "table", {'display': 'none'}
    return no_update, no_update, no_update

# -------------------------
# Entrypoint
# -------------------------
if __name__ == '__main__':
    app.run(debug=True)
