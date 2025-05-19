import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx
import csv
import os
import random
import re
from dash.dependencies import Input, Output, State
from dash import no_update


file_path = "ccc-phase3-public.csv"
US_POPULATION = 340_100_000

# Load data
df = pd.read_csv(file_path, encoding='latin1', low_memory=False)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['size_mean'] = pd.to_numeric(df['size_mean'], errors='coerce')
df['participants_numeric'] = df['size_mean']
df['targets'] = df['targets'].astype(str).str.lower()
df['organizations'] = df['organizations'].astype(str).str.lower()

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.title = "Protest Dashboard"

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='filtered-data'),
    html.Div(id='page-content')
])

# PAGE 1 

# Sidebar content as separate components
filter_panel = html.Div([
    html.H2("Filters", style={'marginBottom': '20px'}),

    html.Label("Date Range"),
    dcc.DatePickerRange(
        id='date-range',
        start_date=df['date'].min(),
        end_date=df['date'].max(),
        display_format='YYYY-MM-DD',
        style={'marginBottom': '20px', 'width': '100%'}
    ),

    html.Label("Participant Size Filter"),
    dcc.RadioItems(
        id='size-filter',
        options=[
            {'label': 'Has participant size', 'value': 'has'},
            {'label': 'No participant size', 'value': 'no'},
            {'label': 'All events', 'value': 'all'}
        ],
        value='all',
        labelStyle={'display': 'block'},
        style={'marginBottom': '20px'}
    ),

    dcc.Checklist(
        id='trump-filter',
        options=[{'label': 'Only anti-Trump events', 'value': 'trump'}],
        value=[],
        style={'marginBottom': '20px'}
    ),

    html.Label("Organization Search"),
    dcc.Input(
        id='org-search',
        type='text',
        placeholder="Type organizations, separated by commas",
        style={'width': '100%', 'marginBottom': '5px'}
    ),
    html.Div("↩ Separate multiple organizations with commas", style={'fontSize': '0.8em', 'color': '#666', 'marginBottom': '15px'}),

    html.Label("State/Territory"),
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
        style={'marginBottom': '20px'}
    ),

    html.Label("Download Data"),
    dcc.Dropdown(
        id='download-choice',
        options=[
            {'label': 'Filtered View Only', 'value': 'filtered'},
            {'label': 'Full Dataset', 'value': 'full'}
        ],
        value='filtered',
        clearable=False,
        style={'marginBottom': '10px'}
    ),
    html.Button("Download CSV", id="download-btn", style={'marginBottom': '20px'}),
    dcc.Download(id="download-data"),
], id='filter-panel')

definitions_panel = html.Div([
    html.H3("Data Definitions & Sources", style={'marginTop': '20px'}),
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
], id='definitions-panel')

# Sidebar with toggle button and content container
sidebar = html.Div([
    html.Button(
        id='toggle-sidebar', 
        n_clicks=0, 
        children="Show Data Definitions & Sources",
        style={'marginBottom': '18px', 'width': '100%', 'fontWeight': 'bold'}
    ),
    html.Div(filter_panel, id='sidebar-content')
], style={
    'width': '280px',
    'padding': '24px',
    'boxSizing': 'border-box',
    'backgroundColor': '#f9f9f9',
    'borderRight': '1px solid #ccc',
    'flexShrink': '0',
    'flexGrow': '1',
    'overflowY': 'auto',
    'boxShadow': '2px 0 8px rgba(0,0,0,0.07)',
    'borderRadius': '0 12px 12px 0',
    'position': 'relative',
    'zIndex': 1050,
    'paddingBottom': '48px'
})

dashboard_layout = html.Div([
    sidebar,
    html.Div([
        html.Div(id='kpi-cards', style={
            'display': 'flex',
            'justifyContent': 'space-around',
            'margin': '24px 0 24px 0',
            'gap': '16px'
        }),

        html.Div([
            dcc.Graph(id='map-graph', style={'height': '320px', 'marginBottom': '24px', 'borderRadius': '10px', 'backgroundColor': '#fff', 'boxShadow': '0 2px 8px rgba(0,0,0,0.05)'}),
            dcc.Graph(id='momentum-graph', style={'height': '250px', 'marginBottom': '24px', 'borderRadius': '10px', 'backgroundColor': '#fff', 'boxShadow': '0 2px 8px rgba(0,0,0,0.05)'}),
            dcc.Graph(id='daily-graph', style={'height': '250px', 'borderRadius': '10px', 'backgroundColor': '#fff', 'boxShadow': '0 2px 8px rgba(0,0,0,0.05)'})
        ], style={'minWidth': '0'})
    ], style={
        'width': 'calc(100% - 280px)',
        'padding': '24px',
        'boxSizing': 'border-box',
        'flexGrow': '1',
        'overflow': 'auto',
        'backgroundColor': '#f5f6fa',
        'borderRadius': '12px 0 0 12px'
    })
], style={
    'display': 'flex',
    'flexDirection': 'row',
    'flexWrap': 'nowrap',
    'height': '100vh',
    'overflow': 'hidden',
    'fontFamily': 'Arial, sans-serif',
    'backgroundColor': '#e9ecef'
})

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

@app.callback(
    [Output('map-graph', 'figure'),
     Output('momentum-graph', 'figure'),
     Output('daily-graph', 'figure'),
     Output('kpi-cards', 'children'),
     Output('filtered-data', 'data')],  
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('size-filter', 'value'),
     Input('trump-filter', 'value'),
     Input('org-search', 'value'),
     Input('state-filter', 'value')]  \
)
def update_all(start_date, end_date, size_filter, trump_filter, org_search, state_filter):
    dff = df.copy()

    if start_date and end_date:
        dff = dff[(dff['date'] >= start_date) & (dff['date'] <= end_date)]
    if size_filter == 'has':
        dff = dff[dff['size_mean'].notna()]
    elif size_filter == 'no':
        dff = dff[dff['size_mean'].isna()]
    if 'trump' in trump_filter:
        dff = dff[dff['targets'].str.contains("trump", na=False)]
    if org_search:
        org_terms = [o.strip().lower() for o in org_search.split(',') if o.strip()]
        pattern = '|'.join([re.escape(org) for org in org_terms])
        if pattern:
            dff = dff[dff['organizations'].str.contains(pattern, na=False, regex=True)]
            dff = dff.drop_duplicates(subset='id' if 'id' in dff.columns else None)
    if state_filter:
        dff = dff[dff['state'].isin(state_filter)]

    total_events = len(dff)
    total_participants = dff['size_mean'].sum()
    # Use peak single-day turnout for percent_us_pop
    peak_day = dff.groupby('date')['size_mean'].sum().max()
    percent_us_pop = (peak_day / US_POPULATION) * 100 if peak_day else 0
    cumulative_total_events = len(dff)
    mean_size = dff['size_mean'].mean()


    kpis = [
        html.Div([
            html.H3(f"{total_events:,}", style={'color': 'orange'}),
            html.P("Total Events in Range")
        ], style={'width': '24%', 'textAlign': 'center', 'padding': '10px', 'borderRadius': '8px', 'backgroundColor': '#f0f0f0'}),

        html.Div([
            html.H3(f"{percent_us_pop:.4f}%", style={'color': 'green'}),
            html.P("Size Mean as % of US Population")
        ], style={'width': '24%', 'textAlign': 'center', 'padding': '10px', 'borderRadius': '8px', 'backgroundColor': '#f0f0f0'}),

        html.Div([
            html.H3(f"{mean_size:,.0f}", style={'color': 'purple'}),
            html.P("Mean Protest Size")
        ], style={
            'width': '24%',
            'textAlign': 'center',
            'padding': '10px',
            'borderRadius': '8px',
            'backgroundColor': '#f0f0f0'
        }),

        html.Div([
            html.H3(f"{cumulative_total_events:,}", style={'color': '#555'}),
            html.P("Cumulative Total Events")
        ], style={'width': '24%', 'textAlign': 'center', 'padding': '10px', 'borderRadius': '8px', 'backgroundColor': '#f0f0f0'})
    ]

    dff_map = dff.dropna(subset=['lat', 'lon']).copy()
    dff_map = jitter_coords(dff_map, lat_col='lat', lon_col='lon', jitter_amount=0.03)

    # Group by location (not coordinates), aggregate events at each site
    def aggregate_events(df):
        # Pre-format event title/date/org for all events
        df = df.copy()
        df['event_label'] = df.apply(
            lambda row: f"{row['title']} ({row['date'].date() if pd.notnull(row['date']) else ''})<br>Org: {row['organizations']}", axis=1
        )
        # Aggregate by location
        agg = df.groupby('location').agg(
            lat=('lat', 'first'),
            lon=('lon', 'first'),
            count=('title', 'size'),
            event_list=('event_label', lambda x: "<br><br>".join(x)),
            title=('title', lambda x: "; ".join(x.astype(str))),
            size_mean=('size_mean', lambda x: x.mean() if x.notna().any() else np.nan)
        ).reset_index()
        agg['text'] = agg.apply(lambda row: f"{row['location']} ({row['count']} event{'s' if row['count'] > 1 else ''})", axis=1)
        agg['hover'] = agg.apply(
            lambda row: (
                f"<b>{row['location']}</b><br>"
                f"Events at this site: {row['count']}<br><br>"
                f"<b>Events:</b><br>{row['event_list']}"
            ), axis=1
        )
        return agg

    agg_map = aggregate_events(dff_map)

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
            marker=go.scattermapbox.Marker(
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
            marker=go.scattermapbox.Marker(
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
    )  # Shorter, less padding

    dff_daily = dff.set_index('date').resample('D').size().reset_index(name='count')
    fig_daily = px.bar(
        dff_daily,
        x='date',
        y='count',
        title="Daily Event Count",
        height=270,
        template="plotly_white" 
    )
    fig_daily.update_layout(margin=dict(t=30, b=10, l=10, r=10))  # Shorter, less padding

    return fig_map, fig_momentum, fig_daily, kpis, dff.to_json(date_format='iso', orient='split')

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
    prevent_initial_call=True
)
def download_csv(n_clicks, choice, start_date, end_date, size_filter, trump_filter, org_search, state_filter):
    if choice == 'full':
        export_df = df.copy()
    else:
        export_df = df.copy()
        if start_date and end_date:
            export_df = export_df[(export_df['date'] >= start_date) & (export_df['date'] <= end_date)]
        if size_filter == 'has':
            export_df = export_df[export_df['size_mean'].notna()]
        elif size_filter == 'no':
            export_df = export_df[export_df['size_mean'].isna()]
        if 'trump' in trump_filter:
            export_df = export_df[export_df['targets'].str.contains("trump", na=False)]
        if org_search:
            org_terms = [o.strip().lower() for o in org_search.split(',') if o.strip()]
            pattern = '|'.join([re.escape(org) for org in org_terms])
            if pattern:
                export_df = export_df[export_df['organizations'].str.contains(pattern, na=False, regex=True)]
                export_df = export_df.drop_duplicates(subset='id' if 'id' in export_df.columns else None)
        if state_filter:
            export_df = export_df[export_df['state'].isin(state_filter)]

    return dcc.send_data_frame(export_df.to_csv, "protest_data.csv", index=False)

@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    return dashboard_layout

# Callback to toggle sidebar content
@app.callback(
    Output('sidebar-content', 'children'),
    Output('toggle-sidebar', 'children'),
    Input('toggle-sidebar', 'n_clicks'),
    prevent_initial_call=False
)
def toggle_sidebar_content(n_clicks):
    if n_clicks % 2 == 1:
        return definitions_panel, "Show Filters"
    else:
        return filter_panel, "Show Data Definitions & Sources"

html.Div(id='event-details-panel', style={'margin': '16px 0', 'padding': '12px', 'backgroundColor': '#fff', 'borderRadius': '8px', 'boxShadow': '0 2px 8px rgba(0,0,0,0.05)'})

@app.callback(
    Output('event-details-panel', 'children'),
    Input('map-graph', 'clickData'),
    State('filtered-data', 'data')
)
def show_event_details(clickData, filtered_json):
    if not clickData or 'points' not in clickData or not clickData['points']:
        return "Click a map marker to see event details for that location."
    if not filtered_json:
        return "No filtered data available."
    dff = pd.read_json(filtered_json, orient='split')
    point = clickData['points'][0]
    location = point['text'].split(' (')[0]
    events = dff[dff['location'] == location]
    if events.empty:
        return "No events found for this location."

    # List all events at this location with title, date, and organizations
    event_items = []
    for _, row in events.iterrows():
        event_items.append(
            html.Div([
                html.H4(row['title']),
                html.P(f"Date: {row['date'].date() if pd.notnull(row['date']) else ''}"),
                html.P(f"Organizations: {row['organizations']}"),
                html.Hr(style={'margin': '10px 0'})
            ], style={'marginBottom': '10px'})
        )

    return html.Div([
        html.H3(f"Events at {location} ({len(events)})"),
        html.Div(event_items)
    ])

@app.callback(
    Output('event-detail-content', 'children'),
    Input('event-dropdown', 'value'),
    State('map-graph', 'clickData'),
    State('filtered-data', 'data')
)
def update_event_detail(selected_idx, clickData, filtered_json):
    if not clickData or 'points' not in clickData or not clickData['points']:
        return ""
    if not filtered_json:
        return ""
    dff = pd.read_json(filtered_json, orient='split')
    location = clickData['points'][0]['text'].split(' (')[0]
    events = dff[dff['location'] == location]
    if selected_idx not in events.index:
        return ""
    row = events.loc[selected_idx]
    return html.Div([
        html.H4(row['title']),
        html.P(f"Date: {row['date'].date() if pd.notnull(row['date']) else ''}"),
        html.P(f"Size: {row['size_mean'] if pd.notnull(row['size_mean']) else 'Unknown'}"),
        html.P(f"Organizations: {row['organizations']}"),
        html.P(f"Claims: {row.get('claims_summary', '')}"),
        html.P(f"Notes: {row['notes'] if 'notes' in row else ''}")
    ])

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8050, debug=True)
