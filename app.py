import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx
import csv
import os
import random


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
    html.Div(id='page-content')
])

# PAGE 1 

dashboard_layout = html.Div([
    # Sidebar
    html.Div([
        # dcc.Link('📥 Submit New Protest Event', href='/submit', style={'display': 'block', 'margin': '10px 0 20px 0', 'fontWeight': 'bold'}),
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
            value='50501',
            debounce=True,
            style={'width': '100%', 'marginBottom': '5px'}
        ),
        html.Div("↩ Press Enter to update", style={'fontSize': '0.8em', 'color': '#666', 'marginBottom': '15px'}),

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
        dcc.Download(id="download-data")
    ], style={
        'width': '280px',
        'padding': '24px',
        'boxSizing': 'border-box',
        'backgroundColor': '#f9f9f9',
        'borderRight': '1px solid #ccc',
        'flexShrink': '0',
        'flexGrow': '1',
        'height': '100vh',
        'overflowY': 'auto',
        'overflow': 'visible',  # <-- Add this line
        'boxShadow': '2px 0 8px rgba(0,0,0,0.07)',
        'borderRadius': '0 12px 12px 0',
        'position': 'relative',
        'zIndex': 1050
    }),

    # Main content
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

def jitter_coords(df, lat_col='lat', lon_col='lon', jitter_amount=0.08):
    """Add random jitter to duplicate lat/lon pairs in a DataFrame."""
    df = df.copy().reset_index(drop=True)  # Ensure integer index
    coords = df[[lat_col, lon_col]].astype(str).agg('_'.join, axis=1)
    counts = coords.value_counts()
    dup_coords = counts[counts > 1].index

    for coord in dup_coords:
        idxs = df.index[coords == coord].tolist()
        for offset, i in enumerate(idxs):
            if offset == 0:
                continue  # Leave the first as is
            angle = random.uniform(0, 2 * np.pi)
            radius = random.uniform(0.0005, jitter_amount)
            df.at[i, lat_col] += np.cos(angle) * radius
            df.at[i, lon_col] += np.sin(angle) * radius
    return df

@app.callback(
    [Output('map-graph', 'figure'),
     Output('momentum-graph', 'figure'),
     Output('daily-graph', 'figure'),
     Output('kpi-cards', 'children')],
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('size-filter', 'value'),
     Input('trump-filter', 'value'),
     Input('org-search', 'value')]
)
def update_all(start_date, end_date, size_filter, trump_filter, org_search):
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
        dff = dff[dff['organizations'].str.contains(org_search.lower(), na=False)]

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
    dff_map = jitter_coords(dff_map, lat_col='lat', lon_col='lon', jitter_amount=0.03)  # Add jitter here

    has_size = dff_map[dff_map['size_mean'].notna()]
    no_size = dff_map[dff_map['size_mean'].isna()]

    fig_map = go.Figure()

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
            text=has_size['title'],
            hovertext=has_size['location'] + "<br>" +
                      has_size['date'].astype(str) + "<br>" +
                      "Size: " + has_size['size_mean'].astype(str) + "<br>" +
                      has_size['notes'].fillna(""),
            name="Has Size"
        ))

    if not no_size.empty:
        fig_map.add_trace(go.Scattermapbox(
            lat=no_size['lat'],
            lon=no_size['lon'],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=10,
                color='red',
                symbol='circle'
            ),
            text=no_size['title'],
            hovertext=no_size['location'] + "<br>" +
                      no_size['date'].astype(str) + "<br>" +
                      "Size: Unknown<br>" +
                      no_size['notes'].fillna(""),
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
    fig_daily = px.bar(dff_daily, x='date', y='count', title="Daily Event Count", height=270)
    fig_daily.update_layout(margin=dict(t=30, b=10, l=10, r=10))  # Shorter, less padding

    return fig_map, fig_momentum, fig_daily, kpis

@app.callback(
    Output("download-data", "data"),
    Input("download-btn", "n_clicks"),
    State("download-choice", "value"),
    State('date-range', 'start_date'),
    State('date-range', 'end_date'),
    State('size-filter', 'value'),
    State('trump-filter', 'value'),
    State('org-search', 'value'),
    prevent_initial_call=True
)
def download_csv(n_clicks, choice, start_date, end_date, size_filter, trump_filter, org_search):
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
            export_df = export_df[export_df['organizations'].str.contains(org_search.lower(), na=False)]

    return dcc.send_data_frame(export_df.to_csv, "protest_data.csv", index=False)

@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    return dashboard_layout

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8050, debug=True)
