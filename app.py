import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, no_update, dash_table, ctx, callback_context
from flask_caching import Cache
import os
import random
import re
import io
from io import StringIO
import time
import datetime
import traceback
from dash.dependencies import ALL

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
    # Optimize: Convert string columns used in filters to category dtype for speed/memory
    df['targets'] = df['targets'].astype(str).str.lower().astype('category')
    df['organizations'] = df['organizations'].astype(str).str.lower().astype('category')
    df['state'] = df['state'].astype('category')
    if 'trump_stance' in df.columns:
        df['trump_stance'] = df['trump_stance'].astype(str).str.lower()

    # Ensure numeric columns are actually numeric for filtering
    for col in [
        'participant_injuries', 'police_injuries', 'arrests',
        'participant_deaths', 'police_deaths'
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # After loading df (right after reading CSV or Parquet), add this:
    if 'property_damage' in df.columns:
        df['property_damage_any'] = (
            df['property_damage'].notna() & (df['property_damage'].astype(str).str.strip() != "")
        ).astype(int)

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
standard_margin = dict(t=20, b=20, l=18, r=18)

# Define filter_panel and definitions_panel first
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
        style={
            'marginBottom': '4px',
            'width': '100%',
            'zIndex': 1200,
            'position': 'relative',
            'overflow': 'visible',
        },
        calendar_orientation='vertical',
        day_size=32,
        with_portal=True,
    ),
     dcc.Dropdown(
        id='national-days-dropdown',
        options=[
            {'label': 'February 5', 'value': '02-05'},
            {'label': 'February 17', 'value': '02-17'},
            {'label': 'March 4', 'value': '03-04'},
            {'label': 'April 5', 'value': '04-05'},
            {'label': 'April 19', 'value': '04-19'},
            {'label': 'May 1', 'value': '05-01'},
            # {'label': 'June 14', 'value': '06-14'}, //we'll add this with the next update this month
        ],
        placeholder="National Day of Action",
        clearable=True,
        style={'marginBottom': '12px', 'width': '100%'}
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
    html.Label("Participant Count Filter", style={'fontFamily': FONT_FAMILY}),
    dcc.RadioItems(
        id='size-filter',
        options=[
            {'label': 'Has participant count', 'value': 'has'},
            {'label': 'No participant count', 'value': 'no'},
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
    html.Label("City", style={'fontFamily': FONT_FAMILY}),
    dcc.Dropdown(
        id='city-filter',
        options=[],  # Start empty; will be populated by callback
        value=[],
        multi=True,
        placeholder="Select a state first for cities",        clearable=True,
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
            "Questions or feedback? Email ",
            html.A("metricscalculations@gmail.com", href="mailto:metricscalculations@gmail.com"),
            "."
        ], style={'marginBottom': '10px', 'color': PRIMARY_BLUE, 'fontWeight': 'bold'}),
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
                html.B("Participant Count: "),
                "The 'size_mean' field is an average of the upper and lower range estimates of crowd size, as reported. "
                "This provides a standardized estimate of participant size for each event. Some events may have missing or uncertain size estimates."
            ]),
            html.Li([
                html.B("Momentum of Dissent: "),
                "For each day, the sum of estimated participants is multiplied by the number of events that day. "
                "The 'Momentum of Dissent' shown in the dashboard is the sum of these daily values over the most recent 7 days (a rolling 7-day sum). "
                "This highlights periods of sustained, high-volume protest activity. "
                "The concept and approach for 'momentum' as a protest metric is inspired by the methodology described in: ",
                html.A(
                    "Chenoweth, E., Perkoski, E., & Kang, S. (2017). State Repression and Nonviolent Resistance. Research in Social Movements, Conflicts and Change, 41, 85–117.",
                    href="https://bura.brunel.ac.uk/bitstream/2438/19075/1/FullText.pdf",
                    target="_blank"
                ),
                ". In that work, 'movement momentum' is used to capture the intensity and persistence of protest activity over time."
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
                html.B("Largest Day of Action as % of US population: "),
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

# Define the get_sidebar function
def get_sidebar(is_open):
    sidebar_style = {
        'width': '320px' if is_open else '0px',
        'minWidth': '0px',
        'padding': '18px 16px 48px 16px' if is_open else '0px',
        'boxSizing': 'border-box',
        'background': 'linear-gradient(135deg, #f5f6fa, #e9ecef)',
        'borderRight': '1px solid #ccc',
        'flexShrink': '0',
        'flexGrow': '0',
        'overflowY': 'auto',
        'overflowX': 'hidden',
        'height': '100vh',
        'boxShadow': '2px 0 8px rgba(0,0,0,0.07)',
        'borderRadius': '0 12px 12px 0',
        'position': 'relative',
        'zIndex': 1050,
        'fontFamily': FONT_FAMILY,
        'transition': 'width 0.3s cubic-bezier(.4,2,.6,1), padding 0.3s cubic-bezier(.4,2,.6,1)'
    }
    
    toggle_icon = "❮" if is_open else "❯"
    toggle_tab = html.Div(
        toggle_icon,
        id='sidebar-toggle-tab',
        n_clicks=0,
        style={
            'position': 'fixed',
            'top': '50%',
            'left': '320px' if is_open else '0px',
            'transform': 'translateY(-50%)',
            'width': '32px',
            'height': '64px',
            'background': PRIMARY_BLUE,
            'color': PRIMARY_WHITE,
            'borderRadius': '0 16px 16px 0',
            'display': 'flex',
            'alignItems': 'center',
            'justifyContent': 'center',
            'cursor': 'pointer',
            'boxShadow': '2px 0 8px rgba(36,76,196,0.08)',
            'fontSize': '1.5em',
            'zIndex': 2000,
            'border': f'2px solid {PRIMARY_BLUE}',
            'borderLeft': 'none',
            'transition': 'left 0.3s cubic-bezier(.4,2,.6,1)'
        }
    )
    content = html.Div([
        html.Div(filter_panel, id='filter-panel-container', style={
            'display': 'block',
            'visibility': 'visible' if is_open else 'hidden',
            'height': 'auto' if is_open else '0',
            'overflow': 'hidden'
        }),
        html.Div(definitions_panel, id='definitions-panel-container', style={
            'display': 'none'
        })
    ], id='sidebar-content', style={
        'width': '100%',
        'transition': 'all 0.3s ease-in-out'
    })
    bottom_btn = html.Button(
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
            'zIndex': 1100,
            'display': 'block' if is_open else 'none'
        },
        className="hover-button"
    )
    return html.Div([
        html.Div([content, bottom_btn], id='sidebar', style=sidebar_style),
        toggle_tab
    ], style={'display': 'flex', 'flexDirection': 'row'})

sidebar = get_sidebar(is_open=True)

# Define app.layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='filtered-data'),
    dcc.Store(id='sidebar-open', data=True),
    dcc.Store(id='show-missing-link-store', data=0),
    dcc.Store(id='three-point-five-check'),
    html.Div(id='sidebar-dynamic', children=get_sidebar(is_open=True)),
    html.Div(id='main-content', children=[
        html.Div(
            "Anti-Trump Events - 2025",
            style={
                'fontFamily': FONT_FAMILY,
                'fontWeight': 'bold',
                'fontSize': '2rem',
                'color': PRIMARY_BLUE,
                'textAlign': 'center',
                'marginBottom': '10px',
                'marginTop': '0px'
            }
        ),
        # KPIs: Always visible, two rows, red/blue checkerboarded
        html.Div([
            # First row
            html.Div([
                html.Div(id='total-events-kpi', style={
                    'flex': '1', 'textAlign': 'center', 'padding': '4px', 'borderRadius': '8px',
                    'backgroundColor': PRIMARY_BLUE, 'color': PRIMARY_WHITE, 'fontWeight': 'bold', 'margin': '0 2px'
                }),
                html.Div(id='mean-size-kpi', style={
                    'flex': '1', 'textAlign': 'center', 'padding': '4px', 'borderRadius': '8px',
                    'backgroundColor': PRIMARY_RED, 'color': PRIMARY_WHITE, 'fontWeight': 'bold', 'margin': '0 2px'
                }),
                html.Div(id='no-injuries-kpi', style={
                    'flex': '1', 'textAlign': 'center', 'padding': '4px', 'borderRadius': '8px',
                    'backgroundColor': PRIMARY_BLUE, 'color': PRIMARY_WHITE, 'fontWeight': 'bold', 'margin': '0 2px'
                }),
                html.Div(id='no-arrests-kpi', style={
                    'flex': '1', 'textAlign': 'center', 'padding': '4px', 'borderRadius': '8px',
                    'backgroundColor': PRIMARY_RED, 'color': PRIMARY_WHITE, 'fontWeight': 'bold', 'margin': '0 2px'
                }),
                html.Div(id='no-damage-kpi', style={
                    'flex': '1', 'textAlign': 'center', 'padding': '4px', 'borderRadius': '8px',
                    'backgroundColor': PRIMARY_BLUE, 'color': PRIMARY_WHITE, 'fontWeight': 'bold', 'margin': '0 2px'
                }),
            ], style={'display': 'flex', 'gap': '3px', 'marginBottom': '2px'}),
            # Second row
            html.Div([
                html.Div(id='total-participants-kpi', style={
                    'flex': '1', 'textAlign': 'center', 'padding': '4px', 'borderRadius': '8px',
                    'backgroundColor': PRIMARY_BLUE, 'color': PRIMARY_WHITE, 'fontWeight': 'bold', 'margin': '0 2px'
                }),
                html.Div(id='largest-event-kpi', style={
                    'flex': '1', 'textAlign': 'center', 'padding': '4px', 'borderRadius': '8px',
                    'backgroundColor': PRIMARY_RED, 'color': PRIMARY_WHITE, 'fontWeight': 'bold', 'margin': '0 2px'
                }),
                html.Div(id='largest-day-kpi', style={
                    'flex': '1', 'textAlign': 'center', 'padding': '4px', 'borderRadius': '8px',
                    'backgroundColor': PRIMARY_RED, 'color': PRIMARY_WHITE, 'fontWeight': 'bold', 'margin': '0 2px'
                }),
                html.Div(id='percent-us-pop-kpi', style={
                    'flex': '1', 'textAlign': 'center', 'padding': '4px', 'borderRadius': '8px',
                    'backgroundColor': PRIMARY_BLUE, 'color': PRIMARY_WHITE, 'fontWeight': 'bold', 'margin': '0 2px'
                }),
            ], style={'display': 'flex', 'gap': '3px', 'marginBottom': '4px'}),
        ], style={'marginBottom': '4px'}),
        # 3.5% Threshold Row (move here, after KPIs)
        html.Div(
            id='threshold-row',
            style={
                'display': 'flex',
                'justifyContent': 'center',
                'alignItems': 'center',
                'gap': '12px', 
                'marginBottom': '8px',
                'marginTop': '0px',
                'fontFamily': FONT_FAMILY,
                'fontSize': '1.05rem',
                'border': f'2px solid {PRIMARY_BLUE}',
                'borderRadius': '10px',
                'boxShadow': '0 2px 8px rgba(36,76,196,0.07)',
                'padding': '7px 16px',
                'background': '#f8faff',
            }
        ),
        # Tabs and tabbed content (table last)
        html.Div(
            dcc.Tabs(
                id='dashboard-tabs',
                value='map',
                children=[
                dcc.Tab(label='Map', value='map', children=[
                    dcc.Graph(
                        id='map-graph',
                        config={
                            'displayModeBar': True,
                            'modeBarButtonsToRemove': ['select2d', 'lasso2d']
                        }
                    ),
                    html.Div([
                        html.Div(id='missing-count-message-text-map', style={
                            'fontSize': '0.95em',
                            'margin': '6px 0 8px 0',
                            'textAlign': 'center',
                            'color': '#111'
                        }),
                        html.Button(
                            "Click here to see the events missing participant counts.",
                            id="show-missing-link-map",
                            style={
                                'background': 'none',
                                'border': 'none',
                                'color': PRIMARY_BLUE,
                                'textDecoration': 'underline',
                                'cursor': 'pointer',
                                'padding': 0,
                                'font': 'inherit',
                                'display': 'none',
                            }
                        ),
                        html.Div(id='event-details-panel', style={'marginTop': '12px'})
                    ], id='missing-count-message-map', style={'textAlign': 'center'}),

                ]),
                dcc.Tab(label='Graphs', value='graphs', children=[
                    html.Div([
                        html.Div("Momentum of Dissent", style={
                            'fontWeight': 'bold',
                            'fontSize': '1em',
                            'marginBottom': '14px',
                            'textAlign': 'center',
                            'lineHeight': '1.1',
                            'paddingBottom': '0px',
                        }),
                        dcc.Graph(
                            id='momentum-graph',
                            config={
                                'displayModeBar': True,
                                'modeBarButtonsToRemove': ['select2d', 'lasso2d']
                            },
                            style={'marginTop': '0px', 'paddingTop': '0px'}
                        )
                    ], style={'marginBottom': '36px', 'paddingBottom': '0px'}),
                    html.Div([
                        html.Div("Daily Event Count", style={
                            'fontWeight': 'bold',
                            'fontSize': '1em',
                            'marginBottom': '14px',
                            'textAlign': 'center',
                            'lineHeight': '1.1',
                            'paddingBottom': '0px',
                        }),
                        dcc.Graph(
                            id='daily-graph',
                            config={
                                'displayModeBar': True,
                                'modeBarButtonsToRemove': ['select2d', 'lasso2d']
                            },
                            style={'marginTop': '0px', 'paddingTop': '0px'}
                        )
                    ], style={'marginBottom': '36px', 'paddingBottom': '0px'}),
                    html.Div([
                        html.Div("Cumulative Total Events", style={
                            'fontWeight': 'bold',
                            'fontSize': '1em',
                            'marginBottom': '14px',
                            'textAlign': 'center',
                            'lineHeight': '1.1',
                            'paddingBottom': '0px',
                        }),
                        dcc.Graph(
                            id='cumulative-graph',
                            config={
                                'displayModeBar': True,
                                'modeBarButtonsToRemove': ['select2d', 'lasso2d']
                            },
                            style={'marginTop': '0px', 'paddingTop': '0px'}
                        )
                    ], style={'marginBottom': '36px', 'paddingBottom': '0px'}),
                    html.Div([
                        html.Div("Daily Participant Count", style={
                            'fontWeight': 'bold',
                            'fontSize': '1em',
                            'marginBottom': '14px',
                            'textAlign': 'center',
                            'lineHeight': '1.1',
                            'paddingBottom': '0px',
                        }),
                        dcc.Graph(
                            id='daily-participant-graph',
                            config={
                                'displayModeBar': True,
                                'modeBarButtonsToRemove': ['select2d', 'lasso2d']
                            },
                            style={'marginTop': '0px', 'paddingTop': '0px'}
                        )
                    ], style={'marginBottom': '36px', 'paddingBottom': '0px'}),
                    html.Div([
                        html.Div(id='missing-count-message-text-graphs', style={
                            'fontSize': '0.95em',
                            'margin': '6px 0 8px 0',
                            'textAlign': 'center',
                            'color': '#111'
                        }),
                        html.Button(
                            "Click here to see the events missing participant counts.",
                            id="show-missing-link-graphs",
                            style={
                                'background': 'none',
                                'border': 'none',
                                'color': PRIMARY_BLUE,
                                'textDecoration': 'underline',
                                'cursor': 'pointer',
                                'padding': 0,
                                'font': 'inherit',
                                'display': 'none',
                            }
                        )
                    ], id='missing-count-message-graphs', style={'textAlign': 'center'}),
                ]),
                dcc.Tab(label='Table', value='table', children=[
                    dash_table.DataTable(
                        id='filtered-table',
                        columns=[],
                        style_table={
                            'overflowY': 'auto',
                            'maxHeight': '400px',
                            'overflowX': 'auto',
                            'width': '100%',
                            'minWidth': '100%'
                        },
                        style_cell={
                            'textAlign': 'left',
                            'padding': '7px',
                            'fontFamily': FONT_FAMILY,
                            'fontSize': '13px'
                        },
                        style_header={
                            'fontWeight': 'bold',
                            'backgroundColor': PRIMARY_BLUE,
                            'color': PRIMARY_WHITE
                        },
                        virtualization=True,
                        fixed_rows={'headers': True}
                    ),
                    html.Div([
                        html.Div(id='missing-count-message-text-table', style={
                            'fontSize': '0.95em',
                            'margin': '6px 0 8px 0',
                            'textAlign': 'center',
                            'color': '#111'
                        }),
                        html.Button(
                            "Click here to see the events missing participant counts.",
                            id="show-missing-link-table",
                            style={
                                'background': 'none',
                                'border': 'none',
                                'color': PRIMARY_BLUE,
                                'textDecoration': 'underline',
                                'cursor': 'pointer',
                                'padding': 0,
                                'font': 'inherit',
                                'display': 'none',
                            }
                        )
                    ], id='missing-count-message-table', style={'textAlign': 'center'}),
                ])
            ],
            style={
                'marginBottom': '12px',
                'height': '40px',
                'minHeight': '38px',
                'fontSize': '1em',
                'paddingBottom': '10px',
                'marginTop': '0px',
                'display': 'flex',
                'alignItems': 'flex-start',
            },
            className='dashboard-tabs-custom'
        ),
        style={'marginBottom': '50px'}
        )
    ], style={
        'width': 'calc(100% - 320px)',
        'padding': '32px',
        'boxSizing': 'border-box',
        'flexGrow': '1',
        'overflow': 'auto',
        'backgroundColor': PRIMARY_WHITE,
        'borderRadius': '12px 0 0 12px',
        'fontFamily': FONT_FAMILY,
        'transition': 'width 0.3s cubic-bezier(.4,2,.6,1)',
        'marginTop': '18px'
    })
], style={
    'display': 'flex',
    'flexDirection': 'row',
    'flexWrap': 'nowrap',
    'height': '100vh',
    'overflow': 'hidden',
    'fontFamily': FONT_FAMILY,
    'backgroundColor': PRIMARY_WHITE
})

# Sidebar with collapsible toggle tab
def get_sidebar(is_open):
    sidebar_style = {
        'width': '320px' if is_open else '0px',
        'minWidth': '0px',
        'padding': '18px 16px 48px 16px' if is_open else '0px',
        'boxSizing': 'border-box',
        'background': 'linear-gradient(135deg, #f5f6fa, #e9ecef)',
        'borderRight': '1px solid #ccc',
        'flexShrink': '0',
        'flexGrow': '0',
        'overflowY': 'auto',
        'overflowX': 'hidden',
        'height': '100vh',
        'boxShadow': '2px 0 8px rgba(0,0,0,0.07)',
        'borderRadius': '0 12px 12px 0',
        'position': 'relative',
        'zIndex': 1050,
        'fontFamily': FONT_FAMILY,
        'transition': 'width 0.3s cubic-bezier(.4,2,.6,1), padding 0.3s cubic-bezier(.4,2,.6,1)'
    }
    # Use more compatible arrows for mobile
    toggle_icon = "❮" if is_open else "❯"
    toggle_tab = html.Div(
        toggle_icon,
        id='sidebar-toggle-tab',
        n_clicks=0,
        style={
            'position': 'fixed',
            'top': '50%',
            'left': '320px' if is_open else '0px',
            'transform': 'translateY(-50%)',
            'width': '32px',
            'height': '64px',
            'background': PRIMARY_BLUE,
            'color': PRIMARY_WHITE,
            'borderRadius': '0 16px 16px 0',
            'display': 'flex',
            'alignItems': 'center',
            'justifyContent': 'center',
            'cursor': 'pointer',
            'boxShadow': '2px 0 8px rgba(36,76,196,0.08)',
            'fontSize': '1.5em',
            'zIndex': 2000,
            'border': f'2px solid {PRIMARY_BLUE}',
            'borderLeft': 'none',
            'transition': 'left 0.3s cubic-bezier(.4,2,.6,1)'
        }
    )
    # Both panels always present, only one visible
    content = html.Div([
        html.Div(filter_panel, id='filter-panel-container', style={
            'display': 'block',
            'visibility': 'visible' if is_open else 'hidden',
            'height': 'auto' if is_open else '0',
            'overflow': 'hidden'
        }),
        html.Div(definitions_panel, id='definitions-panel-container', style={
            'display': 'none'
        })
    ], id='sidebar-content', style={
        'width': '100%',
        'transition': 'all 0.3s ease-in-out'
    })
    bottom_btn = html.Button(
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
            'zIndex': 1100,
            'display': 'block' if is_open else 'none'
        },
        className="hover-button"
    )
    return html.Div([
        html.Div([content, bottom_btn], id='sidebar', style=sidebar_style),
        toggle_tab
    ], style={'display': 'flex', 'flexDirection': 'row'})

# Callbacks for sidebar toggle and resizing main content

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
    # Adjust main content width based on sidebar state
    main_style = {
        'width': 'calc(100% - 320px)' if is_open else '100%',
        'padding': '32px',
        'boxSizing': 'border-box',
        'flexGrow': '1',
        'overflow': 'auto',
        'backgroundColor': PRIMARY_WHITE,
        'borderRadius': '12px 0 0 12px',
        'fontFamily': FONT_FAMILY,
        'transition': 'width 0.3s cubic-bezier(.4,2,.6,1)'
    }
    return get_sidebar(is_open), main_style

def jitter_coords(df, lat_col='lat', lon_col='lon', jitter_amount=0.05):
    """
    For duplicate lat/lon pairs, arrange all but the first equidistantly in a circle around the main point.
    The first event stays at the center.
    """
    df = df.copy().reset_index(drop=True)
    coords = df[[lat_col, lon_col]].round(5).astype(str).agg('_'.join, axis=1)
    counts = coords.value_counts()
    dup_coords = counts[counts > 1].index

    for coord in dup_coords:
        idxs = df.index[coords == coord].tolist()
        n = len(idxs)
        if n <= 1:
            continue
        center_lat = float(df.at[idxs[0], lat_col])
        center_lon = float(df.at[idxs[0], lon_col])
        # Place the rest in a circle around the center
        for k, i in enumerate(idxs[1:], 1):
            angle = 2 * np.pi * (k - 1) / (n - 1)
            radius = jitter_amount
            df.at[i, lat_col] = center_lat + np.cos(angle) * radius
            df.at[i, lon_col] = center_lon + np.sin(angle) * radius
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
    city_filter, any_outcomes_filter, national_day_value=None
):
    dff = df
    mask = pd.Series(True, index=dff.index)

    # Only convert columns that should be numeric for filtering
    for col in [
        'arrests', 'participant_injuries', 'police_injuries',
        'participant_deaths', 'police_deaths'
    ]:
        if col in dff.columns:
            dff[col] = pd.to_numeric(dff[col], errors='coerce')

    # DO NOT convert 'property_damage' to numeric here!
    # The boolean column 'property_damage_any' is already created at load time.

    # Date filter
    if start_date and end_date:
        mask &= (dff['date'] >= start_date) & (dff['date'] <= end_date)

    # National Day of Action filter
    if national_day_value:
        mask &= dff['date'].dt.strftime('%m-%d') == national_day_value

    # Size filter
    if size_filter == 'has':
        mask &= dff['size_mean'].notna()
    elif size_filter == 'no':
        mask &= dff['size_mean'].isna()
    # else 'all': do nothing

    # Organization search (case-insensitive, split by comma)
    if org_search and org_search.strip():
        orgs = [o.strip() for o in org_search.lower().split(',') if o.strip()]
        if orgs:
            pattern = '|'.join(map(re.escape, orgs))
            mask &= dff['organizations'].str.contains(pattern, na=False, regex=True)

    # State filter (only if not empty)
    if state_filter and len(state_filter) > 0:
        mask &= dff['state'].isin(state_filter)

    # City filter (only if not empty)
    if city_filter and len(city_filter) > 0:
        mask &= dff['resolved_locality'].isin(city_filter)

    # Outcomes filters
    for outcome in any_outcomes_filter:
        if outcome == 'arrests_any':
            mask &= dff['arrests'].notna() & (dff['arrests'] > 0)
        elif outcome == 'participant_injuries_any':
            mask &= dff['participant_injuries'].notna() & (dff['participant_injuries'] > 0)
        elif outcome == 'police_injuries_any':
            mask &= dff['police_injuries'].notna() & (dff['police_injuries'] > 0)
        elif outcome == 'property_damage_any':
            mask &= dff['property_damage_any'] == 1
        elif outcome == 'participant_deaths_any':
            mask &= dff['participant_deaths'].notna() & (dff['participant_deaths'] > 0)
        elif outcome == 'police_deaths_any':
            mask &= dff['police_deaths'].notna() & (dff['police_deaths'] > 0)

    return dff.loc[mask].copy()

@cache.memoize(timeout=120)
def aggregate_events_for_map(dff_map):
    df_map = dff_map

    # SPEEDUP: Vectorized best_location
    loc = df_map['location'].astype(str).str.strip() if 'location' in df_map.columns else pd.Series(['Unknown']*len(df_map), index=df_map.index)
    loc2 = df_map['locality'].astype(str).str.strip() if 'locality' in df_map.columns else pd.Series(['Unknown']*len(df_map), index=df_map.index)
    state = df_map['state'].astype(str) if 'state' in df_map.columns else pd.Series(['Unknown']*len(df_map), index=df_map.index)
    date = df_map['date'].dt.date.astype(str) if 'date' in df_map.columns else pd.Series(['Unknown']*len(df_map), index=df_map.index)
    location_label = np.where((loc != '') & (loc.str.lower() != 'nan'), loc,
                        np.where((loc2 != '') & (loc2.str.lower() != 'nan'), loc2,
                        state + ', ' + date))
    if 'location_label' not in df_map.columns:
        df_map['location_label'] = location_label
        df_map['location_label'] = df_map['location_label'].replace('', 'Unknown').fillna('Unknown')
    if 'event_label' not in df_map.columns:
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
    Output('map-graph', 'figure'),
    Output('momentum-graph', 'figure'),
    Output('daily-graph', 'figure'),
    Output('filtered-data', 'data'),
    Output('cumulative-graph', 'figure'),
    Output('daily-participant-graph', 'figure'),
    Output('total-events-kpi', 'children'),
    Output('mean-size-kpi', 'children'),
    Output('no-injuries-kpi', 'children'),
    Output('no-arrests-kpi', 'children'),
    Output('no-damage-kpi', 'children'),
    Output('total-participants-kpi', 'children'),
    Output('largest-event-kpi', 'children'),
    Output('largest-day-kpi', 'children'),
    Output('percent-us-pop-kpi', 'children'),
    Output('three-point-five-check', 'data'),  # New output for 3.5% check
    [
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date'),
        Input('size-filter', 'value'),
        Input('org-search', 'value'),
        Input('state-filter', 'value'),
        Input('city-filter', 'value'),
        Input('any-outcomes-filter', 'value'),
        Input('national-days-dropdown', 'value')
    ]
)
def update_all(
    start_date, end_date, size_filter, org_search, state_filter,
    city_filter, any_outcomes_filter, national_day_value=None
):


    try:
        # Filter data
        dff = filter_data(
            start_date, end_date, size_filter, org_search, state_filter,
            city_filter, any_outcomes_filter, national_day_value
        )


        # Defensive: Ensure required columns exist before KPI calculations
        required_cols = [
            'lat', 'lon', 'participant_injuries', 'arrests', 'property_damage_any',
            'participants_numeric', 'size_mean', 'date'
        ]
        missing_cols = [col for col in required_cols if col not in dff.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # SPEEDUP: Limit columns in dcc.Store (filtered-data) 
        # All columns except source_2 to source_30, keep only source_1
        table_columns = [col for col in dff.columns if not (col.startswith('source_') and col != 'source_1')]
        dff_for_store = dff[table_columns].copy()

        # Defensive: Ensure 'lat' and 'lon' columns are not all missing
        if dff['lat'].isnull().all() or dff['lon'].isnull().all():
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text="No matching data available for the selected filters.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=22, color="red"),
                align="center"
            )
            empty_fig.update_layout(
                mapbox_style="carto-positron",
                mapbox_zoom=3,
                mapbox_center={"lat": 39.8283, "lon": -98.5795},
                margin=standard_margin,
                height=500,
                showlegend=False
            )
            # DataTable: always output a list of records, with NaN/None replaced by ""
            empty_json = []
            def dash_kpi(label, icon="—"):
                return html.Div([
                    html.Div("-", style={'fontSize': '1.35rem', 'fontWeight': '700'}),
                    html.Div(icon, style={'fontSize': '1.2rem', 'margin': '0'}),
                    html.Div(label, style={'fontSize': '0.85rem', 'margin': '0'})
                ], style={'marginBottom': '0'})
            return (
                empty_fig,  # map-graph
                empty_fig,  # momentum-graph
                empty_fig,  # daily-graph
                empty_json, # filtered-data (records)
                empty_fig,  # cumulative-graph
                empty_fig,  # daily-participant-graph
                dash_kpi("Total Events", "🗓️"),
                dash_kpi("Average Participants per Event", "📊"),
                dash_kpi("Events with No Injuries", "🚑"),
                dash_kpi("Events with No Arrests", "🚔"),
                dash_kpi("Events with No Property Damage", "🏚️"),
                dash_kpi("Total Participants", "🌟"),
                dash_kpi("Largest Single Event", "🥇"),
                dash_kpi("Largest Day of Action", "📅"),
                dash_kpi("Largest Day as of Action as % of Population", "🌎"),
                None  # three-point-five-check (empty)
            )

        # Special case: If filter is set to "has no participant count", only show Total Events, others as "-"
        if size_filter == "no":
            # Use the same dash_kpi as above for consistency
            dash_kpi = lambda label, icon="—": html.Div([
                html.Div("-", style={'fontSize': '1.35rem', 'fontWeight': '700'}),
                html.Div(icon, style={'fontSize': '1.2rem', 'margin': '0'}),
                html.Div(label, style={'fontSize': '0.85rem', 'margin': '0'})
            ], style={'marginBottom': '0'})
            total_events_kpi = html.Div([
                html.Div("Total Events", style={'fontSize': '0.85rem', 'marginBottom': '4px'}),
                html.Div("🗓️", style={'fontSize': '1.2rem', 'marginTop': '4px'}),
                html.Div(f"{len(dff):,}", style={'fontSize': '1.05rem', 'fontWeight': '700'}),
            ], style={'textAlign': 'center', 'padding': '7px', 'borderRadius': '10px', 'backgroundColor': PRIMARY_BLUE, 'color': PRIMARY_WHITE, 'fontWeight': 'bold', 'margin': '0 3px'})
            # Still populate map, graphs, and table, but only show total events for KPIs
            # Jitter coordinates for map visualization
            dff_jittered = jitter_coords(dff, lat_col='lat', lon_col='lon', jitter_amount=0.01)
            if 'size_mean' in dff.columns and 'size_mean' not in dff_jittered.columns:
                dff_jittered['size_mean'] = dff['size_mean']
            # Limit columns for dcc.Store
            table_columns = [col for col in dff_jittered.columns if not (col.startswith('source_') and col != 'source_1')]
            dff_for_store = dff_jittered[table_columns].copy()
            # DataTable: always output a list of records, with NaN/None replaced by ""
            table_records = dff_for_store.replace({np.nan: "", None: ""}).to_dict(orient='records')

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

            dff_jittered['location_label'] = dff_jittered.apply(best_location, axis=1)
            dff_jittered['location_label'] = dff_jittered['location_label'].replace('', 'Unknown').fillna('Unknown')
            agg_map = aggregate_events_for_map(dff_jittered)

            has_size = agg_map[agg_map['size_mean'].notna()]
            no_size = agg_map[agg_map['size_mean'].isna()]
            fig_map = go.Figure()

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
                        opacity=.5,
                        sizemode='area',
                        sizeref=sizeref,
                        sizemin=5
                    ),
                    text=has_size['text'],
                    customdata=has_size[['count', 'size_mean']].values,
                    hovertemplate=(
                        "<b>%{text}</b><br><br>"
                        "Events at this site: %{customdata[0]}<br>"
                        "Participants: %{customdata[1]:,.0f}<br>"
                        "<extra></extra>"
                    ),
                    name="Has Participant Count",
                    showlegend=False
                ))

            if not no_size.empty:
                fig_map.add_trace(go.Scattermapbox(
                    lat=no_size['lat'],
                    lon=no_size['lon'],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=PRIMARY_RED,
                        sizemode='area',
                        opacity=.5,
                        sizeref=1,
                        sizemin=5
                    ),
                    text=no_size['text'],
                    customdata=no_size[['count']].values,
                    hovertemplate=(
                        "<b>%{text}</b><br><br>"
                        "Events at this site: %{customdata[0]}<br>"
                        "<extra></extra>"
                    ),
                    name="Missing Participant Count",
                    showlegend=False
                ))

            fig_map.add_trace(go.Scattermapbox(
                lat=[None], lon=[None],
                mode='markers',
                marker=dict(size=16, color=PRIMARY_BLUE),
                name="Has Participant Count",
                showlegend=True
            ))
            fig_map.add_trace(go.Scattermapbox(
                lat=[None], lon=[None],
                mode='markers',
                marker=dict(size=16, color=PRIMARY_RED),
                name="Missing Participant Count",
                showlegend=True
            ))

            if not dff.empty and (city_filter and len(city_filter) > 0):
                center_lat = dff['lat'].mean()
                center_lon = dff['lon'].mean()
                zoom = 10 if len(city_filter) == 1 else 13
            elif not dff.empty and (state_filter and len(state_filter) > 0):
                center_lat = dff['lat'].mean()
                center_lon = dff['lon'].mean()
                zoom = 5
            else:
                center_lat = 39.8283
                center_lon = -98.5795
                zoom = 3

            fig_map.update_layout(
                mapbox_style="carto-positron",
                mapbox_zoom=zoom,
                mapbox_center={"lat": center_lat, "lon": center_lon},
                margin=standard_margin,
                height=500,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.08,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=12)
                )
            )

            # Optimized Momentum Graph
            dff_momentum = dff[['date', 'participants_numeric']].dropna()
            if not dff_momentum.empty:
                # Use groupby and transform for vectorized rolling
                dff_momentum = dff_momentum.groupby('date').agg(sum=('participants_numeric', 'sum'), count=('participants_numeric', 'size'))
                dff_momentum['momentum'] = (dff_momentum['sum'] * dff_momentum['count']).rolling(7, min_periods=1).sum()
                dff_momentum.reset_index(inplace=True)
                fig_momentum = go.Figure()
                fig_momentum.add_trace(go.Scatter(
                    x=dff_momentum['date'],
                    y=dff_momentum['momentum'],
                    mode='lines',
                    name='Momentum',
                    hovertemplate=(
                        "<b>Momentum of Dissent</b>: %{y:,.0f}<br>"
                        "Date: %{x|%Y-%m-%d}<br>"
                        "<span style='font-size:0.95em;'>"
                        "Momentum of Dissent = (participants on a given day) × (number of events in the 7 days prior)"
                        "</span><extra></extra>"
                    )
                ))
                valid = dff_momentum['momentum'].notna()
                if valid.sum() > 1:
                    # Use ordinal encoding for dates for polyfit
                    xvals = pd.to_datetime(dff_momentum.loc[valid, 'date']).map(pd.Timestamp.toordinal)
                    yvals = dff_momentum.loc[valid, 'momentum']
                    z = np.polyfit(xvals, yvals, 1)
                    p = np.poly1d(z)
                    fig_momentum.add_trace(go.Scatter(
                        x=dff_momentum['date'],
                        y=p(pd.to_datetime(dff_momentum['date']).map(pd.Timestamp.toordinal)),
                        mode='lines',
                        line=dict(dash='dash', color='gray'),
                        name='Trendline',
                        hoverinfo='skip'
                    ))
                fig_momentum.update_layout(height=270, margin=standard_margin)
            else:
                fig_momentum = go.Figure()
                fig_momentum.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
                fig_momentum.update_layout(height=270, margin=standard_margin)

            dff_daily = dff.set_index('date').resample('D').size().reset_index(name='count')
            fig_daily = px.bar(dff_daily, x='date', y='count', height=270, template="plotly_white")
            fig_daily.update_layout(margin={**standard_margin, 'b': 50}, barmode='group')

            dff_cum = dff.set_index('date').resample('D').size().reset_index(name='count')
            dff_cum['cumulative'] = dff_cum['count'].cumsum()
            fig_cumulative = px.line(dff_cum, x='date', y='cumulative', height=250, template="plotly_white")
            fig_cumulative.update_layout(margin={**standard_margin, 'b': 50})  # more space below

            dff_daily_participant = dff.groupby('date').agg(daily_participants=('size_mean', 'sum')).reset_index()
            fig_daily_participant = go.Figure()
            fig_daily_participant.add_trace(go.Bar(
                x=dff_daily_participant['date'],
                y=dff_daily_participant['daily_participants'],
                marker_color=PRIMARY_RED,
                hovertemplate=(
                    "<b>Daily Participant Count</b>: %{y:,.0f}<br>"
                    "Date: %{x|%Y-%m-%d}<br>"
                    "<extra></extra>"
                )
            ))
            fig_daily_participant.update_layout(margin={**standard_margin, 'b': 50})

            # For this filter, show a dash figure for momentum and daily participant count graphs
            def dash_figure(label="-"):
                fig = go.Figure()
                fig.add_annotation(
                    text="-",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=36, color="#888"),
                    align="center"
                )
                fig.update_layout(
                    margin=standard_margin,
                    height=250,
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    plot_bgcolor="#fff",
                    paper_bgcolor="#fff",
                    showlegend=False
                )
                return fig

            outputs = [
                fig_map,  # map-graph
                dash_figure(),  # momentum-graph (shows dash)
                fig_daily,  # daily-graph
                table_records,  # filtered-data (LIMITED COLUMNS, records)
                fig_cumulative,  # cumulative-graph
                dash_figure(),  # daily-participant-graph (shows dash)
                total_events_kpi,  # total-events-kpi
                dash_kpi("Average Participants per Event", "📊"),
                dash_kpi("Events with No Injuries", "🚑"),
                dash_kpi("Events with No Arrests", "🚔"),
                dash_kpi("Events with No Property Damage", "🏚️"),
                dash_kpi("Total Participants", "🌟"),
                dash_kpi("Largest Single Event", "🥇"),
                dash_kpi("Largest Day of Action", "📅"),
                dash_kpi("Largest Day of Action (%)", "🌎"),
                None  # three-point-five-check (empty)
            ]
            return outputs

        # Jitter coordinates for map visualization
        dff_jittered = jitter_coords(dff, lat_col='lat', lon_col='lon', jitter_amount=0.01)
        if 'size_mean' in dff.columns and 'size_mean' not in dff_jittered.columns:
            dff_jittered['size_mean'] = dff['size_mean']
        # Limit columns for dcc.Store
        table_columns = [col for col in dff_jittered.columns if not (col.startswith('source_') and col != 'source_1')]
        dff_for_store = dff_jittered[table_columns].copy()
        # DataTable: always output a list of records, with NaN/None replaced by ""
        table_records = dff_for_store.replace({np.nan: "", None: ""}).to_dict(orient='records')

        # Reuse the same location label logic from aggregate_events_for_map
        # SPEEDUP: Vectorized best_location
        loc = dff_jittered['location'].astype(str).str.strip() if 'location' in dff_jittered.columns else pd.Series(['Unknown']*len(dff_jittered), index=dff_jittered.index)
        loc2 = dff_jittered['locality'].astype(str).str.strip() if 'locality' in dff_jittered.columns else pd.Series(['Unknown']*len(dff_jittered), index=dff_jittered.index)
        state = dff_jittered['state'].astype(str) if 'state' in dff_jittered.columns else pd.Series(['Unknown']*len(dff_jittered), index=dff_jittered.index)
        date = dff_jittered['date'].dt.date.astype(str) if 'date' in dff_jittered.columns else pd.Series(['Unknown']*len(dff_jittered), index=dff_jittered.index)
        location_label = np.where((loc != '') & (loc.str.lower() != 'nan'), loc,
                            np.where((loc2 != '') & (loc2.str.lower() != 'nan'), loc2,
                            state + ', ' + date))
        dff_jittered['location_label'] = location_label
        dff_jittered['location_label'] = dff_jittered['location_label'].replace('', 'Unknown').fillna('Unknown')
        agg_map = aggregate_events_for_map(dff_jittered)

        has_size = agg_map[agg_map['size_mean'].notna()]
        no_size = agg_map[agg_map['size_mean'].isna()]
        fig_map = go.Figure()

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
                    opacity=.5,
                    sizemode='area',
                    sizeref=sizeref,
                    sizemin=5
                ),
                text=has_size['text'],
                customdata=has_size[['count', 'size_mean']].values,
                hovertemplate=(
                    "<b>%{text}</b><br><br>"
                    "Events at this site: %{customdata[0]}<br>"
                    "Participants: %{customdata[1]:,.0f}<br>"
                    "<extra></extra>"
                ),
                name="Has Participant Count",
                showlegend=False
            ))

        if not no_size.empty:
            fig_map.add_trace(go.Scattermapbox(
                lat=no_size['lat'],
                lon=no_size['lon'],
                mode='markers',
                marker=dict(
                    size=12,
                    color=PRIMARY_RED,
                    sizemode='area',
                    opacity=.5,
                    sizeref=1,
                    sizemin=5
                ),
                text=no_size['text'],
                customdata=no_size[['count']].values,
                hovertemplate=(
                    "<b>%{text}</b><br><br>"
                    "Events at this site: %{customdata[0]}<br>"
                    "<extra></extra>"
                ),
                name="Missing Participant Count",
                showlegend=False
            ))

        # Only these traces will appear in the legend, with consistent dot sizes:
        fig_map.add_trace(go.Scattermapbox(
            lat=[None], lon=[None],
            mode='markers',
            marker=dict(size=16, color=PRIMARY_BLUE),
            name="Has Participant Count",
            showlegend=True
        ))
        fig_map.add_trace(go.Scattermapbox(
            lat=[None], lon=[None],
            mode='markers',
            marker=dict(size=16, color=PRIMARY_RED),
            name="Missing Participant Count",
            showlegend=True
        ))

        # Determine map center and zoom
        if not dff.empty and (city_filter and len(city_filter) > 0):
            center_lat = dff['lat'].mean()
            center_lon = dff['lon'].mean()
            zoom = 10 if len(city_filter) == 1 else 13
        elif not dff.empty and (state_filter and len(state_filter) > 0):
            center_lat = dff['lat'].mean()
            center_lon = dff['lon'].mean()
            zoom = 5
        else:
            center_lat = 39.8283
            center_lon = -98.5795
            zoom = 3

        fig_map.update_layout(
            mapbox_style="carto-positron",
            mapbox_zoom=zoom,
            mapbox_center={"lat": center_lat, "lon": center_lon},
            margin=standard_margin,
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.08,
                xanchor="center",
                x=0.5,
                font=dict(size=12)
            )
        )

        # Momentum graph
        dff_momentum = dff[['date', 'participants_numeric']].dropna()
        dff_momentum = dff_momentum.set_index('date').resample('D').agg(['sum', 'count'])
        dff_momentum.columns = ['sum', 'count']
        # Momentum of Dissent OVER 7 DAYS = (sum of participants per day) × (number of events per day), summed over the last 7 days
        dff_momentum['momentum'] = (dff_momentum['sum'] * dff_momentum['count']).rolling(7).sum()
        dff_momentum = dff_momentum.reset_index()

        fig_momentum = go.Figure()
        fig_momentum.add_trace(go.Scatter(
            x=dff_momentum['date'],
            y=dff_momentum['momentum'],
            mode='lines',
            name='Momentum',
            hovertemplate=(
                "<b>Momentum of Dissent</b>: %{y:,.0f}<br>"
                "Date: %{x|%Y-%m-%d}<br>"
                "<span style='font-size:0.95em;'>"
                "Momentum of Dissent = (participants on a given day) × (number of events in the 7 days prior)"
                "</span><extra></extra>"
            )
        ))
        # Add trendline (linear regression) to the 7-day momentum
        valid = dff_momentum['momentum'].notna()
        if valid.sum() > 1:
            z = np.polyfit(
                pd.to_numeric(dff_momentum.loc[valid, 'date']), dff_momentum.loc[valid, 'momentum'], 1
            )
            p = np.poly1d(z)
            fig_momentum.add_trace(go.Scatter(
                x=dff_momentum['date'],
                y=p(pd.to_numeric(dff_momentum['date'])),
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name='Trendline',
                hoverinfo='skip'
            ))
        fig_momentum.update_layout(
            margin={**standard_margin, 'b': 20},
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.1,
                xanchor="center",
                x=0.5,
                font=dict(size=12)
            )
        )


        # Daily event count graph
        dff_daily = dff.set_index('date').resample('D').size().reset_index(name='count')
        fig_daily = px.bar(dff_daily, x='date', y='count', height=270, template="plotly_white")
        fig_daily.update_layout(margin={**standard_margin, 'b': 50}, barmode='group')

        # Cumulative total events graph (correct: cumulative count of events per day)
        dff_cum = dff.set_index('date').resample('D').size().reset_index(name='count')
        dff_cum['cumulative'] = dff_cum['count'].cumsum()
        fig_cumulative = px.line(dff_cum, x='date', y='cumulative', height=250, template="plotly_white")
        fig_cumulative.update_layout(margin={**standard_margin, 'b': 50})  # more space below

        # Daily participant count graph
        dff_daily_participant = dff.groupby('date').agg(daily_participants=('size_mean', 'sum')).reset_index()
        fig_daily_participant = go.Figure()
        fig_daily_participant.add_trace(go.Bar(
            x=dff_daily_participant['date'],
            y=dff_daily_participant['daily_participants'],
            marker_color=PRIMARY_RED,
            hovertemplate=(
                "<b>Daily Participant Count</b>: %{y:,.0f}<br>"
                "Date: %{x|%Y-%m-%d}<br>"
                "<extra></extra>"
            )
        ))
        fig_daily_participant.update_layout(margin={**standard_margin, 'b': 50})


        kpi_number_font = {'fontSize': '1.05rem', 'fontWeight': '700'}

        # KPI CALCULATIONS
        # 1. Total Events
        total_events_kpi = html.Div([
            html.Div("Total Events", style={'fontSize': '0.85rem', 'marginBottom': '4px'}),
            html.Div("🗓️", style={'fontSize': '1.2rem', 'marginTop': '4px'}),
            html.Div(f"{len(dff):,}", style=kpi_number_font),
        ], style={'textAlign': 'center', 'padding': '7px', 'borderRadius': '10px', 'backgroundColor': PRIMARY_BLUE, 'color': PRIMARY_WHITE, 'fontWeight': 'bold', 'margin': '0 3px'})


        # 2. Average Participants per Event (NumPy for speed)
        size_mean_arr = dff['size_mean'].to_numpy()
        mean_size_val = np.nanmean(size_mean_arr) if size_mean_arr.size > 0 else 0.0
        mean_size_kpi = html.Div([
            html.Div("Average Participants per Event", style={'fontSize': '0.85rem', 'marginBottom': '4px'}),
            html.Div("📊", style={'fontSize': '1.2rem', 'marginTop': '4px'}),
            html.Div(f"{mean_size_val:,.0f}", style=kpi_number_font),
        ], style={'textAlign': 'center', 'padding': '7px', 'borderRadius': '10px', 'backgroundColor': PRIMARY_RED, 'color': PRIMARY_WHITE, 'fontWeight': 'bold', 'margin': '0 3px'})

        # 3. Events with No Injuries (NumPy for speed)
        injuries_arr = dff['participant_injuries'].to_numpy()
        no_injuries_pct = (np.count_nonzero((np.isnan(injuries_arr)) | (injuries_arr == 0)) / len(injuries_arr) * 100) if len(injuries_arr) > 0 else 0.0
        no_injuries_kpi = html.Div([
            html.Div("Events with No Injuries", style={'fontSize': '0.85rem', 'marginBottom': '4px'}),
            html.Div("🚑", style={'fontSize': '1.2rem', 'marginTop': '4px'}),
            html.Div(f"{no_injuries_pct:.2f}%", style=kpi_number_font),
        ], style={'textAlign': 'center', 'padding': '7px', 'borderRadius': '10px', 'backgroundColor': PRIMARY_BLUE, 'color': PRIMARY_WHITE, 'fontWeight': 'bold', 'margin': '0 3px'})

        # 4. Events with No Arrests (NumPy for speed)
        arrests_arr = dff['arrests'].to_numpy()
        no_arrests_pct = (np.count_nonzero((np.isnan(arrests_arr)) | (arrests_arr == 0)) / len(arrests_arr) * 100) if len(arrests_arr) > 0 else 0.0
        no_arrests_kpi = html.Div([
            html.Div("Events with No Arrests", style={'fontSize': '0.85rem', 'marginBottom': '4px'}),
            html.Div("🚔", style={'fontSize': '1.2rem', 'marginTop': '4px'}),
            html.Div(f"{no_arrests_pct:.2f}%", style=kpi_number_font),
        ], style={'textAlign': 'center', 'padding': '7px', 'borderRadius': '10px', 'backgroundColor': PRIMARY_RED, 'color': PRIMARY_WHITE, 'fontWeight': 'bold', 'margin': '0 3px', 'border': 'none'})

        # 5. Events with No Property Damage (NumPy for speed)
        prop_damage_arr = dff['property_damage_any'].to_numpy()
        no_damage_pct = (np.count_nonzero(prop_damage_arr == 0) / len(prop_damage_arr) * 100) if len(prop_damage_arr) > 0 else 0.0
        no_damage_kpi = html.Div([
            html.Div("Events with No Property Damage", style={'fontSize': '0.85rem', 'marginBottom': '4px'}),
            html.Div("🏚️", style={'fontSize': '1.2rem', 'marginTop': '4px'}),
            html.Div(f"{no_damage_pct:.2f}%", style=kpi_number_font),
        ], style={'textAlign': 'center', 'padding': '7px', 'borderRadius': '10px', 'backgroundColor': PRIMARY_BLUE, 'color': PRIMARY_WHITE, 'fontWeight': 'bold', 'margin': '0 3px'})

        # 6. Total Participants (NumPy for speed)
        participants_arr = dff['participants_numeric'].to_numpy()
        total_participants_val = int(round(np.nansum(participants_arr))) if participants_arr.size > 0 else 0
        total_participants_kpi = html.Div([
            html.Div("Total Participants", style={'fontSize': '0.85rem', 'marginBottom': '4px'}),
            html.Div("🌟", style={'fontSize': '1.2rem', 'marginTop': '4px'}),
            html.Div(f"{total_participants_val:,}", style=kpi_number_font),
        ], style={'textAlign': 'center', 'padding': '7px', 'borderRadius': '10px', 'backgroundColor': PRIMARY_BLUE, 'color': PRIMARY_WHITE, 'fontWeight': 'bold', 'margin': '0 3px'})

        # 7. Largest Single Event (NumPy for speed)
        largest_event_val = np.nanmax(size_mean_arr) if size_mean_arr.size > 0 else float('nan')
        largest_event_kpi = html.Div([
            html.Div("Largest Single Event", style={'fontSize': '0.85rem', 'marginBottom': '4px'}),
            html.Div("🥇", style={'fontSize': '1.2rem', 'marginTop': '4px'}),
            html.Div(f"{int(round(largest_event_val)):,} participants" if not np.isnan(largest_event_val) else "-", style=kpi_number_font),
        ], style={'textAlign': 'center', 'padding': '7px', 'borderRadius': '10px', 'backgroundColor': PRIMARY_RED, 'color': PRIMARY_WHITE, 'fontWeight': 'bold', 'margin': '0 3px'})


        # --- Optimized 3.5% Threshold and Largest Day Calculation ---
        # Group by date and sum size_mean only once
        # --- Optimized 3.5% Threshold and Related KPIs ---
        # Calculate daily participant sums ONCE and reuse
        if not dff.empty and 'date' in dff.columns and 'size_mean' in dff.columns:
            daily_participant_sums = dff.groupby('date', sort=False)['size_mean'].sum()
            daily_participant_arr = daily_participant_sums.values
            if daily_participant_arr.size > 0:
                largest_day_total = np.nanmax(daily_participant_arr)
            else:
                largest_day_total = float('nan')
        else:
            largest_day_total = float('nan')
            daily_participant_arr = np.array([])

        largest_day_kpi = html.Div([
            html.Div("Largest Day of Action", style={'fontSize': '0.85rem', 'marginBottom': '4px'}),
            html.Div("📅", style={'fontSize': '1.2rem', 'marginTop': '4px'}),
            html.Div(f"{int(round(largest_day_total)):,} participants" if not np.isnan(largest_day_total) else "-", style=kpi_number_font),
        ], style={'textAlign': 'center', 'padding': '7px', 'borderRadius': '10px', 'backgroundColor': PRIMARY_RED, 'color': PRIMARY_WHITE, 'fontWeight': 'bold', 'margin': '0 3px'})

        # 9. Largest Day as % of Population (US or state-adjusted)
        from state_pop import STATE_POP
        pop_denom = US_POPULATION
        pop_label = "US population"
        if state_filter and len(state_filter) > 0:
            state_abbrevs = []
            for s in state_filter:
                if s in STATE_POP:
                    state_abbrevs.append(s)
                else:
                    found = [abbr for abbr in STATE_POP if abbr.lower() == s.lower()]
                    if found:
                        state_abbrevs.append(found[0])
            pop_denom = sum([STATE_POP.get(abbr, 0) for abbr in state_abbrevs])
            if pop_denom > 0:
                pop_label = f"{'/'.join(state_abbrevs)} population"
            else:
                pop_denom = US_POPULATION
                pop_label = "US population"
        if pop_denom > 0 and largest_day_total and not np.isnan(largest_day_total):
            percent_val = (largest_day_total / pop_denom * 100)
        else:
            percent_val = 0.0
        percent_label = f"{percent_val:.2f}% of {pop_label}"
        percent_us_pop_kpi = html.Div([
            html.Div("Largest Day of Action (%)", style={'fontSize': '0.85rem', 'marginBottom': '4px'}),
            html.Div("🌎", style={'fontSize': '1.2rem', 'marginTop': '4px'}),
            html.Div(percent_label, style=kpi_number_font),
        ], style={'textAlign': 'center', 'padding': '7px', 'borderRadius': '10px', 'backgroundColor': PRIMARY_BLUE, 'color': PRIMARY_WHITE, 'fontWeight': 'bold', 'margin': '0 3px'})

        # 3.5% of US population (or selected states)
        three_point_five_threshold = pop_denom * 0.035
        met_threshold = bool(largest_day_total and not np.isnan(largest_day_total) and largest_day_total >= three_point_five_threshold)
        three_point_five_result = {
            'threshold': int(round(three_point_five_threshold)),
            'largest_day_total': int(round(largest_day_total)) if not np.isnan(largest_day_total) else 0,
            'met': met_threshold,
            'percent': percent_val,
            'population': int(pop_denom),
            'label': pop_label
        }

        # Sanitize three_point_five_result for JSON serialization
        def safe_int(val):
            try:
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    return 0
                return int(val)
            except Exception:
                return 0
        def safe_float(val):
            try:
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    return 0.0
                return float(val)
            except Exception:
                return 0.0
        def safe_str(val):
            if val is None:
                return ""
            if isinstance(val, float) and np.isnan(val):
                return ""
            return str(val)
        def safe_bool(val):
            return bool(val)
        three_point_five_result_safe = {
            'threshold': safe_int(three_point_five_result.get('threshold', 0)),
            'largest_day_total': safe_int(three_point_five_result.get('largest_day_total', 0)),
            'met': safe_bool(three_point_five_result.get('met', False)),
            'percent': safe_float(three_point_five_result.get('percent', 0.0)),
            'population': safe_int(three_point_five_result.get('population', 0)),
            'label': safe_str(three_point_five_result.get('label', "")),
        }
        # Return the calculated outputs
        # Return only unique KPIs in the correct order
        return [
            fig_map,  # map-graph
            fig_momentum,  # momentum-graph
            fig_daily,  # daily-graph
            table_records,  # filtered-data (LIMITED COLUMNS, records)
            fig_cumulative,  # cumulative-graph
            fig_daily_participant,  # daily-participant-graph
            total_events_kpi,  # total-events-kpi
            mean_size_kpi,  # mean-size-kpi
            no_injuries_kpi,  # no-injuries-kpi
            no_arrests_kpi,  # no-arrests-kpi
            no_damage_kpi,  # no-damage-kpi
            total_participants_kpi,  # total-participants-kpi
            largest_event_kpi,  # largest-event-kpi
            largest_day_kpi,  # largest-day-kpi
            percent_us_pop_kpi,  # percent-us-pop-kpi
            three_point_five_result_safe  # three-point-five-check (safe for JSON)
        ]

    except Exception as e:
        print("Exception in update_all callback:", e)
        traceback.print_exc()
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text=f"An error occurred while processing the data: {e}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=22, color="red"),
            align="center"
        )
        empty_fig.update_layout(
            mapbox_style="carto-positron",
            mapbox_zoom=3,
            mapbox_center={"lat": 39.8283, "lon": -98.5795},
            margin=standard_margin,
            height=500,
            showlegend=False
        )
        # DataTable: always output a list of records, with NaN/None replaced by ""
        empty_json = []
        # Return blank/empty for each KPI output, not a list of Divs (prevents duplicate KPIs)
        blank_kpi = html.Div("-", style={'fontSize': '1.35rem', 'fontWeight': '700', 'textAlign': 'center'})
        return [
            empty_fig,  # map-graph
            empty_fig,  # momentum-graph
            empty_fig,  # daily-graph
            empty_json, # filtered-data (records)
            empty_fig,  # cumulative-graph
            empty_fig,  # daily-participant-graph
            blank_kpi,  # total-events-kpi
            blank_kpi,  # mean-size-kpi
            blank_kpi,  # no-injuries-kpi
            blank_kpi,  # no-arrests-kpi
            blank_kpi,  # no-damage-kpi
            blank_kpi,  # total-participants-kpi
            blank_kpi,  # largest-event-kpi
            blank_kpi,  # largest-day-kpi
            blank_kpi,  # percent-us-pop-kpi
            None  # three-point-five-check (empty)
        ]

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
        import json
        # Support both string (JSON) and list (records) for filtered_data
        if isinstance(filtered_data, str):
            try:
                dff = pd.read_json(io.StringIO(filtered_data), orient='split')
            except Exception:
                try:
                    dff = pd.read_json(io.StringIO(filtered_data), orient='records')
                except Exception:
                    dff = pd.DataFrame(json.loads(filtered_data))
        elif isinstance(filtered_data, list):
            dff = pd.DataFrame(filtered_data)
        else:
            return html.Div("No event data available.", style={'color': 'red'})

        if dff.empty:
            return html.Div("No event data available.", style={'color': 'red'})

        point = click_data['points'][0]
        location_label = point.get('text')
        lat = point.get('lat')
        lon = point.get('lon')
        if not location_label and (lat is None or lon is None):
            return html.Div("No details available for this location.", style={'color': '#555', 'margin': '12px 0'})

        # Normalize for robust matching
        def norm(x):
            if x is None:
                return ''
            return str(x).strip().lower().replace('\u200b', '').replace('\xa0', ' ').replace('  ', ' ')

        # Try matching by normalized location_label
        norm_label = norm(location_label)
        if 'location_label' in dff.columns:
            dff['__norm_label'] = dff['location_label'].apply(norm)
        else:
            dff['__norm_label'] = ''
        location_events = dff[dff['__norm_label'] == norm_label]

        # Fallback: substring match if exact match fails
        if location_events.empty and norm_label:
            location_events = dff[dff['__norm_label'].str.contains(norm_label, na=False)]

        # Fallback: match by rounded lat/lon if still empty
        if location_events.empty and lat is not None and lon is not None:
            def round_coord(val):
                try:
                    return round(float(val), 4)
                except Exception:
                    return None
            lat_r = round_coord(lat)
            lon_r = round_coord(lon)
            if 'lat' in dff.columns and 'lon' in dff.columns:
                dff['__lat_r'] = dff['lat'].apply(round_coord)
                dff['__lon_r'] = dff['lon'].apply(round_coord)
                location_events = dff[(dff['__lat_r'] == lat_r) & (dff['__lon_r'] == lon_r)]

        if location_events.empty:
            return html.Div("No event details found for this marker.", style={'color': '#555', 'margin': '12px 0'})

        # Always show these fields (with "Unknown" if missing)
        always_fields = [
            ('Title', 'title'),
            ('Date', 'date'),
            ('Location', 'location'),
            ('City', 'resolved_locality'),
            ('State', 'resolved_state'),
            ('County', 'resolved_county'),
            ('Organizations', 'organizations'),
            ('Participants', 'size_mean'),
            ('Targets', 'claims_summary')
        ]

        # Show these fields only if not Unknown
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

            # Always show these fields
            for label, col in always_fields:
                value = event.get(col, 'Unknown')
                if pd.isnull(value) or (isinstance(value, str) and (not value.strip() or value.strip().lower() == 'nan')):
                    value = 'Unknown'
                if col == 'date' and value != 'Unknown' and pd.notnull(value):
                    try:
                        value = pd.to_datetime(value).strftime('%Y-%m-%d')
                    except Exception:
                        value = 'Unknown'
                if col == 'size_mean' and value != 'Unknown' and pd.notnull(value):
                    try:
                        value = f"{int(round(float(value))):,}"
                    except Exception:
                        pass
                # Title: left aligned, not bold
                if label == 'Title':
                    event_detail.append(html.P(f"{label}: {value}", style={'margin': '0 0 4px 0', 'textAlign': 'left'}))
                else:
                    event_detail.append(html.P(f"{label}: {value}", style={'margin': '0 0 4px 0', 'textAlign': 'left'}))

            # Only show optional fields if not Unknown
            for label, col in optional_fields:
                value = event.get(col, 'Unknown')
                if pd.isnull(value) or (isinstance(value, str) and (not value.strip() or value.strip().lower() == 'nan') or str(value).lower() == 'unknown'):
                    continue  # Skip if Unknown
                event_detail.append(html.P(f"{label}: {value}", style={'margin': '0 0 4px 0', 'textAlign': 'left'}))

            title = event.get('title', 'Unknown')
            date = event.get('date', 'Unknown')
            if pd.notnull(date) and date != 'Unknown':
                try:
                    date = pd.to_datetime(date).strftime('%Y-%m-%d')
                except Exception:
                    date = 'Unknown'
            header = f"{title} - {date}"

            details.append(
                html.Details([
                    html.Summary(header, style={'fontWeight': 'bold', 'fontSize': '1.1em', 'textAlign': 'center'}),
                    html.Div(event_detail, style={'marginLeft': '12px'})
                ], open=True, style={'marginBottom': '16px'})
            )

        return html.Div(details, style={'padding': '12px'})

    except Exception as e:
        return html.Div(
            [
                html.Div("An error occurred while loading event details:", style={'color': PRIMARY_RED, 'fontWeight': 'bold'}),
                html.Pre(str(e), style={'color': PRIMARY_RED, 'fontSize': '.9em'})
            ],
            style={'fontStyle': 'italic', 'textAlign': 'center', 'padding': '16px 0'}
        )

    

# SPEEDUP & UX: Clean up sources, prettify column names for DataTable
@app.callback(
    [Output('filtered-table', 'data'),
     Output('filtered-table', 'columns')],
    Input('filtered-data', 'data')
)
def update_filtered_table(filtered_data):
    if not filtered_data:
        return [], []
    # PATCH: support both string and list formats
    if isinstance(filtered_data, str):
        dff = pd.read_json(StringIO(filtered_data), orient='split')
    elif isinstance(filtered_data, list):
        dff = pd.DataFrame(filtered_data)
    else:
        dff = pd.DataFrame()
    if dff.empty:
        return [], []

    # Remove source columns (source_1, source_2, ...) that are all empty/blank/NaN
    source_cols = [col for col in dff.columns if col.startswith('source_')]
    nonempty_sources = []
    for col in source_cols:
        # Keep if at least one non-empty, non-NaN, non-blank value
        if dff[col].replace('', np.nan).notna().any():
            nonempty_sources.append(col)
    # Remove empty source columns
    keep_cols = [col for col in dff.columns if not col.startswith('source_') or col in nonempty_sources]
    dff = dff[keep_cols]

    # Prettify column names for display
    def prettify(col):
        if col.startswith('source_'):
            return 'Source'
        col2 = col.replace('_', ' ').title()
        # Custom prettifications
        col2 = col2.replace('Us', 'US').replace('Id', 'ID').replace('Url', 'URL')
        col2 = col2.replace('Lat', 'Latitude').replace('Lon', 'Longitude')
        return col2

    columns = []
    for col in dff.columns:
        # If multiple source columns remain, append number for clarity
        if col.startswith('source_') and len(nonempty_sources) > 1:
            label = f"Source {col.split('_')[1]}"
        elif col.startswith('source_'):
            label = 'Source'
        else:
            label = prettify(col)
        columns.append({'name': label, 'id': col})

    # Replace NaN with blank for display
    data = dff.replace({np.nan: ''}).to_dict('records')
    return data, columns
def update_table(data_json):
    if not data_json:
        return [], []

    try:
        dff = pd.read_json(io.StringIO(data_json), orient='split')
        columns = [{'name': col, 'id': col} for col in dff.columns]
        return dff.to_dict('records'), columns

    except Exception as e:
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


@app.callback(
    Output('city-filter', 'options'),
    Output('city-filter', 'value'),
    Input('state-filter', 'value'),
    State('city-filter', 'value')
)
def update_city_options(selected_states, selected_cities):
    if not selected_states:
        # No state selected: clear city options and selection
        return [], []
    # Filter df for selected states and get unique cities
    filtered = df[df['state'].isin(selected_states)]
    cities = sorted(filtered['resolved_locality'].dropna().unique())
    options = [{'label': c, 'value': c} for c in cities]
    # Remove any selected cities that are not in the new options
    new_selected = [c for c in (selected_cities or []) if c in cities]
    return options, new_selected

@app.callback(
    Output('missing-count-message-text-map', 'children'),
    Output('show-missing-link-map', 'style'),
    Output('missing-count-message-text-graphs', 'children'),
    Output('show-missing-link-graphs', 'style'),
    Output('missing-count-message-text-table', 'children'),
    Output('show-missing-link-table', 'style'),
    Input('filtered-data', 'data'),
    State('size-filter', 'value'),
    State('state-filter', 'value'),
    State('city-filter', 'value'),
    State('org-search', 'value'),
    State('date-range', 'start_date'),
    State('date-range', 'end_date'),
    State('any-outcomes-filter', 'value')
)
def update_missing_count_message(
    filtered_json, size_filter, state_filter, city_filter, org_search, start_date, end_date, any_outcomes_filter
):
    LINK_BUTTON_STYLE = {
        'background': 'none',
        'border': 'none',
        'color': PRIMARY_BLUE,
        'textDecoration': 'underline',
        'cursor': 'pointer',
        'padding': 0,
        'font': 'inherit',
        'display': 'inline'
    }

    if not filtered_json:
        msg = "No events match your filters."
        style = {'display': 'none'}
        return msg, style, msg, style, msg, style

    # Support both old (string) and new (list of dicts) formats for filtered_json
    if isinstance(filtered_json, str):
        dff = pd.read_json(io.StringIO(filtered_json), orient='split')
    elif isinstance(filtered_json, list):
        dff = pd.DataFrame(filtered_json)
    else:
        dff = pd.DataFrame()
    total_events = len(dff)
    if dff.empty or 'size_mean' not in dff.columns:
        return no_update, {'display': 'none'}, no_update, {'display': 'none'}, no_update, {'display': 'none'}

    missing_count = dff['size_mean'].isna().sum()

    missing_pct = 100 * missing_count / total_events if total_events else 0

   

    # Always show the link button if there are missing counts, and wire up the n_clicks
    if size_filter == "no":
        msg = html.Span([
            "There are ",
            html.Span(f"{total_events:,}", style={'color': PRIMARY_BLUE, 'fontWeight': 'bold'}),
            " events in the database for your filter selection missing participant counts."
        ])
        style = {'display': 'none'}
    elif size_filter == "has":
        dff_no = filter_data(
            start_date, end_date, "no", org_search, state_filter, city_filter, any_outcomes_filter
        )
        missing_total = len(dff_no)
        if missing_total == 0:
            msg = html.Span([
                "There are ",
                html.Span(f"{total_events:,}", style={'color': PRIMARY_BLUE, 'fontWeight': 'bold'}),
                " events in the database for your filter selections. All have participant counts."
            ])
            style = {'display': 'none'}
        else:
            msg = html.Span([
                "There are ",
                html.Span(f"{total_events:,}", style={'color': PRIMARY_BLUE, 'fontWeight': 'bold'}),
                " events in the database for your filter selections, but ",
                html.Span(f"{missing_total:,}", style={'color': PRIMARY_RED, 'fontWeight': 'bold'}),
                " additional events are missing participant counts. Participant counts are vital for tracking protest size and progress over time."
            ])
            style = LINK_BUTTON_STYLE.copy()
            style['display'] = 'inline'
    else:  # "all"
        if missing_count > 0:
            msg = html.Span([
                "There are ",
                html.Span(f"{total_events:,}", style={'color': PRIMARY_BLUE, 'fontWeight': 'bold'}),
                " events in the database for your filter selections, but ",
                html.Span(f"{missing_pct:.1f}%", style={'color': PRIMARY_RED, 'fontWeight': 'bold'}),
                " (",
                html.Span(f"{missing_count:,}", style={'color': PRIMARY_RED, 'fontWeight': 'bold'}),
                ") of those are missing vital information needed to track protest size and progress over time."
            ])
            style = LINK_BUTTON_STYLE.copy()
            style['display'] = 'inline'
        else:
            msg = html.Span([
                "There are ",
                html.Span(f"{total_events:,}", style={'color': PRIMARY_BLUE, 'fontWeight': 'bold'}),
                " events in the database for your filter selections. All have participant counts."
            ])
            style = {'display': 'none'}

    return msg, style, msg, style, msg, style

@app.callback(
    Output('size-filter', 'value'),
    Output('dashboard-tabs', 'value'),
    Output('show-missing-link-store', 'data'),
    Input('show-missing-link-map', 'n_clicks'),
    Input('show-missing-link-graphs', 'n_clicks'),
    Input('show-missing-link-table', 'n_clicks'),
    Input('dashboard-tabs', 'value'),
    State('show-missing-link-store', 'data'),
    prevent_initial_call=True
)
def handle_show_missing_and_tab(n_map, n_graphs, n_table, tab_value, store_val):
    ctx = callback_context
    if not ctx.triggered:
        return no_update, no_update, store_val

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'show-missing-link-map':
        n_clicks = n_map
    elif triggered_id == 'show-missing-link-graphs':
        n_clicks = n_graphs
    elif triggered_id == 'show-missing-link-table':
        n_clicks = n_table
    else:
        n_clicks = None

    if n_clicks and n_clicks != store_val:
        return 'no', 'table', n_clicks
    return no_update, no_update, store_val

@app.callback(
    Output('threshold-row', 'children'),
    Input('three-point-five-check', 'data'),
    State('filtered-data', 'data')
)
def update_threshold_row(three_point_five_data, filtered_json):
    if not filtered_json or not three_point_five_data:
        return ""
    # PATCH: support both string and list formats
    if isinstance(filtered_json, str):
        dff = pd.read_json(io.StringIO(filtered_json), orient='split')
    elif isinstance(filtered_json, list):
        dff = pd.DataFrame(filtered_json)
    else:
        dff = pd.DataFrame()
    if dff.empty or 'date' not in dff.columns:
        return ""
    latest_date = dff['date'].max()
    if pd.isnull(latest_date):
        return ""
    date_str = pd.to_datetime(latest_date).strftime('%B %d, %Y')
    met = three_point_five_data.get('met', False)
    answer = html.Span("Yes", style={'color': 'green', 'fontWeight': 'bold', 'fontSize': '1.15em', 'margin': '0 8px'}) if met else html.Span("No", style={'color': PRIMARY_RED, 'fontWeight': 'bold', 'fontSize': '1.15em', 'margin': '0 8px'})
    return [
        html.Span("3.5% threshold met?", style={'fontWeight': 'bold', 'marginRight': '8px'}),
        answer,
        html.Span(f"as of {date_str}", style={'marginLeft': '8px', 'fontSize': '0.97em', 'color': '#888', 'fontStyle': 'italic'})
    ]


@app.callback(
    Output('show-missing-link-store', 'data', allow_duplicate=True),
    Input('dashboard-tabs', 'value'),
    prevent_initial_call=True
)
def reset_missing_link_store(tab_value):
    return 0

# Uncomment the following 2 lines to run the app directly and test locally. Comment back out when deploying to production.
if __name__ == '__main__':
    app.run(debug=True)