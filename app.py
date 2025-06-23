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
import time

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
standard_margin = dict(t=30, b=20, l=18, r=18)

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
    html.Div(id='sidebar-dynamic', children=get_sidebar(is_open=True)),
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
        # --- KPIs: Always visible, two rows, red/blue checkerboarded ---
        html.Div([
            # First row
            html.Div([
                html.Div(id='total-events-kpi', style={
                    'flex': '1', 'textAlign': 'center', 'padding': '7px', 'borderRadius': '10px',
                    'backgroundColor': PRIMARY_BLUE, 'color': PRIMARY_WHITE, 'fontWeight': 'bold', 'margin': '0 3px'
                }),
                html.Div(id='largest-event-kpi', style={
                    'flex': '1', 'textAlign': 'center', 'padding': '7px', 'borderRadius': '10px',
                    'backgroundColor': PRIMARY_RED, 'color': PRIMARY_WHITE, 'fontWeight': 'bold', 'margin': '0 3px'
                }),
                html.Div(id='mean-size-kpi', style={
                    'flex': '1', 'textAlign': 'center', 'padding': '7px', 'borderRadius': '10px',
                    'backgroundColor': PRIMARY_BLUE, 'color': PRIMARY_WHITE, 'fontWeight': 'bold', 'margin': '0 3px'
                }),
                html.Div(id='largest-day-kpi', style={
                    'flex': '1', 'textAlign': 'center', 'padding': '7px', 'borderRadius': '10px',
                    'backgroundColor': PRIMARY_RED, 'color': PRIMARY_WHITE, 'fontWeight': 'bold', 'margin': '0 3px'
                }),
                html.Div(id='total-participants-kpi', style={
                    'flex': '1', 'textAlign': 'center', 'padding': '7px', 'borderRadius': '10px',
                    'backgroundColor': PRIMARY_BLUE, 'color': PRIMARY_WHITE, 'fontWeight': 'bold', 'margin': '0 3px'
                }),
            ], style={'display': 'flex', 'gap': '6px', 'marginBottom': '6px'}),
            # Second row
            html.Div([
                html.Div(id='percent-no-size-kpi', style={
                    'flex': '1', 'textAlign': 'center', 'padding': '7px', 'borderRadius': '10px',
                    'backgroundColor': PRIMARY_RED, 'color': PRIMARY_WHITE, 'fontWeight': 'bold', 'margin': '0 3px'
                }),
                html.Div(id='no-injuries-kpi', style={
                    'flex': '1', 'textAlign': 'center', 'padding': '7px', 'borderRadius': '10px',
                    'backgroundColor': PRIMARY_BLUE, 'color': PRIMARY_WHITE, 'fontWeight': 'bold', 'margin': '0 3px'
                }),
                html.Div(id='no-arrests-kpi', style={
                    'flex': '1', 'textAlign': 'center', 'padding': '7px', 'borderRadius': '10px',
                    'backgroundColor': PRIMARY_RED, 'color': PRIMARY_WHITE, 'fontWeight': 'bold', 'margin': '0 3px'
                }),
                html.Div(id='no-damage-kpi', style={
                    'flex': '1', 'textAlign': 'center', 'padding': '7px', 'borderRadius': '10px',
                    'backgroundColor': PRIMARY_BLUE, 'color': PRIMARY_WHITE, 'fontWeight': 'bold', 'margin': '0 3px'
                }),
                html.Div(id='percent-us-pop-kpi', style={
                    'flex': '1', 'textAlign': 'center', 'padding': '7px', 'borderRadius': '10px',
                    'backgroundColor': PRIMARY_RED, 'color': PRIMARY_WHITE, 'fontWeight': 'bold', 'margin': '0 3px'
                }),
            ], style={'display': 'flex', 'gap': '6px', 'marginBottom': '12px'}),
        ], style={'marginBottom': '10px'}),
        # --- Tabs and tabbed content ---
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
                    html.Div(id='event-details-panel')
                ]),
                dcc.Tab(label='Graphs', value='graphs', children=[
                    html.Div([
                        html.Div("Momentum of Dissent", style={
                            'fontWeight': 'bold',
                            'fontSize': '1.1em',
                            'marginBottom': '4px',
                            'textAlign': 'center'
                        }),
                        dcc.Graph(id='momentum-graph', config={'displayModeBar': True})
                    ]),
                    html.Div([
                        html.Div("Daily Event Count", style={
                            'fontWeight': 'bold',
                            'fontSize': '1.1em',
                            'marginBottom': '4px',
                            'textAlign': 'center'
                        }),
                        dcc.Graph(id='daily-graph', config={'displayModeBar': True})
                    ]),
                    html.Div([
                        html.Div("Cumulative Total Events", style={
                            'fontWeight': 'bold',
                            'fontSize': '1.1em',
                            'marginBottom': '4px',
                            'textAlign': 'center'
                        }),
                        dcc.Graph(id='cumulative-graph', config={'displayModeBar': True})
                    ]),
                    html.Div([
                        html.Div("Daily Participant Count", style={
                            'fontWeight': 'bold',
                            'fontSize': '1.1em',
                            'marginBottom': '4px',
                            'textAlign': 'center'
                        }),
                        dcc.Graph(id='daily-participant-graph', config={'displayModeBar': True})
                    ])
                ]),
                dcc.Tab(label='Table', value='table', children=[
                    dash_table.DataTable(
                        id='filtered-table',
                        columns=[],
                        style_table={
                            'overflowY': 'auto',
                            'maxHeight': '500px',
                            'overflowX': 'auto',
                            'width': '100%',
                            'minWidth': '100%'
                        },
                        style_cell={
                            'textAlign': 'left',
                            'padding': '10px',
                            'fontFamily': FONT_FAMILY,
                            'fontSize': '14px'
                        },
                        style_header={
                            'fontWeight': 'bold',
                            'backgroundColor': PRIMARY_BLUE,
                            'color': PRIMARY_WHITE
                        },
                        virtualization=True,
                        fixed_rows={'headers': True}
                    )
                ])
            ],
            style={'marginBottom': '16px'}
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
        'transition': 'width 0.3s cubic-bezier(.4,2,.6,1)'
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
from dash.dependencies import ALL

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
    city_filter, any_outcomes_filter
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
       Output('filtered-data', 'data'),
       Output('cumulative-graph', 'figure'),
       Output('daily-participant-graph', 'figure'),
       Output('total-events-kpi', 'children'),
       Output('largest-event-kpi', 'children'),
       Output('mean-size-kpi', 'children'),
       Output('largest-day-kpi', 'children'),
       Output('total-participants-kpi', 'children'),
       Output('percent-no-size-kpi', 'children'),
       Output('no-injuries-kpi', 'children'),
       Output('no-arrests-kpi', 'children'),
       Output('no-damage-kpi', 'children'),
       Output('percent-us-pop-kpi', 'children'),
   ],
   [
       Input('date-range', 'start_date'),
       Input('date-range', 'end_date'),
       Input('size-filter', 'value'),
       Input('org-search', 'value'),
       Input('state-filter', 'value'),
       Input('city-filter', 'value'),
       Input('any-outcomes-filter', 'value')
   ]
)
def update_all(
    start_date, end_date, size_filter, org_search, state_filter,
    city_filter, any_outcomes_filter
):
    t0 = time.time()
    dff = filter_data(
        start_date, end_date, size_filter, org_search, state_filter,
        city_filter, any_outcomes_filter
    )
    t1 = time.time()

    # Metrics
    total_events = len(dff)
    total_participants = dff['size_mean'].sum() if 'size_mean' in dff.columns else 0
    mean_size = dff['size_mean'].mean() if 'size_mean' in dff.columns else 0
    percent_no_size = 100 * dff['size_mean'].isna().sum() / total_events if total_events > 0 else 0
    largest_event = dff['size_mean'].max() if 'size_mean' in dff.columns and not dff['size_mean'].isnull().all() else 0
    largest_day = dff.groupby('date')['size_mean'].sum().max() if 'size_mean' in dff.columns and not dff['size_mean'].isnull().all() else 0
    percent_us_pop = (largest_day / US_POPULATION) * 100 if largest_day else 0

    percent_no_injuries = 100 * (dff['participant_injuries'].isna().sum() / total_events) if total_events > 0 else 0
    percent_no_arrests = 100 * (dff['arrests'].isna().sum() / total_events) if total_events > 0 else 0

    # Use property_damage_any for KPI calculation
    if 'property_damage_any' in dff.columns:
        percent_no_damage = 100 * (dff['property_damage_any'] == 0).sum() / total_events if total_events > 0 else 0
    else:
        percent_no_damage = 0

    # KPI children (match layout order)
    total_events_kpi = [
        html.Div([
            html.Div(f"{total_events:,}", style={'fontSize': '1.35rem', 'fontWeight': '700'}),
            html.Div("🗓️", style={'fontSize': '1.2rem', 'margin': '0'}),
            html.Div("Total Events", style={'fontSize': '0.85rem', 'margin': '0'})
        ], style={'marginBottom': '0'})
    ]
    largest_event_kpi = [
        html.Div([
            html.Div(f"{largest_event:,.0f} participants", style={'fontSize': '1.25rem', 'fontWeight': '700'}),
            html.Div("🥇", style={'fontSize': '1.2rem', 'margin': '0'}),
            html.Div("Largest Event", style={'fontSize': '0.85rem', 'margin': '0'})
        ], style={'marginBottom': '0'})
    ]
    mean_size_kpi = [
        html.Div([
            html.Div(f"{mean_size:,.0f}", style={'fontSize': '1.35rem', 'fontWeight': '700'}),
            html.Div("📊", style={'fontSize': '1.2rem', 'margin': '0'}),
            html.Div("Average Participant Count", style={'fontSize': '0.85rem', 'margin': '0'})
        ], style={'marginBottom': '0'})
    ]
    largest_day_kpi = [
        html.Div([
            html.Div(f"{largest_day:,.0f} participants", style={'fontSize': '1.25rem', 'fontWeight': '700'}),
            html.Div("🥇", style={'fontSize': '1.2rem', 'margin': '0'}),
            html.Div("Largest Day", style={'fontSize': '0.85rem', 'margin': '0'})
        ], style={'marginBottom': '0'})
    ]
    total_participants_kpi = [
        html.Div([
            html.Div(f"{total_participants:,.0f}", style={'fontSize': '1.35rem', 'fontWeight': '700'}),
            html.Div("🌟", style={'fontSize': '1.2rem', 'margin': '0'}),
            html.Div("Total Participants", style={'fontSize': '0.85rem', 'margin': '0'})
        ], style={'marginBottom': '0'})
    ]
    percent_no_size_kpi = [
        html.Div([
            html.Div(f"{percent_no_size:.1f}%", style={'fontSize': '1.35rem', 'fontWeight': '700'}),
            html.Div("🔍", style={'fontSize': '1.2rem', 'margin': '0'}),
            html.Div("Events Missing Size", style={'fontSize': '0.85rem', 'margin': '0'})
        ], style={'marginBottom': '0'})
    ]
    no_injuries_kpi = [
        html.Div([
            html.Div(f"{percent_no_injuries:.1f}%", style={'fontSize': '1.35rem', 'fontWeight': '700'}),
            html.Div("🚑", style={'fontSize': '1.2rem', 'margin': '0'}),
            html.Div("% with No Injuries", style={'fontSize': '0.85rem', 'margin': '0'})
        ], style={'marginBottom': '0'})
    ]
    no_arrests_kpi = [
        html.Div([
            html.Div(f"{percent_no_arrests:.1f}%", style={'fontSize': '1.35rem', 'fontWeight': '700'}),
            html.Div("🚔", style={'fontSize': '1.2rem', 'margin': '0'}),
            html.Div("% with No Arrests", style={'fontSize': '0.85rem', 'margin': '0'})
        ], style={'marginBottom': '0'})
    ]
    no_damage_kpi = [
        html.Div([
            html.Div(f"{percent_no_damage:.1f}%", style={'fontSize': '1.35rem', 'fontWeight': '700'}),
            html.Div("🏚️", style={'fontSize': '1.2rem', 'margin': '0'}),
            html.Div("% with No Property Damage", style={'fontSize': '0.85rem', 'margin': '0'})
        ], style={'marginBottom': '0'})
    ]
    percent_us_pop_kpi = [
        html.Div([
            html.Div(f"{percent_us_pop:.4f}%", style={'fontSize': '1.35rem', 'fontWeight': '700'}),
            html.Div("🌎", style={'fontSize': '1.2rem', 'margin': '0'}),
            html.Div("Most Daily Participants as % of USA", style={'fontSize': '0.85rem', 'margin': '0'})
        ], style={'marginBottom': '0'})
    ]

    # Defensive: Ensure 'lat' and 'lon' columns exist and are not all missing
    if 'lat' not in dff.columns or 'lon' not in dff.columns or dff['lat'].isnull().all() or dff['lon'].isnull().all():
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
        empty_json = dff.to_json(date_format='iso', orient='split')
        dash_kpi = lambda label, icon="—": [
            html.Div([
                html.Div("-", style={'fontSize': '1.35rem', 'fontWeight': '700'}),
                html.Div(icon, style={'fontSize': '1.2rem', 'margin': '0'}),
                html.Div(label, style={'fontSize': '0.85rem', 'margin': '0'})
            ], style={'marginBottom': '0'})
        ]
        return (
            empty_fig,  # map-graph
            empty_fig,  # momentum-graph
            empty_fig,  # daily-graph
            empty_json, # filtered-data
            empty_fig,  # cumulative-graph
            empty_fig,  # daily-participant-graph
            dash_kpi("Total Events", "🗓️"),
            dash_kpi("Largest Event", "🥇"),
            dash_kpi("Average Participant Count", "📊"),
            dash_kpi("Largest Day", "🥇"),
            dash_kpi("Total Participants", "🌟"),
            dash_kpi("Events Missing Size", "🔍"),
            dash_kpi("% with No Injuries", "🚑"),
            dash_kpi("% with No Arrests", "🚔"),
            dash_kpi("% with No Property Damage", "🏚️"),
            dash_kpi("Most Daily Participants as % of USA", "🌎")
        )

    # Jitter coordinates for map visualization
    dff_jittered = jitter_coords(dff, lat_col='lat', lon_col='lon', jitter_amount=0.01)
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
            showlegend=False  # Hide from legend
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
            showlegend=False  # Hide from legend
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
            y=-0.18,
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
        x=dff_momentum['date'], y=dff_momentum['momentum'], mode='lines', name='Momentum (7-Day Sum)'
    ))
    # Add trendline (linear regression) to the 7-day momentum
    valid = dff_momentum['momentum'].notna()
    if valid.sum() > 1:
        z = np.polyfit(
            pd.to_numeric(dff_momentum.loc[valid, 'date']), dff_momentum.loc[valid, 'momentum'], 1
        )
        p = np.poly1d(z)
        fig_momentum.add_trace(go.Scatter(
            x=dff_momentum.loc[valid, 'date'],
            y=p(pd.to_numeric(dff_momentum.loc[valid, 'date'])),
            mode='lines',
            name='Trendline',
            line=dict(dash='dash', color='gray')
        ))
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
    fig_daily_participant_graph = px.bar(
        dff_participants, x='date', y='participants', height=250, template="plotly_white"
    )
    fig_daily_participant_graph.update_layout(margin=standard_margin)

    # Ensure location_label is present in dff before storing
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

    if 'location_label' not in dff.columns:
        dff['location_label'] = dff.apply(best_location, axis=1)

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
        percent_no_size_kpi,
        no_injuries_kpi,
        no_arrests_kpi,
        no_damage_kpi,
        percent_us_pop_kpi
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
        location_label = point.get('text')
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

        # Always show these fields (with "Unknown" if missing)
        always_fields = [
            ('Title', 'title'),
            ('Date', 'date'),
            ('Location', 'location'),
            ('Organizations', 'organizations'),
            ('Participants', 'size_mean'),
            ('Targets', 'targets'),
            ('Claims', 'claims_summary')
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
                if col == 'date' and pd.notnull(value) and value != 'Unknown':
                    try:
                        value = pd.to_datetime(value).strftime('%Y-%m-%d')
                    except Exception:
                        value = 'Unknown'
                event_detail.append(html.P(f"{label}: {value}", style={'margin': '0 0 4px 0'}))

            # Only show optional fields if not Unknown
            for label, col in optional_fields:
                value = event.get(col, 'Unknown')
                if pd.isnull(value) or (isinstance(value, str) and (not value.strip() or value.strip().lower() == 'nan')):
                    continue  # Skip if Unknown
                event_detail.append(html.P(f"{label}: {value}", style={'margin': '0 0 4px 0'}))

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
                    html.Summary(header, style={'fontWeight': 'bold', 'fontSize': '1.1em'}),
                    html.Div(event_detail, style={'marginLeft': '12px'})
                ], open=True, style={'marginBottom': '16px'})
            )

        return html.Div(details, style={'padding': '12px'})

    except Exception as e:
        return html.Div(
            f"An error occurred while loading event details: {str(e)}",
            style={'color': PRIMARY_RED, 'fontSize': '.9em', 'fontStyle': 'italic', 'textAlign': 'center', 'padding': '16px 0'}
        )


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

    except Exception as e:
        return [], []


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

# Uncomment the following 2 lines to run the app directly and test locally. Comment back out when deploying to production.
if __name__ == '__main__':
    app.run(debug=True)