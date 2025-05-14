import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx
import csv
import os


file_path = "C:/Users/jamie/OneDrive/Documents/CCC Project/ccc-phase3-public.csv"
US_POPULATION = 340_100_000

# Load data
df = pd.read_csv(file_path, encoding='latin1', low_memory=False)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['size_mean'] = pd.to_numeric(df['size_mean'], errors='coerce')
df['participants_numeric'] = df['size_mean']
df['targets'] = df['targets'].astype(str).str.lower()
df['organizations'] = df['organizations'].astype(str).str.lower()

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Protest Dashboard"

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# PAGE 1 

dashboard_layout = html.Div([
    # Sidebar
    html.Div([
        dcc.Link('📥 Submit New Protest Event', href='/submit', style={'display': 'block', 'margin': '10px 0 20px 0', 'fontWeight': 'bold'}),
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

# PAGE 2 (Submission Form)

form_layout = html.Div([
    dcc.Link('← Back to Dashboard', href='/', style={'display': 'block', 'margin': '10px'}),
    html.H2("Submit a Protest Event"),

    # Email
    html.Label("Email"),
    dcc.Input(id='input-email', type='email', placeholder='Email', style={'marginBottom': '10px', 'width': '100%', 'borderRadius': '6px', 'boxShadow': '0 1px 4px rgba(0,0,0,0.04)'}),

    # Date of Event
    html.Label("Date of Event"),
    dcc.DatePickerSingle(id='input-date', placeholder='Date of Event', style={'marginBottom': '10px'}),

    # Locality (general city)
    html.Label("Locality (General City)"),
    dcc.Input(id='input-locality', type='text', placeholder='General City', style={'marginBottom': '10px', 'width': '100%', 'borderRadius': '6px', 'boxShadow': '0 1px 4px rgba(0,0,0,0.04)'}),

    # State or Territory
    html.Label("State or Territory"),
    dcc.Dropdown(
        id='input-state',
        options=[{'label': s, 'value': s} for s in [
            "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "District of Columbia", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Puerto Rico", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming", "Other"
        ]],
        placeholder='Select State or Territory',
        style={'marginBottom': '10px'}
    ),

    # Location (specific)
    html.Label("Location (Specific City/Town/Place)"),
    dcc.Input(id='input-location', type='text', placeholder='Specific Location', style={'marginBottom': '10px', 'width': '100%', 'borderRadius': '6px', 'boxShadow': '0 1px 4px rgba(0,0,0,0.04)'}),

    # Title of Event
    html.Label("Title of Event"),
    dcc.Input(id='input-title', type='text', placeholder='Title of Event', style={'marginBottom': '10px', 'width': '100%', 'borderRadius': '6px', 'boxShadow': '0 1px 4px rgba(0,0,0,0.04)'}),

    # Event Type (multi-select)
    html.Label("Event Type (Select all that apply)"),
    dcc.Dropdown(
        id='input-event-type',
        options=[
            {'label': t, 'value': t} for t in [
                "Protest", "Rally", "Demonstration", "March", "Caravan", "Bicycle ride", "Direct action", "Wear Coordinated Colors", "Walk-Out", "Vigil*", "Silent Protest", "Parade*", "Disrupt Public Comment", "Commemorative Gathering*", "Sit-In", "Go-Slow", "Coordinated Non-Compliance", "Banner Drop", "Car Caravan", "Die-In", "Memorial*", "Strike", "Picket", "Boat parade", "Motorcycle ride", "Run", "Walk", "Counter-protest", "Other"
            ]
        ],
        multi=True,
        placeholder='Select Event Type(s)',
        style={'marginBottom': '10px'}
    ),

    # Organization Name
    html.Label("Organization Name"),
    dcc.Input(id='input-organization', type='text', placeholder='Organization Name', style={'marginBottom': '10px', 'width': '100%', 'borderRadius': '6px', 'boxShadow': '0 1px 4px rgba(0,0,0,0.04)'}),

    # Participant Type (multi-select)
    html.Label("Participant Type (Select all that apply)"),
    dcc.Dropdown(
        id='input-participant-type',
        options=[
            {'label': t, 'value': t} for t in [
                "Students", "Workers", "Teachers", "Parents", "Community Members", "Activists", "Other"
            ]
        ],
        multi=True,
        placeholder='Select Participant Type(s)',
        style={'marginBottom': '10px'}
    ),

    # Notable Participants
    html.Label("Notable Participants (well-known people)"),
    dcc.Input(id='input-notable-participants', type='text', placeholder='Notable Participants', style={'marginBottom': '10px', 'width': '100%', 'borderRadius': '6px', 'boxShadow': '0 1px 4px rgba(0,0,0,0.04)'}),

    # Targets
    html.Label("Targets (e.g., Trump, Musk, DOGE)"),
    dcc.Input(id='input-targets', type='text', placeholder='Targets', style={'marginBottom': '10px', 'width': '100%', 'borderRadius': '6px', 'boxShadow': '0 1px 4px rgba(0,0,0,0.04)'}),

    # Claims Summary
    html.Label("Claims Summary (e.g., pro-democracy; against President Trump)"),
    dcc.Input(id='input-claims-summary', type='text', placeholder='Claims Summary', style={'marginBottom': '10px', 'width': '100%', 'borderRadius': '6px', 'boxShadow': '0 1px 4px rgba(0,0,0,0.04)'}),

    # Claims Verbatim
    html.Label("Claims Verbatim (unique words/phrases on signs/chants)"),
    dcc.Textarea(id='input-claims-verbatim', placeholder='Claims Verbatim', style={'marginBottom': '10px', 'width': '100%', 'borderRadius': '6px', 'boxShadow': '0 1px 4px rgba(0,0,0,0.04)'}),

    # Macroevent
    html.Label("Macroevent (larger event being disrupted, if any)"),
    dcc.Input(id='input-macroevent', type='text', placeholder='Macroevent', style={'marginBottom': '10px', 'width': '100%', 'borderRadius': '6px', 'boxShadow': '0 1px 4px rgba(0,0,0,0.04)'}),

    # Online (binary)
    html.Label("Was this an online event?"),
    dcc.RadioItems(
        id='input-online',
        options=[
            {'label': 'Yes', 'value': 'Yes'},
            {'label': 'No', 'value': 'No'}
        ],
        value='No',
        labelStyle={'display': 'inline-block', 'marginRight': '15px'},
        style={'marginBottom': '10px'}
    ),

    # Crowd Size Estimate (lower)
    html.Label("Lower End Estimate of Crowd Size"),
    dcc.Input(id='input-size-lower', type='number', placeholder='Lower End Estimate', style={'marginBottom': '10px'}),

    # Crowd Size Estimate (higher)
    html.Label("Higher End Estimate of Crowd Size"),
    dcc.Input(id='input-size-higher', type='number', placeholder='Higher End Estimate', style={'marginBottom': '10px'}),

    # Participant Measures (multi-select)
    html.Label("Participant Measures (Select all that apply)"),
    dcc.Dropdown(
        id='input-participant-measures',
        options=[
            {'label': m, 'value': m} for m in [
                "Banner overpass", "Candlelight vigil", "Wear coordinated colors", "Stand on roadside with signs", "Amplified sound", "Stand on over pass", "Painted stencils", "Stood in front of media headquarters", "Verbally confronted rally goers", "Disrupted parade", "Vigil", "Drums", "Megaphone", "Pots & pans", "Banners", "Flags", "Stood outside [building]", "Group meditation", "Stood at intersection", "Bicycle rode", "Used white boards", "Flyered passers-by", "Verbal clashes with counter protesters", "Signs", "Stood on median", "Chanted", "Painted mural", "Memorial display", "car caravan", "Other"
            ]
        ],
        multi=True,
        placeholder='Select Participant Measures',
        style={'marginBottom': '10px'}
    ),

    # Police Measures (multi-select)
    html.Label("Police Measures (Select all that apply)"),
    dcc.Dropdown(
        id='input-police-measures',
        options=[
            {'label': m, 'value': m} for m in [
                "on scene", "called to scene", "arrived on scene", "parked nearby", "outside of the venue", "observed from a distance", "state troopers", "campus security", "members of Police Department", "County Sheriff’s Officers", "many jurisdictions involved", "Mobile Response Team", "security guards", "separated counter protesters", "forced protestors off street and into sidewalk", "arrested protester", "formed skirmish lines to block marchers", "kept demonstrators out of street", "gave dispersal order", "fired less-lethal munitions", "kettled", "formed perimeter", "locked all doors to building", "horseback", "stood between demonstrators and counter-protesters", "arrest warning for trespassing", "escorted marchers", "formed lines in front of protesters", "scuffled with protesters while making arrest", "flanked marchers with bicycles", "pulled counter-protester away from marchers", "walked with march", "stood in loose line between demonstrators and location", "gave demonstrators instructions on where to stand", "trailed march in cruisers", "intervened in confrontation between demonstrators and counter-protesters", "arrest warning for demonstrating in roadway", "formed skirmish line in street", "dispersal order and arrest warnings", "confiscated banners", "instructed activists to sit on curb", "discussed what to do with activists", "removed protesters from room", "designated area across street for counter-protesters", "followed and flanked marchers", "confront demonstrators", "fired pepper balls at demonstrators", "sprayed mace at demonstrators", "swarmed and arrested participants in march", "told demonstrators they were not allowed to use amplified sound", "threatened to arrest protestors if they didn’t move onto the sidewalks", "intervened in scuffle between protestors and counter protestors", "warned protesters about disruptions", "carried three protesters out of area", "handcuffed counter-protester", "handcuffed counter-protester then let them go", "handcuffed protester", "handcuffed protester then let him go", "declared unlawful assembly", "used tear gas to disperse crowd", "ordered demonstrators to clear roadway", "gave arrest warnings via LRAD [what languages?]", "closed surrounding streets", "directed traffic around participants", "handed out public advisement sheets", "advised participants not to block traffic or entrances to public buildings", "released chemical agent to disperse crowd", "fired smoke canisters", "fired flash grenades", "withdrew", "withdrew and returned", "closed intersection", "rerouted traffic", "barred entry", "anyone who left not allowed to re-enter", "in large numbers", "Option 77", "barricades", "bicycles", "barricades", "openly carried long guns", "temporary security fencing encircling", "helmets and batons", "line of cruisers", "riot gear with shields", "riot gear", "Other"
            ]
        ],
        multi=True,
        placeholder='Select Police Measures',
        style={'marginBottom': '10px'}
    ),

    # Binary with other option: participant injury
    html.Label("Participant Injury?"),
    dcc.RadioItems(
        id='input-participant-injury',
        options=[
            {'label': 'Yes', 'value': 'Yes'},
            {'label': 'No', 'value': 'No'},
            {'label': 'Other', 'value': 'Other'}
        ],
        value='No',
        labelStyle={'display': 'inline-block', 'marginRight': '15px'},
        style={'marginBottom': '10px'}
    ),
    dcc.Input(id='input-participant-injury-other', type='text', placeholder='If Other, please specify', style={'marginBottom': '10px', 'width': '100%', 'borderRadius': '6px', 'boxShadow': '0 1px 4px rgba(0,0,0,0.04)'}),

    # Binary with other option: police injury
    html.Label("Police Injury?"),
    dcc.RadioItems(
        id='input-police-injury',
        options=[
            {'label': 'Yes', 'value': 'Yes'},
            {'label': 'No', 'value': 'No'},
            {'label': 'Other', 'value': 'Other'}
        ],
        value='No',
        labelStyle={'display': 'inline-block', 'marginRight': '15px'},
        style={'marginBottom': '10px'}
    ),
    dcc.Input(id='input-police-injury-other', type='text', placeholder='If Other, please specify', style={'marginBottom': '10px', 'width': '100%', 'borderRadius': '6px', 'boxShadow': '0 1px 4px rgba(0,0,0,0.04)'}),

    # Binary with other option: arrests
    html.Label("Arrests?"),
    dcc.RadioItems(
        id='input-arrests',
        options=[
            {'label': 'Yes', 'value': 'Yes'},
            {'label': 'No', 'value': 'No'},
            {'label': 'Other', 'value': 'Other'}
        ],
        value='No',
        labelStyle={'display': 'inline-block', 'marginRight': '15px'},
        style={'marginBottom': '10px'}
    ),
    dcc.Input(id='input-arrests-other', type='text', placeholder='If Other, please specify', style={'marginBottom': '10px', 'width': '100%', 'borderRadius': '6px', 'boxShadow': '0 1px 4px rgba(0,0,0,0.04)'}),

    # Binary with other option: property damage
    html.Label("Property Damage?"),
    dcc.RadioItems(
        id='input-property-damage',
        options=[
            {'label': 'Yes', 'value': 'Yes'},
            {'label': 'No', 'value': 'No'},
            {'label': 'Other', 'value': 'Other'}
        ],
        value='No',
        labelStyle={'display': 'inline-block', 'marginRight': '15px'},
        style={'marginBottom': '10px'}
    ),
    dcc.Input(id='input-property-damage-other', type='text', placeholder='If Other, please specify', style={'marginBottom': '10px', 'width': '100%', 'borderRadius': '6px', 'boxShadow': '0 1px 4px rgba(0,0,0,0.04)'}),

    # Binary with other option: participant casualties
    html.Label("Participant Casualties?"),
    dcc.RadioItems(
        id='input-participant-casualties',
        options=[
            {'label': 'Yes', 'value': 'Yes'},
            {'label': 'No', 'value': 'No'},
            {'label': 'Other', 'value': 'Other'}
        ],
        value='No',
        labelStyle={'display': 'inline-block', 'marginRight': '15px'},
        style={'marginBottom': '10px'}
    ),
    dcc.Input(id='input-participant-casualties-other', type='text', placeholder='If Other, please specify', style={'marginBottom': '10px', 'width': '100%', 'borderRadius': '6px', 'boxShadow': '0 1px 4px rgba(0,0,0,0.04)'}),

    # Binary with other option: police casualties
    html.Label("Police Casualties?"),
    dcc.RadioItems(
        id='input-police-casualties',
        options=[
            {'label': 'Yes', 'value': 'Yes'},
            {'label': 'No', 'value': 'No'},
            {'label': 'Other', 'value': 'Other'}
        ],
        value='No',
        labelStyle={'display': 'inline-block', 'marginRight': '15px'},
        style={'marginBottom': '10px'}
    ),
    dcc.Input(id='input-police-casualties-other', type='text', placeholder='If Other, please specify', style={'marginBottom': '10px', 'width': '100%', 'borderRadius': '6px', 'boxShadow': '0 1px 4px rgba(0,0,0,0.04)'}),

    # Notes (multi-select with other)
    html.Label("Notes (Select all that apply)"),
    dcc.Dropdown(
        id='input-notes',
        options=[
            {'label': n, 'value': n} for n in [
                "Part of a nat day of action", "daily event", "every school day", "weekly event", "annual event [i.e. 10th annual]", "A certain day each month [14th of each month; 1st Monday of the month]", "notes from organizers", "before a [city, county, state, corporate, organization, etc] meeting", "If on a campus where on campus [outside a building, in a plaza, etc]", "any trespass warnings", "if it's in conjunction with a meeting", "Indoor/outdoor", "Any change of plans", "First day of legislative session", "Any notable speakers", "Other"
            ]
        ],
        multi=True,
        placeholder='Select Notes',
        style={'marginBottom': '10px'}
    ),
    dcc.Input(id='input-notes-other', type='text', placeholder='If Other, please specify', style={'marginBottom': '10px', 'width': '100%', 'borderRadius': '6px', 'boxShadow': '0 1px 4px rgba(0,0,0,0.04)'}),

    # Sources (text input with option to add another)
    html.Label("Sources (news, media, social posts, etc.)"),
    dcc.Textarea(id='input-sources', placeholder='List sources, separated by semicolons or line breaks', style={'marginBottom': '10px', 'width': '100%', 'borderRadius': '6px', 'boxShadow': '0 1px 4px rgba(0,0,0,0.04)'}),

    html.Button('Submit', id='submit-button', n_clicks=0),
    html.Div(id='submit-status', style={'marginTop': '20px', 'color': 'green'})
])



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
    percent_us_pop = (total_participants / US_POPULATION) * 100
    cumulative_total_events = len(df)  
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
    fig_momentum.add_trace(go.Scatter(x=dff_momentum['date'], y=dff_momentum['momentum'], mode='lines', name='Original Momentum'))
    fig_momentum.add_trace(go.Scatter(x=dff_momentum['date'], y=dff_momentum['alt_momentum'], mode='lines', name='7-Day Rolling Total'))
    fig_momentum.update_layout(title="Momentum of Dissent", height=270, margin=dict(t=30, b=10, l=10, r=10))  # Shorter, less padding

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
    if pathname == '/submit':
        return form_layout
    return dashboard_layout
    
@app.callback(
    Output('submit-status', 'children'),
    Input('submit-button', 'n_clicks'),
    State('input-email', 'value'),
    State('input-date', 'date'),
    State('input-locality', 'value'),
    State('input-title', 'value'),
    State('input-type', 'value'),
    State('input-summary', 'value'),
    State('input-size', 'value'),
    prevent_initial_call=True
)
def submit_event(n_clicks, email, date, locality, title, etype, summary, size):
    new_row = [email, date, locality, title, etype, summary, size]
    headers = ['Email', 'Date', 'Locality', 'Title', 'Event Type', 'Claims Summary', 'Size Estimate']

    with open('manual_submissions.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(headers)
        writer.writerow(new_row)

    return "Thanks! Your protest event was submitted."

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8050, debug=True)
