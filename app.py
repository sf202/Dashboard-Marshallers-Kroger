import dash
from dash import dcc, html, callback, Output, Input, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import base64
import io
# Function to load default data from a CSV file

def load_default_data():
                # Attempt to read the default CSV file

    try:
        df = pd.read_csv("AverageDispatchTime.csv")
        return df
                # If the file is not found, print an error message and return None

    except FileNotFoundError:
        print("Default file 'AverageDispatchTime.csv' not found. Please upload a CSV file.")
        return None
    # Function to parse uploaded file contents


def parse_contents(contents, filename):
            # Split the content string to separate the data from the header

    content_type, content_string = contents.split(',')
            # Decode the base64 encoded string

    decoded = base64.b64decode(content_string)
    try:
                        # If the file is a CSV, read it into a pandas DataFrame

        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                                    # If the file is not a CSV, return an error message

        else:
            return None, 'Unsupported file type'
            # If successful, return the DataFrame and a success message
# Function to suggest number of marshallers based on value and turn time

    except Exception as e:
                        # If there's an error processing the file, print the error and return a message

        print(e)
        return None, 'There was an error processing this file.'

    return df, 'File successfully processed'

def suggest_marshallers(value, turn_time):
            # Convert turn_time to an integer

    """
    Suggests number of marshallers based on value and selected turn time
    Args:
        value (float): The input value to evaluate
        turn_time (int): Selected marshaller turn time (6-10 minutes)
    Returns:
        float: Suggested number of marshallers (1-5)
    """
    turn_time = int(turn_time)
        # Define thresholds for different turn times

    thresholds = {
        6: [10, 20, 30, 40, 50],
        7: [8, 17, 25, 34, 42],
        8: [7, 15, 22, 30, 37],
        9: [6, 13, 20, 26, 33],
        10: [6, 12, 18, 24, 30]
    }
        # Get the thresholds for the current turn time, defaulting to 8 minutes if not found

    current_thresholds = thresholds.get(turn_time, thresholds[8])

    for i, threshold in enumerate(current_thresholds, 1):
        if value <= threshold:
            return float(i)
    return 5.0
# Initialize the Dash app

app = dash.Dash(__name__)
server = app.server

# Load default data
default_df = load_default_data()
if default_df is not None:
    current_df = default_df
else:
    current_df = pd.DataFrame()
# Add title
# Kroger-inspired color scheme
COLORS = {
    'primary': '#004990',  # Kroger blue
    'secondary': '#84BD00',  # Kroger green
    'background': '#F5F5F5',
    'text': '#333333',
    'accent': '#FF7900',  # Orange for highlights
}
    # Add file upload component

app.layout = html.Div(style={'backgroundColor': COLORS['background'], 'minHeight': '100vh'}, children=[
    html.Div(style={'backgroundColor': COLORS['primary'], 'padding': '20px', 'color': 'white'}, children=[
        html.H1('Kroger Forest Park FC Marshaller Report', style={'fontFamily': 'Arial, sans-serif', 'marginBottom': '0'}),
        html.P('Optimize your marshaller allocation', style={'fontFamily': 'Arial, sans-serif', 'marginTop': '5px'})
    ]),
    
    html.Div(style={'padding': '20px', 'maxWidth': '1200px', 'margin': 'auto'}, children=[
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files', style={'color': COLORS['primary'], 'textDecoration': 'underline'})
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '20px 0',
                'backgroundColor': 'white'
            },
            multiple=False
        ),
    # Add div to display upload status

        html.Div(id='output-data-upload', style={'marginBottom': '20px', 'color': COLORS['text']}),
    # Add button to download CSV

        html.Button("Download CSV", id="btn-download-csv", style={
            'backgroundColor': COLORS['secondary'],
            'color': 'white',
            'border': 'none',
            'padding': '10px 20px',
            'borderRadius': '5px',
            'cursor': 'pointer',
            'marginBottom': '20px'
        }),
        dcc.Download(id="download-csv"),
    # Add dropdown to select days

        html.Div([
            html.Label('Select Days:', style={'fontFamily': 'Arial, sans-serif', 'marginRight': '10px', 'color': COLORS['text']}),
            dcc.Dropdown(
                id='day-selector',
                multi=True,
                style={'width': '100%', 'marginBottom': '20px'}
            )
        ]),
        # Add dropdown to select marshaller turn time

        html.Div([
            html.Label('Select Marshaller Turn Time (minutes):',
                      style={'fontFamily': 'Arial, sans-serif', 'marginRight': '10px', 'color': COLORS['text']}),
            dcc.Dropdown(
                id='turn-time-dropdown',
                options=[{'label': f'{i} minutes', 'value': i} for i in range(6, 11)],
                value=8,
                style={'width': '200px', 'marginBottom': '20px'}
            )
        ]),
        # Add slider to adjust dispatch count

        html.Div([
            html.Label('Adjust Dispatch Count', style={'fontFamily': 'Arial, sans-serif', 'marginLeft': '10px', 'fontWeight': 'bold', 'color': COLORS['text']}),
            dcc.RangeSlider(
                id='interval-slider',
                min=0,
                max=20,
                step=1,
                marks={i: f'+{i}' for i in range(0, 21, 5)},
                value=[0, 0],
                tooltip={"placement": "bottom", "always_visible": True},
                updatemode='drag'
            ),
        ], style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '5px', 'marginBottom': '20px'}),
        # Add tabs for different views

        dcc.Tabs(style={'fontFamily': 'Arial, sans-serif'}, children=[
            dcc.Tab(label='Table View', style={'backgroundColor': COLORS['primary'], 'color': 'white'}, selected_style={'backgroundColor': COLORS['secondary'], 'color': 'white'}, children=[
                dcc.Graph(id='marshaller-table', config={'displayModeBar': False}, style={'marginTop': '30px'})
            ]),
            dcc.Tab(label='Heatmap View', style={'backgroundColor': COLORS['primary'], 'color': 'white'}, selected_style={'backgroundColor': COLORS['secondary'], 'color': 'white'}, children=[
                dcc.Graph(id='marshaller-heatmap', config={'displayModeBar': False}, style={'marginTop': '30px'})
            ]),
            dcc.Tab(label='Bar Graph View', style={'backgroundColor': COLORS['primary'], 'color': 'white'}, selected_style={'backgroundColor': COLORS['secondary'], 'color': 'white'}, children=[
                html.Div([
                    dcc.Dropdown(
                        id='bar-day-dropdown',
                        style={'width': '200px', 'marginBottom': '20px', 'marginTop': '20px'}
                    ),
                    dcc.Graph(id='marshaller-bar', config={'displayModeBar': False})
                ])
            ]),
            dcc.Tab(label='Line Plot View', style={'backgroundColor': COLORS['primary'], 'color': 'white'}, selected_style={'backgroundColor': COLORS['secondary'], 'color': 'white'}, children=[
                dcc.Graph(id='marshaller-line-plot', config={'displayModeBar': False}, style={'marginTop': '30px'})
            ]),
            dcc.Tab(label='Stacked Bar Chart View', style={'backgroundColor': COLORS['primary'], 'color': 'white'}, selected_style={'backgroundColor': COLORS['secondary'], 'color': 'white'}, children=[
                dcc.Graph(id='marshaller-stacked-bar', config={'displayModeBar': False}, style={'marginTop': '30px'})
            ])
        ])
    ])
])
# Callback to update data when a file is uploaded

@app.callback(
    Output('output-data-upload', 'children'),
    Output('day-selector', 'options'),
    Output('day-selector', 'value'),
    Output('bar-day-dropdown', 'options'),
    Output('bar-day-dropdown', 'value'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    global current_df

    if contents is None:
        if current_df.empty:
                                    # No file uploaded and no default data

            return 'No file uploaded and default file not found.', [], [], [], None
        else:
            message = 'Using default file: AverageDispatchTime.csv'
    else:
                                # Using default data
                # Process uploaded file

        df, message = parse_contents(contents, filename)
        if df is not None:
            current_df = df
        else:
            return message, [], [], [], None
    # Create options for dropdowns

    options = [{'label': day, 'value': day} for day in current_df.columns[1:]]
    values = current_df.columns[1:].tolist()

    return message, options, values, options, values[0]
# Callback to download CSV file

@app.callback(
    Output("download-csv", "data"),
    Input("btn-download-csv", "n_clicks"),
    prevent_initial_call=True,
)
def download_csv(n_clicks):
    if n_clicks is None:
        return dash.no_update
            # Return the current DataFrame as a CSV file

    return dcc.send_data_frame(current_df.to_csv, "marshaller_data.csv", index=False)
# Main callback to update all graphs

@app.callback(
    Output('marshaller-table', 'figure'),
    Output('marshaller-heatmap', 'figure'),
    Output('marshaller-bar', 'figure'),
    Output('marshaller-line-plot', 'figure'),
    Output('marshaller-stacked-bar', 'figure'),
    Input('interval-slider', 'value'),
    Input('bar-day-dropdown', 'value'),
    Input('turn-time-dropdown', 'value'),
    Input('day-selector', 'value')
)
def update_graphs(slider_range, selected_day, turn_time, selected_days):
    if current_df.empty:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # Filter the DataFrame based on selected days
    df_filtered = current_df[['Time'] + selected_days]

    # Apply the slider value to increase the numerical levels
    for col in df_filtered.columns[1:]:
        df_filtered[col] = df_filtered[col].astype(float) + slider_range[1]

    # Apply the suggest_marshallers function to each cell with the selected turn time
    for col in df_filtered.columns[1:]:
        df_filtered[col] = df_filtered[col].apply(lambda x: suggest_marshallers(x, turn_time))

    # Calculate averages for each time period
    df_filtered['Averages'] = df_filtered.iloc[:, 1:].mean(axis=1).round(1)

    # Calculate daily averages
    daily_averages = df_filtered.iloc[:, 1:-1].mean().round(1).tolist()
    daily_averages.append(round(np.mean(daily_averages), 1))

    # Add daily averages to the DataFrame
    df_filtered.loc[len(df_filtered)] = ['Daily Averages'] + daily_averages

    # Create the table
    table_fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df_filtered.columns),
            fill_color=COLORS['primary'],
            font=dict(color='white', size=12, family='Arial, sans-serif'),
            align='center',
            line_color='white',
            height=40
        ),
        cells=dict(
            values=[df_filtered[col] for col in df_filtered.columns],
            fill_color=[['white', COLORS['background']] * (len(df_filtered) // 2 + 1)],
            font=dict(color=COLORS['text'], size=11, family='Arial, sans-serif'),
            align='center',
            line_color='white',
            height=30
        )
    )])

    # Update table layout
    table_fig.update_layout(
        title=f'Suggested QTY Marshallers (Increased by {slider_range[1]}, Turn Time: {turn_time} min)',
        title_font=dict(size=20, color=COLORS['primary'], family='Arial, sans-serif'),
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='white',
        plot_bgcolor='white',
        width=1200,
        height=700
    )

   # Create heatmap
    heatmap_df = df_filtered.iloc[:-1, 1:-1]  # Exclude 'Time' column and 'Averages' column
    heatmap_fig = px.imshow(
        heatmap_df.values,
        labels=dict(x="Day of Week", y="Time of Day", color="Suggested Marshallers"),
        x=heatmap_df.columns,
        y=df_filtered['Time'][:-1],
        aspect="auto",
        color_continuous_scale='YlOrRd'  # Yellow-Orange-Red color scale
    )

    # Update heatmap layout
    heatmap_fig.update_layout(
        title=f'Heatmap of Suggested Marshallers (Increased by {slider_range[1]}, Turn Time: {turn_time} min)',
        title_font=dict(size=20, color=COLORS['primary'], family='Arial, sans-serif'),
        xaxis_title="Day of Week",
        yaxis_title="Time of Day",
        width=1200,
        height=700,
        font=dict(family="Arial, sans-serif", color=COLORS['text'])
    )
    # Create horizontal bar graph
    if selected_day in df_filtered.columns:
        bar_data = df_filtered.iloc[:21, [0, df_filtered.columns.get_loc(selected_day)]]  # Get data from 12 AM to 8 PM for selected day
        bar_fig = go.Figure(data=[
            go.Bar(
                y=bar_data['Time'],
                x=bar_data[selected_day],
                orientation='h',
                marker_color=COLORS['secondary']
            )
        ])

        # Update bar graph layout
        bar_fig.update_layout(
            title=f'Suggested Marshallers for {selected_day} (Increased by {slider_range[1]}, Turn Time: {turn_time} min)',
            title_font=dict(size=20, color=COLORS['primary'], family='Arial, sans-serif'),
            xaxis_title="Number of Marshallers",
            yaxis_title="Time of Day",
            width=1200,
            height=700,
            xaxis=dict(range=[0, 5]),
            font=dict(family="Arial, sans-serif", color=COLORS['text'])
        )
    else:
        bar_fig = go.Figure()
        bar_fig.update_layout(
            title="Please select a day from the dropdown",
            title_font=dict(size=20, color=COLORS['primary'], family='Arial, sans-serif'),
            width=1200,
            height=700
        )

    # Create line plot
    line_data = df_filtered.iloc[:-1]  # Exclude the 'Daily Averages' row
    line_fig = px.line(line_data, x='Time', y=selected_days,
                       labels={'value': 'Suggested Marshallers', 'variable': 'Day of Week'},
                       title=f'Suggested Marshallers Over Time (Increased by {slider_range[1]}, Turn Time: {turn_time} min)',
                       color_discrete_sequence=[COLORS['primary'], COLORS['secondary'], COLORS['accent']])

    # Update line plot layout
    line_fig.update_layout(
        xaxis_title="Time of Day",
        yaxis_title="Suggested Marshallers",
        yaxis=dict(range=[0, 5.5], dtick=1),  # Set y-axis range from 0 to 5 with tick every 1
        width=1200,
        height=700,
        legend_title="Day of Week",
        font=dict(family="Arial, sans-serif", color=COLORS['text'])
    )

    # Create stacked bar chart
    stacked_bar_data = df_filtered.iloc[:-1]  # Exclude the 'Daily Averages' row
    stacked_bar_fig = go.Figure()

    for i, day in enumerate(selected_days):
        stacked_bar_fig.add_trace(go.Bar(
            x=stacked_bar_data['Time'],
            y=stacked_bar_data[day],
            name=day,
            hovertemplate='%{y:.1f}',
            marker_color=[COLORS['primary'], COLORS['secondary'], COLORS['accent']][i % 3]
        ))

    # Update stacked bar chart layout
    stacked_bar_fig.update_layout(
        barmode='stack',
        title=f'Total Suggested Marshallers by Time of Day (Increased by {slider_range[1]}, Turn Time: {turn_time} min)',
        xaxis_title="Time of Day",
        yaxis_title="Total Suggested Marshallers",
        legend_title="Day of Week",
        font=dict(family="Arial, sans-serif", color=COLORS['text']),
        width=1200,
        height=700
    )

    return table_fig, heatmap_fig, bar_fig, line_fig, stacked_bar_fig

if __name__ == '__main__':
    app.run_server(debug=True)