import dash
from dash import dcc, html, callback, Output, Input, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import base64
import io

def load_default_data():
    try:
        df = pd.read_csv("AverageDispatchTime.csv")
        return df
    except FileNotFoundError:
        print("Default file 'AverageDispatchTime.csv' not found. Please upload a CSV file.")
        return None

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            return None, 'Unsupported file type'
    except Exception as e:
        print(e)
        return None, 'There was an error processing this file.'

    return df, 'File successfully processed'

def suggest_marshallers(value, turn_time):
    """
    Suggests number of marshallers based on value and selected turn time
    Args:
        value (float): The input value to evaluate
        turn_time (int): Selected marshaller turn time (6-10 minutes)
    Returns:
        float: Suggested number of marshallers (1-5)
    """
    turn_time = int(turn_time)
    
    thresholds = {
        6: [10, 20, 30, 40, 50],
        7: [8, 17, 25, 34, 42],
        8: [7, 15, 22, 30, 37],
        9: [6, 13, 20, 26, 33],
        10: [6, 12, 18, 24, 30]
    }
    
    current_thresholds = thresholds.get(turn_time, thresholds[8])
    
    for i, threshold in enumerate(current_thresholds, 1):
        if value <= threshold:
            return float(i)
    return 5.0

app = dash.Dash(__name__)
server = app.server

# Load default data
default_df = load_default_data()
if default_df is not None:
    current_df = default_df
else:
    current_df = pd.DataFrame()

app.layout = html.Div(style={'backgroundColor': '#F9F9F9', 'padding': '20px'}, children=[
    html.H1('Kroger Forest Park FC Marshaller Report', style={'color': '#0D1F2D', 'fontFamily': 'Calibri'}),

    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),

    html.Div(id='output-data-upload'),

    html.Button("Download CSV", id="btn-download-csv"),
    dcc.Download(id="download-csv"),

    html.Div([
        html.Label('Select Days:', style={'fontFamily': 'Calibri', 'marginRight': '10px'}),
        dcc.Dropdown(
            id='day-selector',
            multi=True,
            style={'width': '100%', 'marginBottom': '20px'}
        )
    ]),
    
    html.Div([
        html.Label('Select Marshaller Turn Time (minutes):', 
                  style={'fontFamily': 'Calibri', 'marginRight': '10px'}),
        dcc.Dropdown(
            id='turn-time-dropdown',
            options=[{'label': f'{i} minutes', 'value': i} for i in range(6, 11)],
            value=8,
            style={'width': '200px', 'marginBottom': '20px'}
        )
    ]),
    
    html.Div([
    html.Label('Adjust Dispatch Count', style={'fontFamily': 'Calibri', 'marginLeft': '10px', 'fontWeight': 'bold'}),

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
    ]),
    
    dcc.Tabs([
        dcc.Tab(label='Table View', children=[
            dcc.Graph(id='marshaller-table', 
                     config={'displayModeBar': False}, 
                     style={'marginTop': '30px'})
        ]),
        dcc.Tab(label='Heatmap View', children=[
            dcc.Graph(id='marshaller-heatmap', 
                     config={'displayModeBar': False}, 
                     style={'marginTop': '30px'})
        ]),
        dcc.Tab(label='Bar Graph View', children=[
            html.Div([
                dcc.Dropdown(
                    id='bar-day-dropdown',
                    style={'width': '200px', 'marginBottom': '20px'}
                ),
                dcc.Graph(id='marshaller-bar', 
                         config={'displayModeBar': False})
            ])
        ]),
        dcc.Tab(label='Line Plot View', children=[
            dcc.Graph(id='marshaller-line-plot', 
                     config={'displayModeBar': False}, 
                     style={'marginTop': '30px'})
        ]),
        dcc.Tab(label='Stacked Bar Chart View', children=[
            dcc.Graph(id='marshaller-stacked-bar', 
                     config={'displayModeBar': False}, 
                     style={'marginTop': '30px'})
        ])
    ])
])

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
            return 'No file uploaded and default file not found.', [], [], [], None
        else:
            message = 'Using default file: AverageDispatchTime.csv'
    else:
        df, message = parse_contents(contents, filename)
        if df is not None:
            current_df = df
        else:
            return message, [], [], [], None

    options = [{'label': day, 'value': day} for day in current_df.columns[1:]]
    values = current_df.columns[1:].tolist()

    return message, options, values, options, values[0]

@app.callback(
    Output("download-csv", "data"),
    Input("btn-download-csv", "n_clicks"),
    prevent_initial_call=True,
)
def download_csv(n_clicks):
    if n_clicks is None:
        return dash.no_update
    return dcc.send_data_frame(current_df.to_csv, "marshaller_data.csv", index=False)

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
            fill_color='#93C47D',
            font=dict(color='black', size=12, family='Calibri'),
            align='center',
            line_color='black',
            height=30
        ),
        cells=dict(
            values=[df_filtered[col] for col in df_filtered.columns],
            fill_color=[['#F2F2F2' if i % 2 == 0 else 'white' 
                        for i in range(len(df_filtered))]],
            font=dict(color='black', size=11, family='Calibri'),
            align='center',
            line_color='black',
            height=25
        )
    )])
    
    # Update table layout
    table_fig.update_layout(
        title=f'Suggested QTY Marshallers (Increased by {slider_range[1]}, Turn Time: {turn_time} min)',
        title_font=dict(size=20, color='black', family='Calibri'),
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='white',
        plot_bgcolor='white',
        width=900,
        height=700
    )
    
    # Create heatmap
    heatmap_df = df_filtered.iloc[:-1, 1:-1]  # Exclude 'Time' column and 'Averages' column
    heatmap_fig = px.imshow(
        heatmap_df.values,
        labels=dict(x="Day of Week", y="Time of Day", 
                   color="Suggested Marshallers"),
        x=heatmap_df.columns,
        y=df_filtered['Time'][:-1],
        aspect="auto",
        color_continuous_scale="YlOrRd"
    )
    
    # Update heatmap layout
    heatmap_fig.update_layout(
        title=f'Heatmap of Suggested Marshallers (Increased by {slider_range[1]}, Turn Time: {turn_time} min)',
        title_font=dict(size=20, color='black', family='Calibri'),
        xaxis_title="Day of Week",
        yaxis_title="Time of Day",
        width=900,
        height=700
    )
    
    # Create horizontal bar graph
    if selected_day in df_filtered.columns:
        bar_data = df_filtered.iloc[:21, [0, df_filtered.columns.get_loc(selected_day)]]  # Get data from 12 AM to 8 PM for selected day
        bar_fig = go.Figure(data=[
            go.Bar(
                y=bar_data['Time'],
                x=bar_data[selected_day],
                orientation='h'
            )
        ])
        
        # Update bar graph layout
        bar_fig.update_layout(
            title=f'Suggested Marshallers for {selected_day} (Increased by {slider_range[1]}, Turn Time: {turn_time} min)',
            title_font=dict(size=20, color='black', family='Calibri'),
            xaxis_title="Number of Marshallers",
            yaxis_title="Time of Day",
            width=900,
            height=700,
            xaxis=dict(range=[0, 5])  # Set x-axis range from 0 to 5
        )
    else:
        bar_fig = go.Figure()
        bar_fig.update_layout(
            title="Please select a day from the dropdown",
            title_font=dict(size=20, color='black', family='Calibri'),
            width=900,
            height=700
        )
    
    # Create line plot
    line_data = df_filtered.iloc[:-1]  # Exclude the 'Daily Averages' row
    line_fig = px.line(line_data, x='Time', y=selected_days, 
                       labels={'value': 'Suggested Marshallers', 'variable': 'Day of Week'},
                       title=f'Suggested Marshallers Over Time (Increased by {slider_range[1]}, Turn Time: {turn_time} min)')
    
    # Update line plot layout
    line_fig.update_layout(
        xaxis_title="Time of Day",
        yaxis_title="Suggested Marshallers",
        yaxis=dict(range=[0, 5.5], dtick=1),  # Set y-axis range from 0 to 5 with tick every 1
        width=900,
        height=700,
        legend_title="Day of Week",
        font=dict(family="Calibri")
    )

    # Create stacked bar chart
    stacked_bar_data = df_filtered.iloc[:-1]  # Exclude the 'Daily Averages' row
    stacked_bar_fig = go.Figure()

    for day in selected_days:
        stacked_bar_fig.add_trace(go.Bar(
            x=stacked_bar_data['Time'],
            y=stacked_bar_data[day],
            name=day,
            hovertemplate='%{y:.1f}'
        ))

    # Update stacked bar chart layout
    stacked_bar_fig.update_layout(
        barmode='stack',
        title=f'Total Suggested Marshallers by Time of Day (Increased by {slider_range[1]}, Turn Time: {turn_time} min)',
        xaxis_title="Time of Day",
        yaxis_title="Total Suggested Marshallers",
        legend_title="Day of Week",
        font=dict(family="Calibri"),
        width=900,
        height=700
    )
    
    return table_fig, heatmap_fig, bar_fig, line_fig, stacked_bar_fig

if __name__ == '__main__':
    app.run_server