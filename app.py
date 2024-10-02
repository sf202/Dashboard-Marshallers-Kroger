import dash
from dash import dcc, html, callback, Output, Input, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def load_data():
    df = pd.read_csv("AverageDispatchTime.csv")
    return df

df = load_data()

def suggest_marshallers(value, turn_time):
    """
    Suggests number of marshallers based on value and selected turn time
    Args:
        value (float): The input value to evaluate
        turn_time (int): Selected marshaller turn time (6-10 minutes)
    Returns:
        float: Suggested number of marshallers (1-5)
    """
    # Convert turn_time to int for comparison
    turn_time = int(turn_time)
    
    # Lookup dictionary for thresholds based on turn time
    thresholds = {
        6: [10, 20, 30, 40, 50],
        7: [8, 17, 25, 34, 42],
        8: [7, 15, 22, 30, 37],
        9: [6, 13, 20, 26, 33],
        10: [6, 12, 18, 24, 30]
    }
    
    # Get thresholds for selected turn time
    current_thresholds = thresholds.get(turn_time, thresholds[8])  # Default to 8 if invalid
    
    # Determine number of marshallers needed
    for i, threshold in enumerate(current_thresholds, 1):
        if value <= threshold:
            return float(i)
    return 5.0  # Maximum of 5 marshallers

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(style={'backgroundColor': '#F9F9F9', 'padding': '20px'}, children=[
    html.H1('Dispatch Performance Data', style={'color': '#0D1F2D', 'fontFamily': 'Calibri'}),
    html.H2('Adjust Dispatch Count', style={'color': '#334E68', 'fontFamily': 'Calibri'}),
    
    html.Div([
        html.Label('Select Days:', style={'fontFamily': 'Calibri', 'marginRight': '10px'}),
        dcc.Dropdown(
            id='day-selector',
            options=[{'label': day, 'value': day} for day in df.columns[1:]],
            value=df.columns[1:].tolist(),
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
                    options=[{'label': day, 'value': day} for day in df.columns[1:]],
                    value=df.columns[1],
                    style={'width': '200px', 'marginBottom': '20px'}
                ),
                dcc.Graph(id='marshaller-bar', 
                         config={'displayModeBar': False})
            ])
        ])
    ])
])

@app.callback(
    Output('marshaller-table', 'figure'),
    Output('marshaller-heatmap', 'figure'),
    Output('marshaller-bar', 'figure'),
    Input('interval-slider', 'value'),
    Input('bar-day-dropdown', 'value'),
    Input('turn-time-dropdown', 'value'),
    Input('day-selector', 'value')
)
def update_graphs(slider_range, selected_day, turn_time, selected_days):
    # Filter the DataFrame based on selected days
    df_filtered = df[['Time'] + selected_days]
    
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
    
    return table_fig, heatmap_fig, bar_fig

if __name__ == '__main__':
    app.run_server(debug=True)