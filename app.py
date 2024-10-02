# app.py
import dash
from dash import dcc, html, callback, Output, Input
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


# Read CSV file and create input_data
def load_data():
    df = pd.read_csv("AverageDispatchTime.csv")
    input_data = [df.columns.tolist()] + df.values.tolist()
    return input_data

input_data = load_data()

# Function to calculate suggested marshallers
def suggest_marshallers(value):
    if value < 8:
        return 1.0  # Minimum of 2 marshallers
    elif value <= 8:
        return 1.0
    elif value <= 17:
        return 2.0
    elif value <= 25:
        return 3.0
    elif value <= 34:
        return 4.0
    elif value <= 42:
        return 5.0
    else:
        return 5.0
        # Maximum of 5 marshallers

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server  # Expose the server variable for WSGI

# Define the layout of the app
app.layout = html.Div(style={'backgroundColor': '#F9F9F9', 'padding': '20px'}, children=[
    html.H1('Route Level Dispatch Performance Data: Suggested Marshallers Per Hour', style={'color': '#0D1F2D', 'fontFamily': 'Calibri'}),
    html.H2('Increment Dispatch Count +5', style={'color': '#334E68', 'fontFamily': 'Calibri'}),
    dcc.RangeSlider(
        id='interval-slider',
        min=0,
        max=20,
        step=5,
        marks={i: f'+{i}' for i in range(0, 25, 5)},
        value=[0, 0],
        tooltip={"placement": "bottom", "always_visible": True},
        updatemode='drag'
    ),
    dcc.Tabs([
        dcc.Tab(label='Table View', children=[
            dcc.Graph(id='marshaller-table', config={'displayModeBar': False}, style={'marginTop': '30px'})
        ]),
        dcc.Tab(label='Heatmap View', children=[
            dcc.Graph(id='marshaller-heatmap', config={'displayModeBar': False}, style={'marginTop': '30px'})
        ]),
        dcc.Tab(label='Bar Graph View', children=[
            html.Div([
                dcc.Dropdown(
                    id='day-dropdown',
                    options=[
                        {'label': day, 'value': day} for day in ['Sun', 'Mon', 'Tues', 'Wed', 'Thur', 'Fri', 'Sat']
                    ],
                    value='Sun',
                    style={'width': '200px', 'marginBottom': '20px'}
                ),
                dcc.Graph(id='marshaller-bar', config={'displayModeBar': False})
            ])
        ])
    ])
])

@app.callback(
    Output('marshaller-table', 'figure'),
    Output('marshaller-heatmap', 'figure'),
    Output('marshaller-bar', 'figure'),
    Input('interval-slider', 'value'),
    Input('day-dropdown', 'value')
)
def update_graphs(slider_range, selected_day):
    # Create DataFrame
    df = pd.DataFrame(input_data[1:], columns=input_data[0])

    # Apply the slider value to increase the numerical levels
    for col in df.columns[1:]:
        df[col] = df[col].astype(float) + slider_range[1]

    # Apply the function to each cell
    for col in df.columns[1:]:
        df[col] = df[col].apply(suggest_marshallers)

    # Calculate averages for each time period
    df['Averages'] = df.iloc[:, 1:].mean(axis=1).round(1)

    # Calculate daily averages
    daily_averages = df.iloc[:, 1:-1].mean().round(1).tolist()
    daily_averages.append(round(np.mean(daily_averages), 1))

    # Add daily averages to the DataFrame
    df.loc[len(df)] = ['Daily Averages'] + daily_averages

    # Create the table
    table_fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df.columns),
            fill_color='#93C47D',
            font=dict(color='black', size=12, family='Calibri'),
            align='center',
            line_color='black',
            height=30
        ),
        cells=dict(
            values=[df[col] for col in df.columns],
            fill_color=[['#F2F2F2' if i % 2 == 0 else 'white' for i in range(len(df))]],
            font=dict(color='black', size=11, family='Calibri'),
            align='center',
            line_color='black',
            height=25
        )
    )])

    # Update table layout
    table_fig.update_layout(
        title=f'Suggested Marshallers (Increased by {slider_range[1]})',
        title_font=dict(size=20, color='black', family='Calibri'),
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='white',
        plot_bgcolor='white',
        width=900,
        height=700
    )

    # Create heatmap
    heatmap_df = df.iloc[:-1, 1:-1]  # Exclude 'Time' column and 'Averages' column
    heatmap_fig = px.imshow(heatmap_df.values,
                            labels=dict(x="Day of Week", y="Time of Day", color="Suggested Marshallers"),
                            x=heatmap_df.columns,
                            y=df['Time'][:-1],
                            aspect="auto",
                            color_continuous_scale="YlOrRd")

    # Update heatmap layout
    heatmap_fig.update_layout(
        title=f'Heatmap of Suggested Marshallers (Increased by {slider_range[1]})',
        title_font=dict(size=20, color='black', family='Calibri'),
        xaxis_title="Day of Week",
        yaxis_title="Time of Day",
        width=900,
        height=700
    )

    # Create horizontal bar graph
    bar_data = df.iloc[:21, [0, df.columns.get_loc(selected_day)]]  # Get data from 12 AM to 8 PM for selected day
    bar_fig = go.Figure(data=[
        go.Bar(
            y=bar_data['Time'],
            x=bar_data[selected_day],
            orientation='h'
        )
    ])

    # Update bar graph layout
    bar_fig.update_layout(
        title=f'Suggested Marshallers for {selected_day} (Increased by {slider_range[1]})',
        title_font=dict(size=20, color='black', family='Calibri'),
        xaxis_title="Number of Marshallers",
        yaxis_title="Time of Day",
        width=900,
        height=700,
        xaxis=dict(range=[0, 5])  # Set x-axis range from 0 to 5
    )

    return table_fig, heatmap_fig, bar_fig

if __name__ == '__main__':
    app.run_server(debug=True)