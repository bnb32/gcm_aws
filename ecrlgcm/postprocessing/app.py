import plotly.graph_objects as go # or plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from ecrlgcm.postprocessing import get_interactive_globe

dropdown_options = ['RELHUM','TS']

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([html.H6('Paleoclimate Modeling',style={'width':'100%', 'textAlign': 'center','font-size': '25px',"padding-top": "1px"}), 
    html.Div([
        "Year (in Ma BP): ",
        dcc.Input(id='land_year', value=0, type='text')
    ],style={'width': '33%', 'display': 'inline-block','textAlign':'center','height':'10px'}),
    html.Div([
        "Field: ",
        dcc.Input(id='field', value='RELHUM', type='text')
    ],style={'width': '33%', 'display': 'inline-block','textAlign':'center'}),
    html.Div([
        "Pressure Level: ",
        dcc.Input(id='plevel', value=None, type='text')
    ],style={'width': '33%', 'display': 'inline-block','textAlign':'center'}),
    #html.H6('Feature Selection:',style={'width':'100%', 'textAlign': 'center','font-size': '26px'}),
    #dcc.Dropdown(id='fields',
    #options=dropdown_options,
    #value=['RELHUM'],
    #multi=False,
    #style={'width':'1800px', 'textAlign': 'center'}),
    html.Div([dcc.Graph(id='my-output')],style={'width':'90%','textAlign': 'center'}),

],style={"width": "1000px",
         "height": "900px",
         "display": "inline-block",
         "border": "3px #5c5c5c solid",
         "padding-top": "1px",
         "padding-left": "1px",
         "overflow": "hidden"})


@app.callback(
    Output(component_id='my-output', component_property='figure'),
    [Input(component_id='land_year', component_property='value'),
     Input(component_id='field', component_property='value'),
     Input(component_id='plevel', component_property='value'),
     ]
)
def update_output_div(land_year,field,plevel):
    try:
        land_year = float(land_year)
        plevel = float(plevel)
        fig = get_interactive_globe(land_year=land_year,field=field,plevel=plevel,fast=True)
        fig.update_layout(width=1000,height=710)
    except:
        fig = get_interactive_globe(fast=True)
        fig.update_layout(width=1000,height=710)
        
    return fig


#if __name__ == '__main__':
#    app.run_server(debug=True)


app.run_server(debug=True, use_reloader=False)
