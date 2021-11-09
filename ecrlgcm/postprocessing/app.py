import plotly.graph_objects as go # or plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from dash.dependencies import Input, Output
from ecrlgcm.postprocessing import get_interactive_globe, variable_dictionary
from ecrlgcm.misc import get_logger

logger = get_logger()

dropdown_options = []
for k,v in variable_dictionary.items():
    dropdown_options.append({'label':v,'value':k})

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([html.H6('Paleoclimate Modeling',style={'width':'100%', 'textAlign': 'center','font-size': '25px'}),
    html.Div(["Year (Ma BP): ",
        dcc.Slider(
        id='land_year',
        min=0,
        max=500,
        step=10,
        value=0,
        tooltip={"placement": "bottom", "always_visible": True})],style={'width':'20%','display': 'inline-block', 'textAlign':'center'}),
    html.Div(["Pressure Level (hPa): ",
        dcc.Slider(
        id='plevel',
        min=0,
        max=1000,
        step=10,
        value=850,
        tooltip={"placement": "bottom", "always_visible": True})],style={'width':'20%','display': 'inline-block', 'textAlign':'center'}),
    html.Div(["Fields: ",
        dcc.Dropdown(id='field',
            options=dropdown_options,
            value='CLOUDCOVER_CLUBB',#'RELHUM',
            multi=False)],
        style={'width': '30%', 'display': 'inline-block','textAlign':'center','height':'21px','font-size':'18px','padding-left':'10px'}),
    html.Div(["High Resolution: ",
        daq.ToggleSwitch(id='hires',
                         value=False,
                         label='OFF---ON',
                         labelPosition="bottom")],
        style={'width': '10%', 'display': 'inline-block','textAlign':'center','font-size':'18px','padding-left':'10px','padding-top':'10px'}),
    html.Br(),
    html.Div([dcc.Graph(id='my-output')],style={'width':'90%','textAlign': 'center','padding-top':'-50px'}),

],style={"width": "1800px",
         "height": "940px",
         "display": "inline-block",
         "border": "3px #5c5c5c solid",
         "padding-top": "1px",
         "padding-right": "1px",
         "padding-bottom": "1px",
         "padding-left": "1px",
         "overflow": "hidden",
         'textAlign':'center'})


@app.callback(
    Output(component_id='my-output', component_property='figure'),
    [Input(component_id='land_year', component_property='value'),
     Input(component_id='plevel', component_property='value'),
     Input(component_id='field', component_property='value'),
     Input(component_id='hires', component_property='value'),
     ]
)
def update_output_div(land_year,plevel,field,hires):
    land_year = float(land_year)
    plevel = float(plevel)
    fig = get_interactive_globe(land_year=land_year,
                                field=field,
                                plevel=plevel,
                                fast=(not hires))
    logger.info(f'year,plevel,field,fast: {land_year},{plevel},{field},{not hires}')

    return fig

#if __name__ == '__main__':
#    app.run_server(debug=True)

app.run_server(host='0.0.0.0', port=8050, debug=True)
