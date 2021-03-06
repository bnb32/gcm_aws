"""Dashboard app module"""
from ecrlgcm.environment import EnvironmentConfig
from ecrlgcm.postprocessing import PostProcessing
from ecrlgcm import app_argparse
from ecrlgcm.utilities.utilities import get_logger

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from dash.dependencies import Input, Output

logger = get_logger()


if __name__ == '__main__':

    parser = app_argparse()
    args = parser.parse_args()
    config = EnvironmentConfig(args.config)
    post_proc = PostProcessing(config)

    variable_dictionary = post_proc.variable_dictionary

    dropdown_options = []
    for k, v in variable_dictionary.items():
        dropdown_options.append({'label': v, 'value': k})

    app = dash.Dash(__name__)
    server = app.server

    app.layout = html.Div(
        [html.H6('Paleoclimate Modeling',
         style={'width': '100%', 'textAlign': 'center', 'font-size': '25px'}),
         html.Div(["Year (Ma BP): ",
                   dcc.Slider(id='land_year', min=post_proc.min_land_year,
                              max=post_proc.max_land_year,
                              step=10, value=0,
                              tooltip={"placement": "bottom",
                                       "always_visible": True})],
         style={'width': '20%', 'display': 'inline-block',
                'textAlign': 'center'}),
         html.Div(["Pressure Level (hPa): ",
                  dcc.Slider(id='plevel', min=0, max=1000, step=10, value=850,
                             tooltip={"placement": "bottom",
                                      "always_visible": True})],
         style={'width': '20%',
                'display': 'inline-block', 'textAlign': 'center'}),
         html.Div(["Fields: ",
                   dcc.Dropdown(id='field', options=dropdown_options,
                                value='CLOUDCOVER_CLUBB', multi=False)],
         style={'width': '30%', 'display': 'inline-block',
                'textAlign': 'center', 'height': '21px',
                'font-size': '18px', 'padding-left': '10px'}),
         html.Div(["High Resolution: ",
                   daq.ToggleSwitch(id='hires', value=False, label='OFF---ON',
                                    labelPosition="bottom")],
         style={'width': '10%', 'display': 'inline-block',
                'textAlign': 'center', 'font-size': '18px',
                'padding-left': '10px', 'padding-top': '10px'}),
         html.Br(),
         html.Div([dcc.Graph(id='my-output')],
         style={'width': '90%', 'textAlign': 'center',
                'padding-top': '-50px'})],
        style={"width": "1800px", "height": "940px", "display": "inline-block",
               "border": "3px #5c5c5c solid", "padding-top": "1px",
               "padding-right": "1px", "padding-bottom": "1px",
               "padding-left": "1px", "overflow": "hidden",
               'textAlign': 'center'})

    @app.callback(
        Output(component_id='my-output', component_property='figure'),
        [Input(component_id='land_year', component_property='value'),
         Input(component_id='plevel', component_property='value'),
         Input(component_id='field', component_property='value'),
         Input(component_id='hires', component_property='value'),
         ]
    )
    def update_output_div(land_year, plevel, field, hires):
        """Update render"""
        land_year = float(land_year)
        plevel = float(plevel)
        fig = post_proc.get_interactive_globe(land_year=land_year, field=field,
                                              plevel=plevel, fast=(not hires),
                                              time_avg=True)
        logger.info(f'year, plevel, field, fast: {land_year}, {plevel}, '
                    f'{field}, {not hires}')

        return fig

    app.run_server(host='0.0.0.0', port=8050, debug=True)
