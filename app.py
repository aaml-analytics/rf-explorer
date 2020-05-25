import base64
import io
import math
import dash_table
import pandas as pd
import numpy as np
from dash_table.Format import Scheme, Format
from scipy import stats
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly_express as px
from dash.dependencies import Input, Output, State
import urllib.parse

external_stylesheets = [dbc.themes.BOOTSTRAP, 'https://codepen.io/chriddyp/pen/bWLwgP.css',
                        "https://codepen.io/sutharson/pen/ZEbqopm.css"]

SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}
tabs_styles = {'height': '40px', 'font-family': 'Arial', 'fontSize': 14}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '3px solid #4a4a4a',
    'borderBottom': '1px solid #d6d6d6 ',
    'backgroundColor': '#f6f6f6',
    'color': '#4a4a4a',
    # 'fontColor': '#004a4a',
    'fontWeight': 'bold',
    'padding': '6px'
}

tab_mini_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold',
    'width': '200px',
    'color': '#000000',
    'fontColor': '#000000',
}

tab_mini_style_2 = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold',
    'width': '400px',
    'color': '#000000',
    'fontColor': '#000000',
}

tab_mini_selected_style = {
    'borderTop': '3px solid #5e5e5e',
    'borderBottom': '1px solid #d6d6d6 ',
    'backgroundColor': '#5e5e5e',
    'color': '#ffffff',
    # 'fontColor': '#004a4a',
    'fontWeight': 'bold',
    'padding': '6px',
    'width': '200px'
}

tab_mini_selected_style_2 = {
    'borderTop': '3px solid #5e5e5e',
    'borderBottom': '1px solid #d6d6d6 ',
    'backgroundColor': '#5e5e5e',
    'color': '#ffffff',
    # 'fontColor': '#004a4a',
    'fontWeight': 'bold',
    'padding': '6px',
    'width': '400px'
}

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server



app.layout = html.Div(
    [
        html.Div([
            html.Img(
                src='https://raw.githubusercontent.com/aaml-analytics/mof-explorer/master/UOC.png',
                height='35', width='135', style={'display': 'inline-block', 'padding-left': '1%'}),
            html.Img(src='https://raw.githubusercontent.com/aaml-analytics/mof-explorer/master/A2ML-logo.png',
                     height='50', width='125',
                     style={'float': 'right', 'display': 'inline-block', 'padding-right': '2%'}),
            html.H1("Random Forest Visualisation Tools",
                    style={'display': 'inline-block', 'padding-left': '20%', 'text-align': 'center', 'fontSize': 36,
                           'color': 'white', 'font-family': 'Georgia'}),
            html.H1("...", style={'fontColor': '#3c3c3c', 'fontSize': 6})
        ], style={'backgroundColor': '#00113d'}),
        html.Div([html.A('Refresh', href='/')], style={}),
        html.Div([
            html.H2("Upload Data", style={'fontSize': 24, 'font-family': 'Arial', 'color': '#00113d'}, ),
            html.H3("Upload .txt, .csv or .xls files to starting exploring data...", style={'fontSize': 16,
                                                                                            'font-family': 'Arial'}),
            dcc.Store(id='csv-data', storage_type='session', data=None),
            html.Div([dcc.Upload(
                id='data-table-upload',
                children=html.Div([html.Button('Upload File')],
                                  style={'height': "60px", 'borderWidth': '1px',
                                         'borderRadius': '5px',
                                         'textAlign': 'center',

                                         }),
                multiple=False
            ),
                html.Div(id='output-data-upload'),
            ]), ], style={'display': 'inline-block', 'padding-left': '1%', }),
        dcc.Store(id='memory-output'),
        dcc.Store(id='memory-output-2'),
        dcc.Store(id='memory-output-3'),
        dcc.Store(id='memory-output-4'),
        dcc.Tabs([dcc.Tab(label='Preparing data for RF', style=tab_style, selected_style=tab_selected_style,
                          children=[
                              dcc.Tabs(id='sub-tabs1', style=tabs_styles,
                                       children=[dcc.Tab(label='Selecting Data for RF', style=tab_mini_style,
                                                         selected_style=tab_mini_selected_style,
                                                         children=[html.Div([html.P("Selecting Features")],
                                                                            style={'padding-left': '1%',
                                                                                   'font-weight': 'bold'}),
                                                                   html.Div([
                                                                       html.P(
                                                                           "Select variables that you would like as features/descriptors in your analysis:"),
                                                                       html.Label(
                                                                           [
                                                                               "Note: Only input numerical variables (non-numerical variables have already "
                                                                               "been removed from your dataframe).",
                                                                               dcc.Dropdown(id='feature-input',
                                                                                            multi=True,
                                                                                            )])
                                                                   ], style={'padding': 10, 'padding-left': '1%'}),
                                                                   html.Div([
                                                                       html.Label(["Table Overview of Features"])
                                                                   ], style={'font-weight': 'bold',
                                                                             'padding-left': '1%'}),
                                                                   html.Div([
                                                                       dash_table.DataTable(id='data-table-features',
                                                                                            editable=False,
                                                                                            filter_action='native',
                                                                                            sort_action='native',
                                                                                            sort_mode='multi',
                                                                                            selected_columns=[],
                                                                                            selected_rows=[],
                                                                                            page_action='native',
                                                                                            page_current=0,
                                                                                            page_size=20,
                                                                                            style_data={
                                                                                                'height': 'auto'},
                                                                                            style_table={
                                                                                                'overflowX': 'scroll',
                                                                                                'maxHeight': '300px',
                                                                                                'overflowY': 'scroll'},
                                                                                            style_cell={
                                                                                                'minWidth': '0px',
                                                                                                'maxWidth': '220px',
                                                                                                'whiteSpace': 'normal',
                                                                                            }
                                                                                            ),
                                                                       html.Div(id='data-table-contrib-container'),
                                                                   ], style={'padding': 20}),
                                                                   html.Div([html.P("Selecting target variable")],
                                                                            style={'padding-left': '1%',
                                                                                   'font-weight': 'bold'}),
                                                                   html.Div([
                                                                       html.P(
                                                                           " Select target variable (what you would like to predict) in your analysis:"),
                                                                       html.Label(
                                                                           [
                                                                               "Note: Only input numerical variables. Non-numerical variables have already "
                                                                               "been removed from your dataframe.",
                                                                               dcc.Dropdown(id='feature-target',
                                                                                            multi=False,
                                                                                            )])
                                                                   ], style={'padding': 10, 'padding-left': '1%'}), ]),
                                                 dcc.Tab(label='Feature Correlation', style=tab_mini_style,
                                                         selected_style=tab_mini_selected_style,
                                                         children=[html.Div([
                                                             # feature correlation analysis with target
                                                             html.Div([dcc.Graph(id='feature-heatmap')
                                                                       ], style={
                                                                 'padding-right': '23%',
                                                                 'padding-left': '17%'}),
                                                             html.Div([html.Label(["Select color scale:",
                                                                                   dcc.RadioItems(
                                                                                       id='colorscale',
                                                                                       options=[{'label': i, 'value': i}
                                                                                                for i in
                                                                                                ['Viridis', 'Plasma']],
                                                                                       value='Plasma'
                                                                                   )]),
                                                                       html.P(
                                                                           "Note that outliers have been removed from data")
                                                                       ], style={
                                                                 'padding-left': '1%'}),
                                                         ])
                                                         ]), ])]),

                  dcc.Tab(label='Hyperparameter Tuning', style=tab_style,
                          selected_style=tab_selected_style,
                          children=[
                              html.Div([html.P("Random Search Cross Validation")], style={'padding-left': '1%',
                                                                                          'font-weight': 'bold',
                                                                                          'fontSize': 22}),
                              html.Div([html.P("Random Search Hyperparameter Grid:")],
                                       style={'padding-left': '1%', 'font-weight': 'bold'}),
                              html.Div([html.Label([""
                                                    ]),
                                        html.P(
                                            "{'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],"
                                        ),
                                        html.P("'max_features': ['auto', 'sqrt'], "),
                                        html.P("'max_depth': [10, 20, 30, 40, 50, 60, 70, "
                                               "80, 90, 100, 110, None],"),
                                        html.P("'min_samples_split': [2, 5, 10], "),
                                        html.P("'min_samples_leaf': [1, 2, 4], "),
                                        html.P("'bootstrap': [False, True]}"),
                                        html.P("In Random Search Cross Validation, "
                                               "the algorithm will choose a different combination "
                                               "of the features on each iteration. Altogether, there are 2 * 12 * 2 * 3 * 3 * 10 = 4320 settings."
                                               " Choose an appropriate number of iterations. When number of iterations = 100,"
                                               " it takes 30 mins to finish job. As the number of iterations increases, "
                                               "so does the time taken to complete the job."),
                                        html.P('Input number of iterations where 0 < input < 4320:')
                                        ], style={
                                  'float': 'left',
                                  'padding-left': '1%'}
                                       ),
                              html.Div([
                                  html.Div(dcc.Input(id='input_number', type='number')),
                                  html.Button('Submit', id='button'),
                                  html.Div(id='output-container-button',
                                           children='Input number of iterations where 0 < input < 4320:')
                              ], style={'padding-left': '1%'}),
                              # html.Div(
                              #     [
                              #         html.Label([
                              #             "Input number of iterations where"
                              #             " 0 < input < 4320:",
                              #             dcc.Input(
                              #                 id="input_number",
                              #                 type="number",
                              #                 placeholder="input number",
                              #             ), html.Div(id="out-all-types")])
                              #     ], style={'padding-left': '1%'}
                              # ),
                              html.Div([html.P("Best parameters from Random search using 5 fold Cross Validation:")],
                                       style={'padding-left': '1%', 'font-weight': 'bold'}),
                              html.Div([html.Label([""
                                                       , html.Div(
                                      id='RandomisedSearchCV-container')])
                                        ], style={
                                  'width': '100%',
                                  'float': 'left',
                                  'padding-left': '1%'}
                                       ),
                              html.Div([
                                  html.P(["Evaluating Random Search model ",
                                          html.Span(
                                              "(with base model):",
                                              id="tooltip-target",
                                              style={"textDecoration": "underline", "cursor": "pointer"})
                                          ]),
                                  dbc.Tooltip(
                                      "{'bootstrap': True, 'max_depth': None, 'max_features': 'auto', "
                                      "'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100,"
                                      " 'random_state': 42}",
                                      target="tooltip-target",
                                  ),
                                  html.Div(id='Randomised-metrics-container')
                              ], style={
                                  'padding-left': '1%',
                                  'font-weight': 'bold', }
                              ),
                              html.Div([html.P("Grid Search Cross Validation")], style={'padding-left': '1%',
                                                                                        'font-weight': 'bold',
                                                                                        'fontSize': 22}),
                              html.Div(
                                  [html.P(
                                      "Grid Search Hyperparameter Grid (using best random search hyperparameters):"),
                                  ],
                                  style={'padding-left': '1%', 'font-weight': 'bold'}),
                              html.Div([html.Label(["", html.Div(id='GridSearchCV-container')])
                                        ], style={
                                  'width': '100%',
                                  'float': 'left',
                                  'padding-left': '1%'}
                                       ),
                              html.Div([html.P("In Grid Search Cross Validation, another grid is made using the best"
                                               " parameter values from the Random Search model. Altogether, "
                                               "there are 243 possible settings. Here all possible settings"
                                               " are tested.")], style={'padding-left': '1%'}),
                              html.Div([html.P("Best parameters from Grid search using 5 fold Cross Validation:")],
                                       style={'padding-left': '1%', 'font-weight': 'bold'}),
                              html.Div([html.Label(["", html.Div(
                                  id='final-hyper-parameter-container')])
                                        ], style={
                                  'width': '100%',
                                  'float': 'left',
                                  'padding-left': '1%'}
                                       ),
                              html.Div([
                                  html.P(["Evaluating Grid Search model ",
                                          html.Span(
                                              "(with base model):",
                                              id="tooltip-target-grid",
                                              style={"textDecoration": "underline", "cursor": "pointer"})
                                          ]),
                                  dbc.Tooltip(
                                      "{'bootstrap': True, 'max_depth': None, 'max_features': 'auto', "
                                      "'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100,  "
                                      "'random_state': 42}",
                                      target="tooltip-target-grid",
                                  ),
                                  html.Div(id='Grid-improvement-metrics-container')
                              ], style={
                                  'padding-left': '1%',
                                  'font-weight': 'bold', }
                              ),
                          ]),
                  dcc.Tab(label='Output Plots', style=tab_style, selected_style=tab_selected_style,
                          children=[dcc.Tabs(id='sub-tabs2', style=tabs_styles,
                                             children=[
                                                 dcc.Tab(label='Results from Hyperparameter tuning',
                                                         style=tab_mini_style_2,
                                                         selected_style=tab_mini_selected_style_2,
                                                         children=[
                                                             html.Div([dcc.Graph(id='parity-plot')],
                                                                      style={'width': '40%',
                                                                             'display': 'inline-block'}),
                                                             html.Div([dcc.Graph(id='feature-importance')],
                                                                      style={'width': '50%', 'display': 'inline-block',
                                                                             'padding-left': '5%'}),
                                                             html.Div(
                                                                 [dash_table.DataTable(id='performance-metrics-table')],
                                                                 style={'display': 'inline-block',
                                                                        'padding-left': '38%', }),
                                                             html.Div([], style={'padding': 5})
                                                         ]),
                                                 dcc.Tab(label='Random Forest Output', style=tab_mini_style_2,
                                                         selected_style=tab_mini_selected_style_2,
                                                         children=[
                                                             dcc.Tabs(id='sub-tabs3', style=tabs_styles,
                                                                      children=[
                                                                          dcc.Tab(label='Plots', style=tab_mini_style,
                                                                                  selected_style=tab_mini_selected_style,
                                                                                  children=[html.Div([html.Label([
                                                                                      "Input test split here. Value "
                                                                                      "entered should be between 0.0 "
                                                                                      "and 1.0 and "
                                                                                      "represent the proportion of the"
                                                                                      " data set to include in the "
                                                                                      "test set. i.e. 0.3 equates to "
                                                                                      "30% of the data being used in "
                                                                                      "the test set: "
                                                                                      "",
                                                                                      dcc.Input(
                                                                                          id="input-number-test",
                                                                                          type="number",
                                                                                          placeholder="input number",
                                                                                      ), html.Div(
                                                                                          id="out-all-test-split")]), ],
                                                                                      style={'padding-left': '1%'}),
                                                                                      html.Div([dcc.Graph(
                                                                                          id='parity-plot-final')],
                                                                                          style={'width': '40%',
                                                                                                 'display':
                                                                                                     'inline-block'}),
                                                                                      html.Div([dcc.Graph(
                                                                                          id='feature-importance-final')],
                                                                                          style={'width': '50%',
                                                                                                 'display': 'inline-block',
                                                                                                 'padding-left': '5%'}),
                                                                                      html.Div(
                                                                                          [dash_table.DataTable(
                                                                                              id='performance-metrics-table-final')],
                                                                                          style={
                                                                                              'display': 'inline-block',
                                                                                              'padding-left': '38%'}),
                                                                                      html.Div([],
                                                                                               style={'padding': 5})]),
                                                                          dcc.Tab(label='Error Plot',
                                                                                  style=tab_mini_style,
                                                                                  selected_style=tab_mini_selected_style,
                                                                                  children=[
                                                                                      html.Div([
                                                                                          dcc.Graph(id='error-dist')
                                                                                      ], style={'width': '78%',
                                                                                                'padding-left': '12%'}),
                                                                                  ])
                                                                      ]),
                                                         ]),
                                             ]),
                                    ]),
                  dcc.Tab(label='Data tables', style=tab_style, selected_style=tab_selected_style,
                          children=[html.Div([
                              html.Div([
                                  html.Label(["Correlation between Features"])
                              ], style={'font-weight': 'bold'}),
                              html.Div([
                                  dash_table.DataTable(id='data-table-correlation',
                                                       editable=False,
                                                       filter_action='native',
                                                       sort_action='native',
                                                       sort_mode='multi',
                                                       selected_columns=[],
                                                       selected_rows=[],
                                                       page_action='native',
                                                       column_selectable='single',
                                                       page_current=0,
                                                       page_size=20,
                                                       style_data={'height': 'auto'},
                                                       style_table={'overflowX': 'scroll',
                                                                    'maxHeight': '300px',
                                                                    'overflowY': 'scroll'},
                                                       style_cell={
                                                           'minWidth': '0px', 'maxWidth': '220px',
                                                           'whiteSpace': 'normal',
                                                       }
                                                       ),
                                  html.Div(id='data-table-correlation-container'),
                              ]),
                              html.Div([html.A(
                                  'Download Feature Correlation data',
                                  id='download-link-correlation',
                                  href="",
                                  target="_blank"
                              )]),

                          ], style={'padding': 20}),
                              html.Div([
                                  html.Div([
                                      html.Label(["Performance Metrics"])
                                  ], style={'font-weight': 'bold'}),
                                  html.Div([
                                      dash_table.DataTable(id='data-table-performance-met',
                                                           editable=False,
                                                           filter_action='native',
                                                           sort_action='native',
                                                           sort_mode='multi',
                                                           selected_columns=[],
                                                           selected_rows=[],
                                                           page_action='native',
                                                           column_selectable='single',
                                                           page_current=0,
                                                           page_size=20,
                                                           style_data={'height': 'auto'},
                                                           style_table={'overflowX': 'scroll',
                                                                        'maxHeight': '300px',
                                                                        'overflowY': 'scroll'},
                                                           style_cell={
                                                               'minWidth': '0px', 'maxWidth': '220px',
                                                               'whiteSpace': 'normal',
                                                           }
                                                           ),
                                      html.Div(id='data-table-performance-met-container'),
                                  ]),
                                  html.Div([html.A(
                                      'Download Performance Metrics data',
                                      id='download-link-performance-met',
                                      href="",
                                      target="_blank"
                                  )]),

                              ], style={'padding': 20}),
                              html.Div([
                                  html.Div([
                                      html.Label(["Observed and Predicted values from Random Forest"])
                                  ], style={'font-weight': 'bold'}),
                                  html.Div([
                                      dash_table.DataTable(id='data-table-RF',
                                                           editable=False,
                                                           filter_action='native',
                                                           sort_action='native',
                                                           sort_mode='multi',
                                                           selected_columns=[],
                                                           selected_rows=[],
                                                           page_action='native',
                                                           column_selectable='single',
                                                           page_current=0,
                                                           page_size=20,
                                                           style_data={'height': 'auto'},
                                                           style_table={'overflowX': 'scroll',
                                                                        'maxHeight': '300px',
                                                                        'overflowY': 'scroll'},
                                                           style_cell={
                                                               'minWidth': '0px', 'maxWidth': '220px',
                                                               'whiteSpace': 'normal',
                                                           }
                                                           ),
                                      html.Div(id='data-table-RF-container'),
                                  ]),
                                  html.Div([html.A(
                                      'Download Random Forest data',
                                      id='download-link-RF',
                                      href="",
                                      target="_blank"
                                  )]),

                              ], style={'padding': 20}),
                              html.Div([
                                  html.Div([
                                      html.Label(["Feature Importance"])
                                  ], style={'font-weight': 'bold'}),
                                  html.Div([
                                      dash_table.DataTable(id='data-table-feat-imp',
                                                           editable=False,
                                                           filter_action='native',
                                                           sort_action='native',
                                                           sort_mode='multi',
                                                           selected_columns=[],
                                                           selected_rows=[],
                                                           page_action='native',
                                                           column_selectable='single',
                                                           page_current=0,
                                                           page_size=20,
                                                           style_data={'height': 'auto'},
                                                           style_table={'overflowX': 'scroll',
                                                                        'maxHeight': '300px',
                                                                        'overflowY': 'scroll'},
                                                           style_cell={
                                                               'minWidth': '0px', 'maxWidth': '220px',
                                                               'whiteSpace': 'normal',
                                                           }
                                                           ),
                                      html.Div(id='data-table-feat-imp-container'),
                                  ]),
                                  html.Div([html.A(
                                      'Download Feature Importance data',
                                      id='download-link-feat-imp',
                                      href="",
                                      target="_blank"
                                  )]),

                              ], style={'padding': 20})
                          ])
                  ]),
    ])


# READ FILE
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            df.fillna(0)
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
            df.fillna(0)
        elif 'txt' or 'tsv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), delimiter=r'\s+')
            df.fillna(0)
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    return df


@app.callback(Output('csv-data', 'data'),
              [Input('data-table-upload', 'contents')],
              [State('data-table-upload', 'filename')])
def parse_uploaded_file(contents, filename):
    if not filename:
        return dash.no_update
    df = parse_contents(contents, filename)
    df.fillna(0)
    return df.to_json(date_format='iso', orient='split')


@app.callback(Output('feature-input', 'options'),
              [Input('csv-data', 'data')])
def activate_input(data):
    if not data:
        return dash.no_update, dash.no_update
    df = pd.read_json(data, orient='split')
    df = df.set_index(df.iloc[:, 0])
    # DROPPING NON NUMERICAL COLUMNS
    dff = df.select_dtypes(exclude=['object'])
    # REMOVING OUTLIERS
    z = np.abs(stats.zscore(dff))
    dff = dff[(z < 3).all(axis=1)]
    options = [{'label': i, 'value': i} for i in dff.columns]
    return options


@app.callback(Output('feature-target', 'options'),
              [Input('csv-data', 'data')])
def activate_input(data):
    if not data:
        return dash.no_update, dash.no_update
    df = pd.read_json(data, orient='split')
    df = df.set_index(df.iloc[:, 0])
    # DROPPING NON NUMERICAL COLUMNS
    dff = df.select_dtypes(exclude=['object'])
    # REMOVING OUTLIERS
    z = np.abs(stats.zscore(dff))
    dff = dff[(z < 3).all(axis=1)]
    options = [{'label': i, 'value': i} for i in dff.columns]
    return options

@app.callback([Output('data-table-features', 'data'),
               Output('data-table-features', 'columns')],
              [Input('feature-input', 'value'),
               Input('csv-data', 'data')])
def populate_feature_datatable(feature_value, data):
    if not data:
        return dash.no_update, dash.no_update
    df = pd.read_json(data, orient='split')
    df = df.set_index(df.iloc[:, 0])
    # DROPPING NON NUMERICAL COLUMNS
    dff = df.select_dtypes(exclude=['object'])
    # REMOVING OUTLIERS
    z = np.abs(stats.zscore(dff))
    dff = dff[(z < 3).all(axis=1)]
    if feature_value is None:
        raise dash.exceptions.PreventUpdate
    else:
        dff_input = dff[feature_value]
    data = dff_input.to_dict('records')
    columns = [{"name": i, "id": i, "selectable": True, 'type': 'numeric',
                'format': Format(precision=3, scheme=Scheme.fixed)} for i in dff_input.columns]
    return data, columns


@app.callback(Output('feature-heatmap', 'figure'),
              [Input('colorscale', 'value'),
               Input('feature-input', 'value'),
               Input('feature-target', 'value'),
               Input('csv-data', 'data')]
              )
def update_graph_stat(colorscale, feature_value, target, data):
    traces = []
    if not data:
        return dash.no_update
    df = pd.read_json(data, orient='split')
    df = df.set_index(df.iloc[:, 0])
    # DROPPING NON NUMERICAL COLUMNS
    dff = df.select_dtypes(exclude=['object'])
    # REMOVING OUTLIERS
    z = np.abs(stats.zscore(dff))
    dff = dff[(z < 3).all(axis=1)]
    if feature_value is None:
        raise dash.exceptions.PreventUpdate
    if target is None:
        raise dash.exceptions.PreventUpdate
    else:
        # correlation coefficient and coefficient of determination when features dropped
        dff_input_else = dff[feature_value]
        dff_input_else['Target variable'] = dff[target]
        features1_else = dff_input_else.columns
        features_else = list(features1_else)
        correlation_dff_else = dff_input_else.corr(method='pearson', )
        r2_dff_else = correlation_dff_else * correlation_dff_else
        data = r2_dff_else
        feat = features_else
    traces.append(go.Heatmap(
        z=data, x=feat, y=feat, colorscale="Viridis" if colorscale == 'Viridis' else "Plasma",
        # coord: represent the correlation between the various feature and the principal component itself
        colorbar={"title": "R²"}))
    return {'data': traces,
            'layout': go.Layout(title='<b>Feature Correlation Analysis with Target Variable</b>', xaxis={},
                                titlefont=dict(family='Georgia', size=16),
                                yaxis={},
                                hovermode='closest', margin={'b': 110, 't': 50, 'l': 170, 'r': 50},
                                font=dict(family="Helvetica", size=11)),
            }


def evaluate(model, X_out, Y_out):
    predictions = model.predict(X_out)
    errors = abs(predictions - Y_out)
    mape = 100 * np.mean(errors / Y_out)
    accuracy = 100 - mape
    return accuracy


def scaleup(x):
    return round(x * 1.1)


@app.callback(
    Output('output-container-button', 'children'),
    [Input('input_number', 'value')])
def update_output(value):
    return 'The input value was "{}"'.format(value)


@app.callback([
    Output('RandomisedSearchCV-container', 'children'),
    Output('Randomised-metrics-container', 'children'),
    Output('GridSearchCV-container', 'children'),
    Output('final-hyper-parameter-container', 'children'),
    Output('Grid-improvement-metrics-container', 'children'),
    Output('memory-output', 'data'),
    Output('memory-output-2', 'data'),
    Output('parity-plot', 'figure')],
    [
        Input('feature-input', 'value'),
        Input('feature-target', 'value'),
        Input('button', 'n_clicks'),
        Input('csv-data', 'data')],
    [State("input_number", "value")]
)
def populate_randomised_cv_grid(feature_value, target, n_clicks, data, n_inter):
    if not data:
        return dash.no_update
    df = pd.read_json(data, orient='split')
    df = df.set_index(df.iloc[:, 0])
    # DROPPING NON NUMERICAL COLUMNS
    dff = df.select_dtypes(exclude=['object'])
    # REMOVING OUTLIERS
    z = np.abs(stats.zscore(dff))
    dff = dff[(z < 3).all(axis=1)]
    if n_inter is None:
        raise dash.exceptions.PreventUpdate
    if feature_value is None:
        raise dash.exceptions.PreventUpdate
    elif n_clicks >= 1:
        # RANDOM SEARCH
        input_else_X = dff[feature_value]
        input_else_Y = dff[target]
        # need to change shape from (n,1) to (n,)
        X_cv, X_out, Y_cv, Y_out = train_test_split(input_else_X, input_else_Y, test_size=0.15, random_state=42)
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        max_features = ['auto', 'sqrt']
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [False, True]
        random_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth,
                       'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        pprint(random_grid)  # pprint is pretty print
        trf = RandomForestRegressor()
        # n-jobs= -1 means use all processors available in the computer
        # Check the number of processors (i5, i7 --> 12 jobs simultaneously). If doing other things too like opening word or such, do n_jobs = 10 instead of the 12
        trf_random = RandomizedSearchCV(estimator=trf, param_distributions=random_grid, n_iter=n_inter, cv=5, verbose=2,
                                        random_state=42, n_jobs=-1)
        trf_random.fit(X_cv,
                       Y_cv.values.ravel())  # [Parallel(n_jobs=-1)]: Done 500 out of 500 (100 iterations and cv of 5) | elapsed: 75.0min finished
        pprint(trf_random.best_params_)
        # comparing randomised search grid to base model hyperparameter grid
        base_model = RandomForestRegressor(n_estimators=100, random_state=42)
        base_model.fit(X_cv, Y_cv.values.ravel())
        base_accuracy = evaluate(base_model, X_out, Y_out.values.ravel())
        best_random = trf_random.best_estimator_
        random_accuracy = evaluate(best_random, X_out, Y_out.values.ravel())
        percentage_improvement = (100 * (random_accuracy - base_accuracy)) / base_accuracy

        # POPULATE GRID SEARCH
        best_random_params = trf_random.best_params_
        best_random_params_df = pd.DataFrame.from_dict(best_random_params, orient='index')
        best_random_params_df = best_random_params_df.T

        param_grid = {'n_estimators': [abs(best_random_params_df.at[0, 'n_estimators'] - 200)
                                       if int(best_random_params_df.at[0, 'n_estimators'] - 200) > 0 else 1,
                                       best_random_params_df.at[0, 'n_estimators'],
                                       (best_random_params_df.at[0, 'n_estimators'] + 200)],
                      # NEED TO CHANGE MAX FEATURES
                      'max_features': [abs(round(math.sqrt(len(input_else_X.columns))) - 1),
                                       round(math.sqrt(len(input_else_X.columns))),
                                       (round(math.sqrt(len(input_else_X.columns))) + 1)]
                      if best_random_params_df.at[0, 'max_features'] == 'sqrt' else
                      [abs(len(input_else_X.columns) - 1), len(input_else_X.columns), (len(input_else_X.columns) + 1)],
                      'min_samples_leaf': [abs(best_random_params_df.at[0, 'min_samples_leaf'] - 1)
                                           if int(best_random_params_df.at[0, 'min_samples_leaf'] - 1) > 0 else 1,
                                           best_random_params_df.at[0, 'min_samples_leaf'],
                                           (best_random_params_df.at[0, 'min_samples_leaf'] + 1)],
                      'min_samples_split': [abs(best_random_params_df.at[0, 'min_samples_split'] - 1)
                                            if int(best_random_params_df.at[0, 'min_samples_split'] - 1) > 1 else 2,
                                            best_random_params_df.at[0, 'min_samples_split'],
                                            (best_random_params_df.at[0, 'min_samples_split'] + 1)],
                      'max_depth': [abs(best_random_params_df.at[0, 'max_depth'] - 10)
                                    if int(best_random_params_df.at[0, 'max_depth'] - 10) > 0 else 1,
                                    best_random_params_df.at[0, 'max_depth'],
                                    (best_random_params_df.at[0, 'max_depth'] + 10)],
                      'bootstrap': [best_random_params_df.at[0, 'bootstrap']]}
        grf = RandomForestRegressor()
        grid_search = GridSearchCV(estimator=grf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(X_cv, Y_cv.values.ravel())
        pprint(grid_search.best_params_)

        best_grid = grid_search.best_estimator_
        grid_accuracy = evaluate(best_grid, X_out, Y_out.values.ravel())
        percentage_improvement_grid = (100 * (grid_accuracy - base_accuracy)) / base_accuracy

        # GRID SEARCH BEST PARAMS
        best_grid_search = grid_search.best_params_
        best_grid_search_df = pd.DataFrame.from_dict(best_grid_search, orient='index')
        best_grid_search_df = best_grid_search_df.T

        # RANDOM FOREST using best grid search hyperparameters and testing on test set
        regressor = RandomForestRegressor(n_estimators=best_grid_search_df.at[0, 'n_estimators'],
                                          max_depth=best_grid_search_df.at[0, 'max_depth'],
                                          max_features=best_grid_search_df.at[0, 'max_features'],
                                          min_samples_leaf=best_grid_search_df.at[0, 'min_samples_leaf'],
                                          min_samples_split=best_grid_search_df.at[0, 'min_samples_split'],
                                          bootstrap=best_grid_search_df.at[0, 'bootstrap'],
                                          random_state=75)
        regressor.fit(X_cv, Y_cv.values.ravel())
        Y_pred = regressor.predict(X_out)
        all_pred = regressor.predict(input_else_X)
        rfaccuracy = evaluate(regressor, X_out, Y_out.values.ravel())
        MAE_rf1 = metrics.mean_absolute_error(Y_out.values.ravel(), Y_pred)
        MSE_rf1 = metrics.mean_squared_error(Y_out.values.ravel(), Y_pred)
        RMSE_rf1 = np.sqrt(metrics.mean_squared_error(Y_out.values.ravel(), Y_pred))
        R2_rf1 = metrics.r2_score(Y_out.values.ravel(), Y_pred)
        performance_metrics = pd.DataFrame(data=[rfaccuracy, MAE_rf1, MSE_rf1, RMSE_rf1, R2_rf1],
                                           index=["Model Accuracy (%)", 'MAE', 'MSE', 'RMSE', 'R2'.translate(SUP)])
        performance_metrics = performance_metrics.T

        def rf_feat_importance(regressor, X):
            feature_imps = pd.DataFrame(data=regressor.feature_importances_, columns=['Importance'])
            cols = X.columns
            cols = cols.T
            cols_np = cols.to_numpy()
            cols_df = pd.DataFrame(data=cols_np, columns=['Features'])
            feat_importance = pd.concat([cols_df, feature_imps], axis=1).sort_values('Importance', ascending=False)
            feat_importance['Cumulative Importance'] = feat_importance['Importance'].cumsum()
            return feat_importance

        feature_importance = rf_feat_importance(regressor, input_else_X)
        shared_data = pd.concat([best_grid_search_df, feature_importance], axis=1)
        # KDE METHOD
        Y_out_Y_pred = np.vstack([Y_out.values.ravel(), Y_pred])
        color_rf1 = gaussian_kde(Y_out_Y_pred)(Y_out_Y_pred)
        # # 2D HISTOGRAM METHOD
        # data, x_e, y_e = np.histogram2d(Y_out, Y_pred, density=True)
        # color_rf1 = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data,
        #                     np.vstack([Y_out, Y_pred]).T,
        #                     method="splinef2d", bounds_error=False)
        # # To be sure to plot all data
        # z[np.where(np.isnan(z))] = 0.0E

        # Sort the points by density, so that the densest points are plotted last
        idx = color_rf1.argsort()
        Y_out, Y_pred, color_rf1 = Y_out[idx], Y_pred[idx], color_rf1[idx]
        Y_out_np = Y_out.to_numpy()
        Y_out_df = pd.DataFrame(data=Y_out_np, columns=["Y_out"])
        Y_pred_df = pd.DataFrame(data=Y_pred, columns=["Y_pred"])
        color_rf1_df = pd.DataFrame(data=color_rf1, columns=["count"])
        test_plot_df = pd.concat([Y_out_df, Y_pred_df, color_rf1_df], axis=1)
        test_plot_df["Y_errors"] = abs(test_plot_df["Y_pred"] - test_plot_df["Y_out"])
        traces = []
        traces.append(go.Scatter(x=test_plot_df["Y_out"], y=test_plot_df["Y_pred"], mode='markers',
                                 marker_color=test_plot_df["count"], meta=test_plot_df["Y_errors"],
                                 text=Y_out.index,
                                 hovertemplate=
                                 "<b>%{text}</b>" +
                                 '<br>Error: %{meta:.2f}' +
                                 '<br>Observed: %{x:.2f}<br>' +
                                 'Predicted: %{y:.2f}'
                                 "<extra></extra>",
                                 marker=dict(opacity=0.8, showscale=True, size=12,
                                             line=dict(width=0.5, color='DarkSlateGrey'),
                                             colorscale='Viridis',
                                             colorbar=dict(title=dict(text='KDE',
                                                                      font=dict(family='Helvetica'),
                                                                      side='right'), ypad=0),
                                             ),
                                 ))
        traces.append(
            go.Scatter(x=[0, scaleup(test_plot_df["Y_out"].max())], y=[0, scaleup(test_plot_df["Y_pred"].max())],
                       hoverinfo='skip', mode='lines', line=dict(color='Black', width=1, dash='dot')))

    return '{}'.format(trf_random.best_params_), '{:0.2f}% improvement'.format(percentage_improvement), \
           '{}'.format(param_grid), '{}'.format(grid_search.best_params_), '{:0.2f}% improvement'.format(
        percentage_improvement_grid), shared_data.to_dict('records'), performance_metrics.to_dict('records'), \
           {'data': traces,
            'layout': go.Layout(
                title='<b>Parity plot using test set from hyperparameter tuning</b>',
                titlefont=dict(family='Georgia', size=16),
                showlegend=False,
                xaxis={
                    'title': "{} (Observed)".format(target),
                    'mirror': True,
                    'ticks': 'outside',
                    'showline': True, 'range': [0, scaleup(
                        test_plot_df["Y_out"].max())],
                    'rangemode': "tozero"},
                yaxis={
                    'title': "{} (Predicted)".format(target),
                    'mirror': True,
                    'ticks': 'outside',
                    'showline': True, 'rangemode': "tozero",
                    'range': [0, scaleup(
                        test_plot_df["Y_pred"].max())]},
                hovermode='closest',
                font=dict(family="Helvetica"),
                template="simple_white"
            )
            }


@app.callback(
    [Output('performance-metrics-table', 'data'),
     Output('performance-metrics-table', 'columns')],
    [Input('memory-output-2', 'data'),
     ])
def populate_metrics_table(performance_metrics):
    if performance_metrics is None:
        raise dash.exceptions.PreventUpdate
    performance_metrics_df = pd.DataFrame(data=performance_metrics)
    # performance_metrics_df = performance_metrics_df.T
    # performance_metrics_df.insert(0, 'Performance Metrics', performance_metrics_df.index)
    # performance_metrics_df.columns = ['Performance Metrics', ' ']
    data = performance_metrics_df.to_dict('records')
    columns = [{"name": i, "id": i, "selectable": True, 'type': 'numeric',
                'format': Format(precision=2, scheme=Scheme.fixed)} for i in performance_metrics_df.columns]
    return data, columns


@app.callback(Output('feature-importance', 'figure'),
              [Input('memory-output', 'data'),])
def update_shared_data(shared_data):
    if shared_data is None:
        raise dash.exceptions.PreventUpdate
    shared_data_df = pd.DataFrame(data=shared_data)
    dfs = np.split(shared_data_df, [6], axis=1)
    best_grid_param = dfs[0]
    best_grid_param = best_grid_param.dropna()
    best_grid_param = best_grid_param.reset_index(drop=True)
    feature_importance = dfs[1]
    feature_importance = feature_importance.dropna()
    feature_importance = feature_importance.reset_index(drop=True)
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    traces = []
    traces.append(go.Bar(x=feature_importance['Features'], y=feature_importance['Importance'], hoverinfo='skip',
                         text=feature_importance['Features'], hovertemplate=
                         "<b>%{text}</b>" +
                         '<br>Importance: %{y}<br>' +
                         "<extra></extra>",
                         ))
    traces.append(go.Scatter(x=feature_importance['Features'], y=feature_importance['Cumulative Importance'],
                             text=feature_importance['Features'], hoverinfo='skip',
                             mode='lines+markers', line=dict(color='#0a0054'),
                             hovertemplate=
                             "<b>%{text}</b>" +
                             '<br>Cumulative Importance: %{y}<br>' +
                             "<extra></extra>",
                             ))
    return {'data': traces,
            'layout': go.Layout(
                title="<b>Feature Importance</b>",
                titlefont=dict(family='Georgia', size=16),
                showlegend=False,
                xaxis={'title': 'Features/ Descriptors'},
                yaxis={'title': 'Feature Importance'},
                hovermode='closest', font=dict(family="Helvetica"),
                template="simple_white"
            )}


@app.callback([
    Output('memory-output-4', 'data'),
    Output('memory-output-3', 'data'),
    Output('parity-plot-final', 'figure')],
    [Input('feature-input', 'value'),
     Input('feature-target', 'value'),
     Input('input-number-test', 'value'),
     Input('memory-output', 'data'),
     Input('csv-data', 'data')])
def populate_final_RF(feature_value, target, test_size, shared_data, data):
    if not data:
        return dash.no_update, dash.no_update
    df = pd.read_json(data, orient='split')
    df = df.set_index(df.iloc[:, 0])
    # DROPPING NON NUMERICAL COLUMNS
    dff = df.select_dtypes(exclude=['object'])
    # REMOVING OUTLIERS
    z = np.abs(stats.zscore(dff))
    dff = dff[(z < 3).all(axis=1)]
    if shared_data is None:
        raise dash.exceptions.PreventUpdate
    shared_data_df = pd.DataFrame(data=shared_data)
    dfs = np.split(shared_data_df, [6], axis=1)
    best_grid_param = dfs[0]
    best_grid_param = best_grid_param.dropna()
    best_grid_param = best_grid_param.reset_index(drop=True)
    input_else_X = dff[feature_value]
    input_else_Y = dff[target]
    # SECOND RANDOM FOREST using best grid search hyperparameters and testing on new test set (diff random state)
    X_train, X_test, Y_train, Y_test = train_test_split(input_else_X, input_else_Y,
                                                        test_size=0.25 if test_size is None else test_size,
                                                        random_state=34)
    regressor = RandomForestRegressor(n_estimators=int(best_grid_param.at[0, 'n_estimators']),
                                      max_depth=int(best_grid_param.at[0, 'max_depth']),
                                      max_features=int(best_grid_param.at[0, 'max_features']),
                                      min_samples_leaf=int(best_grid_param.at[0, 'min_samples_leaf']),
                                      min_samples_split=int(best_grid_param.at[0, 'min_samples_split']),
                                      bootstrap=best_grid_param.at[0, 'bootstrap'], random_state=64)
    regressor.fit(X_train, Y_train.values.ravel())
    Y_pred = regressor.predict(X_test)
    all_pred = regressor.predict(input_else_X)
    rfaccuracy = evaluate(regressor, X_test, Y_test.values.ravel())

    MAE_rf1 = metrics.mean_absolute_error(Y_test.values.ravel(), Y_pred)
    MSE_rf1 = metrics.mean_squared_error(Y_test.values.ravel(), Y_pred)
    RMSE_rf1 = np.sqrt(metrics.mean_squared_error(Y_test.values.ravel(), Y_pred))
    R2_rf1 = metrics.r2_score(Y_test.values.ravel(), Y_pred)
    performance_metrics = pd.DataFrame(data=[rfaccuracy, MAE_rf1, MSE_rf1, RMSE_rf1, R2_rf1],
                                       index=["Model Accuracy (%)", 'MAE', 'MSE', 'RMSE', 'R2'.translate(SUP)])
    performance_metrics = performance_metrics.T

    def rf_feat_importance(regressor_final, X):
        feature_imps = pd.DataFrame(data=regressor_final.feature_importances_, columns=['Importance'])
        cols = X.columns
        cols = cols.T
        cols_np = cols.to_numpy()
        cols_df = pd.DataFrame(data=cols_np, columns=['Features'])
        feat_importance = pd.concat([cols_df, feature_imps], axis=1).sort_values('Importance', ascending=False)
        feat_importance['Cumulative Importance'] = feat_importance['Importance'].cumsum()
        return feat_importance

    feature_importance = rf_feat_importance(regressor, input_else_X)

    shared_data = pd.concat([performance_metrics, feature_importance], axis=1)

    # KDE METHOD
    Y_test_Y_pred = np.vstack([Y_test.values.ravel(), Y_pred])
    color_rf1 = gaussian_kde(Y_test_Y_pred)(Y_test_Y_pred)
    # # 2D HISTOGRAM METHOD
    # data, x_e, y_e = np.histogram2d(Y_test, Y_pred, density=True)
    # color_rf1 = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data,
    #                     np.vstack([Y_test, Y_pred]).T,
    #                     method="splinef2d", bounds_error=False)
    # # To be sure to plot all data
    # z[np.where(np.isnan(z))] = 0.0E

    # Sort the points by density, so that the densest points are plotted last
    idx = color_rf1.argsort()
    Y_test, Y_pred, color_rf1 = Y_test[idx], Y_pred[idx], color_rf1[idx]
    Y_test_np = Y_test.to_numpy()
    Y_test_df = pd.DataFrame(data=Y_test_np, columns=["Y_test"])
    Y_pred_df = pd.DataFrame(data=Y_pred, columns=["Y_pred"])
    color_rf1_df = pd.DataFrame(data=color_rf1, columns=["count"])
    test_plot_df = pd.concat([Y_test_df, Y_pred_df, color_rf1_df], axis=1)
    test_plot_df["Y_errors"] = (test_plot_df["Y_pred"] - test_plot_df["Y_test"])
    test_plot_df["Index"] = Y_test.index
    traces = []
    traces.append(go.Scatter(x=test_plot_df["Y_test"], y=test_plot_df["Y_pred"], mode='markers',
                             marker_color=test_plot_df["count"], meta=test_plot_df["Y_errors"],
                             text=Y_test.index,
                             hovertemplate=
                             "<b>%{text}</b>" +
                             '<br>Error: %{meta:.2f}' +
                             '<br>Observed: %{x:.2f}<br>' +
                             'Predicted: %{y:.2f}'
                             "<extra></extra>",
                             marker=dict(opacity=0.8, showscale=True, size=12,
                                         line=dict(width=0.5, color='DarkSlateGrey'),
                                         colorscale='Viridis',
                                         colorbar=dict(title=dict(text='KDE',
                                                                  font=dict(family='Helvetica'),
                                                                  side='right'), ypad=0),
                                         ),
                             ))
    traces.append(
        go.Scatter(x=[0, scaleup(test_plot_df["Y_test"].max())], y=[0, scaleup(test_plot_df["Y_pred"].max())],
                   hoverinfo='skip', mode='lines', line=dict(color='Black', width=1, dash='dot')))
    return test_plot_df.to_dict('records'), shared_data.to_dict('records'), \
           {'data': traces,
            'layout': go.Layout(
                title='<b>Parity plot </b>',
                titlefont=dict(family='Georgia',
                               size=16),
                showlegend=False,
                xaxis={
                    'title': "{} (Observed)".format(
                        target),
                    'mirror': True,
                    'ticks': 'outside',
                    'showline': True,
                    'range': [0, scaleup(
                        test_plot_df["Y_test"].max())],
                    'rangemode': "tozero"},
                yaxis={
                    'title': "{} (Predicted)".format(
                        target),
                    'mirror': True,
                    'ticks': 'outside',
                    'showline': True,
                    'rangemode': "tozero",
                    'range': [0, scaleup(
                        test_plot_df[
                            "Y_pred"].max())]},
                hovermode='closest',
                font=dict(family="Helvetica"),
                template="simple_white"
            )
            }


@app.callback(
    [Output('performance-metrics-table-final', 'data'),
     Output('performance-metrics-table-final', 'columns')],
    [Input('memory-output-3', 'data'), ])
def populate_metrics_table(shared_data):
    if shared_data is None:
        raise dash.exceptions.PreventUpdate
    shared_data_df = pd.DataFrame(data=shared_data)
    dfs = np.split(shared_data_df, [5], axis=1)
    performance_metrics = dfs[0]
    performance_metrics = performance_metrics.dropna()
    performance_metrics_df = performance_metrics.reset_index(drop=True)
    # performance_metrics_df = performance_metrics_df.T
    # performance_metrics_df.insert(0, 'Performance Metrics', performance_metrics_df.index)
    # performance_metrics_df.columns = ['Performance Metrics', ' ']
    data = performance_metrics_df.to_dict('records')
    columns = [{"name": i, "id": i, "selectable": True, 'type': 'numeric',
                'format': Format(precision=2, scheme=Scheme.fixed)} for i in performance_metrics_df.columns]
    return data, columns


@app.callback(Output('feature-importance-final', 'figure'),
              [Input('memory-output-3', 'data')])
def update_shared_data_final(shared_data):
    if shared_data is None:
        raise dash.exceptions.PreventUpdate
    shared_data_df = pd.DataFrame(data=shared_data)
    dfs = np.split(shared_data_df, [5], axis=1)
    performance_metrics = dfs[0]
    feature_importance = dfs[1]
    feature_importance = feature_importance.dropna()
    feature_importance = feature_importance.reset_index(drop=True)
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    traces = []
    traces.append(go.Bar(x=feature_importance['Features'], y=feature_importance['Importance'], hoverinfo='skip',
                         text=feature_importance['Features'], hovertemplate=
                         "<b>%{text}</b>" +
                         '<br>Importance: %{y}<br>' +
                         "<extra></extra>",
                         ))
    traces.append(go.Scatter(x=feature_importance['Features'], y=feature_importance['Cumulative Importance'],
                             text=feature_importance['Features'], hoverinfo='skip',
                             mode='lines+markers', line=dict(color='#0a0054'),
                             hovertemplate=
                             "<b>%{text}</b>" +
                             '<br>Cumulative Importance: %{y}<br>' +
                             "<extra></extra>",
                             ))
    return {'data': traces,
            'layout': go.Layout(
                title="<b>Feature Importance</b>",
                titlefont=dict(family='Georgia', size=16),
                showlegend=False,
                xaxis={'title': 'Features/ Descriptors'},
                yaxis={'title': 'Feature Importance'},
                hovermode='closest', font=dict(family="Helvetica"),
                template="simple_white"
            )}


@app.callback(Output('error-dist', 'figure'),
              [Input('memory-output-4', 'data'),
               Input('feature-target', 'value')])
def populate_error_dist(test_plot_data, target):
    if test_plot_data is None:
        raise dash.exceptions.PreventUpdate
    if target is None:
        raise dash.exceptions.PreventUpdate
    test_plot_data = pd.DataFrame(data=test_plot_data)
    test_plot_data["Error (%)"] = (test_plot_data["Y_errors"] / test_plot_data["Y_test"]) * 100
    test_plot_data.rename(columns={
        'Y_test': 'Observed',
        'Y_pred': 'Predicted'},
        inplace=True)
    return px.histogram(test_plot_data, x="Error (%)", marginal="rug", hover_name="Index", template="simple_white",
                        hover_data=["Error (%)", "Observed", "Predicted"]
                        ).update_xaxes(showgrid=False, autorange=True, ticks='outside',
                                       mirror=True, showline=True, tickformat=".1f", title=' '
                                       ).update_yaxes(showgrid=False, ticks='outside',
                                                      mirror=True, autorange=True, showline=True, tickformat=".1f",
                                                      title=' '
                                                      ).update_layout(
        hovermode='closest',
        # margin={'l': 60, 'b': 80, 't': 50, 'r': 10},
        autosize=True, font=dict(family='Helvetica'),
        annotations=[
            dict(x=0.5, y=-0.17, showarrow=False, text='Error (%)', xref='paper', yref='paper',
                 font=dict(size=14, family="Helvetica")),
            dict(x=-0.12, y=0.5, showarrow=False, text="Count", textangle=-90, xref='paper',
                 yref='paper', font=dict(size=14, family="Helvetica"))],
        title=f"<b> Error Distribution of {''.join(target)} predictions", titlefont=dict(
            family="Georgia", size=16))


@app.callback(Output('download-link-correlation', 'download'),
              [Input('feature-target', 'value'),
               ])
def update_filename(target):
    if target is None:
        raise dash.exceptions.PreventUpdate
    download = 'feature_correlation_with_{}.csv'.format(target)
    return download


@app.callback([Output('data-table-correlation', 'data'),
               Output('data-table-correlation', 'columns'),
               Output('download-link-correlation', 'href')],
              [Input("feature-target", 'value'),
               Input("feature-input", 'value'),
               Input('csv-data', 'data')], )
def update_output(target, feature_value, data):
    if not data:
        return dash.no_update, dash.no_update
    df = pd.read_json(data, orient='split')
    df = df.set_index(df.iloc[:, 0])
    # DROPPING NON NUMERICAL COLUMNS
    dff = df.select_dtypes(exclude=['object'])
    # REMOVING OUTLIERS
    z = np.abs(stats.zscore(dff))
    dff = dff[(z < 3).all(axis=1)]
    if target is None:
        raise dash.exceptions.PreventUpdate
    if feature_value is None:
        raise dash.exceptions.PreventUpdate
    dff_input_else_table = dff[feature_value]
    dff_input_else_table['Target variable'] = dff[target]
    correlation_dff_else_table = dff_input_else_table.corr(method='pearson', )
    r2_dff_else_table = correlation_dff_else_table * correlation_dff_else_table
    data = r2_dff_else_table.to_dict('records')
    columns = [{"name": i, "id": i, "deletable": True, "selectable": True, 'type': 'numeric',
                'format': Format(precision=3, scheme=Scheme.fixed)} for i in r2_dff_else_table.columns]
    csv_string = r2_dff_else_table.to_csv(index=False, encoding='utf-8')
    csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_string)
    return data, columns, csv_string


@app.callback(Output('download-link-RF', 'download'),
              [Input('feature-target', 'value'),
               ])
def update_filename(target):
    if target is None:
        raise dash.exceptions.PreventUpdate
    download = 'RF_output_{}.csv'.format(target)
    return download


@app.callback([Output('data-table-RF', 'data'),
               Output('data-table-RF', 'columns'),
               Output('download-link-RF', 'href')],
              [Input("feature-target", 'value'),
               Input("feature-input", 'value'),
               Input('memory-output-4', 'data')], )
def update_output(target, feature_value, test_plot_data):
    if target is None:
        raise dash.exceptions.PreventUpdate
    if feature_value is None:
        raise dash.exceptions.PreventUpdate
    if test_plot_data is None:
        raise dash.exceptions.PreventUpdate
    test_plot_data = pd.DataFrame(data=test_plot_data)
    test_plot_data["Error (%)"] = (test_plot_data["Y_errors"] / test_plot_data["Y_test"]) * 100
    test_plot_data.rename(columns={
        'Y_test': 'Observed',
        'Y_pred': 'Predicted',
        "Y_errors": "Error ",
        'Index': 'Identifier'},
        inplace=True)
    test_plot_data = test_plot_data.drop(columns=['count'])
    test_plot_data = test_plot_data[['Identifier'] + [col for col in test_plot_data.columns if col != 'Identifier']]
    data = test_plot_data.to_dict('records')
    columns = [{"name": i, "id": i, "deletable": True, "selectable": True, 'type': 'numeric',
                'format': Format(precision=3, scheme=Scheme.fixed)} for i in test_plot_data.columns]
    csv_string = test_plot_data.to_csv(index=False, encoding='utf-8')
    csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_string)
    return data, columns, csv_string


@app.callback(Output('download-link-feat-imp', 'download'),
              [Input('feature-target', 'value'),
               ])
def update_filename(target):
    if target is None:
        raise dash.exceptions.PreventUpdate
    download = 'Feature_Importance_{}.csv'.format(target)
    return download


@app.callback([Output('data-table-feat-imp', 'data'),
               Output('data-table-feat-imp', 'columns'),
               Output('download-link-feat-imp', 'href')],
              [Input("feature-target", 'value'),
               Input("feature-input", 'value'),
               Input('memory-output-3', 'data')], )
def update_output(target, feature_value, shared_data):
    if target is None:
        raise dash.exceptions.PreventUpdate
    if feature_value is None:
        raise dash.exceptions.PreventUpdate
    if shared_data is None:
        raise dash.exceptions.PreventUpdate
    shared_data_df = pd.DataFrame(data=shared_data)
    dfs = np.split(shared_data_df, [5], axis=1)
    performance_metrics = dfs[0]
    feature_importance = dfs[1]
    feature_importance = feature_importance.dropna()
    feature_importance = feature_importance.reset_index(drop=True)
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    data = feature_importance.to_dict('records')
    columns = [{"name": i, "id": i, "deletable": True, "selectable": True, 'type': 'numeric',
                'format': Format(precision=3, scheme=Scheme.fixed)} for i in feature_importance.columns]
    csv_string = feature_importance.to_csv(index=False, encoding='utf-8')
    csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_string)
    return data, columns, csv_string


@app.callback(Output('download-link-performance-met', 'download'),
              [Input('feature-target', 'value'),
               ])
def update_filename(target):
    if target is None:
        raise dash.exceptions.PreventUpdate
    download = 'Performance_Metrics_{}.csv'.format(target)
    return download


@app.callback([Output('data-table-performance-met', 'data'),
               Output('data-table-performance-met', 'columns'),
               Output('download-link-performance-met', 'href')],
              [Input("feature-target", 'value'),
               Input("feature-input", 'value'),
               Input('memory-output-3', 'data')], )
def update_output(target, feature_value, shared_data):
    if target is None:
        raise dash.exceptions.PreventUpdate
    if feature_value is None:
        raise dash.exceptions.PreventUpdate
    if shared_data is None:
        raise dash.exceptions.PreventUpdate
    shared_data_df = pd.DataFrame(data=shared_data)
    dfs = np.split(shared_data_df, [5], axis=1)
    performance_metrics = dfs[0]
    performance_metrics = performance_metrics.dropna()
    performance_metrics_df = performance_metrics.reset_index(drop=True)
    data = performance_metrics_df.to_dict('records')
    columns = [{"name": i, "id": i, "deletable": True, "selectable": True, 'type': 'numeric',
                'format': Format(precision=3, scheme=Scheme.fixed)} for i in performance_metrics_df.columns]
    csv_string = performance_metrics_df.to_csv(index=False, encoding='utf-8')
    csv_string = "data:text/csv;charset=utf-8,%EF%BB%BF" + urllib.parse.quote(csv_string)
    return data, columns, csv_string


# serve(server)
if __name__ == '__main__':
    # For Development only, otherwise use gunicorn or uwsgi to launch, e.g.
    # gunicorn -b 0.0.0.0:8050 index:app.server
    app.run_server(debug=False)
