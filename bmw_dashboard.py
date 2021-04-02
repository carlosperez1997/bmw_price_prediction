import dash
import dash_core_components as dcc 
import dash_html_components as html
import dash_table

# DataFrame y matrices
import numpy as np 
import pandas as pd

import os
import pickle

# Visualizacion
import matplotlib.pyplot as plt 
import seaborn as sns


from dash.dependencies import Input, Output, State

app = dash.Dash(__name__)

html_string = '''
<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta http-equiv="X-UA-Compatible" content="ie=edge">
        <link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.6.3/css/all.css" integrity="sha384-UHRtZLI+pbxtHCWp1t77Bi1L4ZtiqrqD80Kn4Z8NTSRyMA2Fd33n5dQ8lWUE00s/" crossorigin="anonymous">

        <title>BMW Price Prediction</title>

    <style>
        * {
            box-sizing: border-box;
            margin:0;
            padding: 0;
            text-align: center;
        }

        .text-primary {
            color: #3c81af;
        }
        
        /* Intro */
        #intro {
            padding:1rem;
            width: 70%;
            margin: auto;
        }

        #intro p {
            padding: 0.8rem;   
        }

        /* Navbar */
        #navbar {
            display: flex;
            position: sticky;
            text-align: center;
            top:0;
            background: #333;
            color: #fff;
            justify-content: space-between;
            z-index: 1;
            padding: 1rem;
        }

        /* Footer */
        footer {
            background: #333;
            bottom:0;
            color: #fff;
            justify-content: space-between;
        }

    </style>
    </head>

    <body id="home">
        <!-- Navbar -->
        <nav id="navbar">
            <h1 class="logo" style="margin: auto">
                <span class="text-primary"> BMW </span> Pricing Prediction
            </h1>
            
        </nav>
        
        <div id="intro">
            <p> This website predicts the price of a second hand BMW given features 
                like the model, power, mileage, colour, type of car, type of fuel or even if 
                it has extras (gps, blueetooth, rear camera...) </p>
            <p> The model implemented is a Random Forest Regressor with: n_estimators: 120, max_depth: 50 and min_samples_leaf: 5.</p>

            <p> You can see the model selection in a notebook in GitHub: <a href="https://github.com/carlosperez1997/bmw_price_prediction/blob/main/price_prediction.ipynb" target="_blank"> Building the Model</a>  </p>
            <p> Whereas the code of this dashboard is in: <a href="https://github.com/carlosperez1997/bmw_price_prediction/blob/main/bmw_dashboard.py" target="_blank"> Building the Dashboard </a>  </p>
        
            <p> Select the car and features and click 'Estimate price': </p>
        </div>

        {%app_entry%}

        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        
            <h1 class="logo" style="margin: auto">
                <span class="text-primary"> BMW </span> Pricing Prediction
            </h1>
            <p> Carlos Pérez Ricardo <a href="https://github.com/carlosperez1997" target="_blank" style="color:white"> GitHub </a></p>

        </footer>
    </body>
</html>
'''

model_options = [{'label': 'i8', 'value': 'modelo_i8'},
 {'label': 'M4', 'value': 'modelo_M4'},
 {'label': 'X6 M', 'value': 'modelo_X6 M'},
 {'label': 'X5 M50', 'value': 'modelo_X5 M50'},
 {'label': 'M5', 'value': 'modelo_M5'},
 {'label': '640 Gran Coupé', 'value': 'modelo_640 Gran Coupé'},
 {'label': 'X5 M', 'value': 'modelo_X5 M'},
 {'label': '740', 'value': 'modelo_740'},
 {'label': '750', 'value': 'modelo_750'},
 {'label': 'M3', 'value': 'modelo_M3'},
 {'label': 'M550', 'value': 'modelo_M550'},
 {'label': 'X6', 'value': 'modelo_X6'},
 {'label': '640', 'value': 'modelo_640'},
 {'label': '435 Gran Coupé', 'value': 'modelo_435 Gran Coupé'},
 {'label': 'X4', 'value': 'modelo_X4'},
 {'label': '435', 'value': 'modelo_435'},
 {'label': '425', 'value': 'modelo_425'},
 {'label': 'X5', 'value': 'modelo_X5'},
 {'label': '430', 'value': 'modelo_430'},
 {'label': 'M235', 'value': 'modelo_M235'},
 {'label': '430 Gran Coupé', 'value': 'modelo_430 Gran Coupé'},
 {'label': 'M135', 'value': 'modelo_M135'},
 {'label': '330 Gran Turismo', 'value': 'modelo_330 Gran Turismo'},
 {'label': '535 Gran Turismo', 'value': 'modelo_535 Gran Turismo'},
 {'label': '335 Gran Turismo', 'value': 'modelo_335 Gran Turismo'},
 {'label': '420 Gran Coupé', 'value': 'modelo_420 Gran Coupé'},
 {'label': '420', 'value': 'modelo_420'},
 {'label': '220', 'value': 'modelo_220'},
 {'label': '730', 'value': 'modelo_730'},
 {'label': '535', 'value': 'modelo_535'},
 {'label': '135', 'value': 'modelo_135'},
 {'label': '335', 'value': 'modelo_335'},
 {'label': 'i3', 'value': 'modelo_i3'},
 {'label': 'ActiveHybrid 5', 'value': 'modelo_ActiveHybrid 5'},
 {'label': '530 Gran Turismo', 'value': 'modelo_530 Gran Turismo'},
 {'label': '418 Gran Coupé', 'value': 'modelo_418 Gran Coupé'},
 {'label': '325 Gran Turismo', 'value': 'modelo_325 Gran Turismo'},
 {'label': '528', 'value': 'modelo_528'},
 {'label': '520 Gran Turismo', 'value': 'modelo_520 Gran Turismo'},
 {'label': '530', 'value': 'modelo_530'},
 {'label': '635', 'value': 'modelo_635'},
 {'label': '225 Active Tourer', 'value': 'modelo_225 Active Tourer'},
 {'label': '218', 'value': 'modelo_218'},
 {'label': ' Active Tourer', 'value': 'modelo_ Active Tourer'},
 {'label': '225', 'value': 'modelo_225'},
 {'label': 'X3', 'value': 'modelo_X3'},
 {'label': '214 Gran Tourer', 'value': 'modelo_214 Gran Tourer'},
 {'label': '320 Gran Turismo', 'value': 'modelo_320 Gran Turismo'},
 {'label': '216 Gran Tourer', 'value': 'modelo_216 Gran Tourer'},
 {'label': '330', 'value': 'modelo_330'},
 {'label': '328', 'value': 'modelo_328'},
 {'label': '518', 'value': 'modelo_518'},
 {'label': '218 Gran Tourer', 'value': 'modelo_218 Gran Tourer'},
 {'label': '520', 'value': 'modelo_520'},
 {'label': '525', 'value': 'modelo_525'},
 {'label': '218 Active Tourer', 'value': 'modelo_218 Active Tourer'},
 {'label': '318 Gran Turismo', 'value': 'modelo_318 Gran Turismo'},
 {'label': '325', 'value': 'modelo_325'},
 {'label': '216 Active Tourer', 'value': 'modelo_216 Active Tourer'},
 {'label': 'X1', 'value': 'modelo_X1'},
 {'label': '125', 'value': 'modelo_125'},
 {'label': '120', 'value': 'modelo_120'},
 {'label': '320', 'value': 'modelo_320'},
 {'label': '220 Active Tourer', 'value': 'modelo_220 Active Tourer'},
 {'label': '114', 'value': 'modelo_114'},
 {'label': '318', 'value': 'modelo_318'},
 {'label': '630', 'value': 'modelo_630'},
 {'label': '316', 'value': 'modelo_316'},
 {'label': '116', 'value': 'modelo_116'},
 {'label': '118', 'value': 'modelo_118'},
 {'label': 'Z4', 'value': 'modelo_Z4'},
 {'label': '123', 'value': 'modelo_123'},
 {'label': '650', 'value': 'modelo_650'},
 {'label': '523', 'value': 'modelo_523'},
 {'label': '216', 'value': 'modelo_216'},
 {'label': '735', 'value': 'modelo_735'}]

fuel_options = [{'label': 'diesel', 'value': 'tipo_gasolina_diesel'},
 {'label': 'petrol', 'value': 'tipo_gasolina_petrol'},
 {'label': 'hybrid_petrol', 'value': 'tipo_gasolina_hybrid_petrol'},
 {'label': 'electro', 'value': 'tipo_gasolina_electro'}]

color_options = [{'label': 'black', 'value': 'color_black'},
 {'label': 'grey', 'value': 'color_grey'},
 {'label': 'white', 'value': 'color_white'},
 {'label': 'red', 'value': 'color_red'},
 {'label': 'silver', 'value': 'color_silver'},
 {'label': 'blue', 'value': 'color_blue'},
 {'label': 'orange', 'value': 'color_orange'},
 {'label': 'beige', 'value': 'color_beige'},
 {'label': 'brown', 'value': 'color_brown'},
 {'label': 'green', 'value': 'color_green'}]

tipo_coche_options = [{'label': 'convertible', 'value': 'tipo_coche_convertible'},
 {'label': 'coupe', 'value': 'tipo_coche_coupe'},
 {'label': 'estate', 'value': 'tipo_coche_estate'},
 {'label': 'hatchback', 'value': 'tipo_coche_hatchback'},
 {'label': 'sedan', 'value': 'tipo_coche_sedan'},
 {'label': 'subcompact', 'value': 'tipo_coche_subcompact'},
 {'label': 'suv', 'value': 'tipo_coche_suv'},
 {'label': 'van', 'value': 'tipo_coche_van'}]

extras_options = [{'label': 'volante_regulable', 'value': 'volante_regulable'},
 {'label': 'aire_acondicionado', 'value': 'aire_acondicionado'},
 {'label': 'camara_trasera', 'value': 'camara_trasera'},
 {'label': 'asientos_traseros_plegables',
  'value': 'asientos_traseros_plegables'},
 {'label': 'elevalunas_electrico', 'value': 'elevalunas_electrico'},
 {'label': 'bluetooth', 'value': 'bluetooth'},
 {'label': 'gps', 'value': 'gps'},
 {'label': 'alerta_lim_velocidad', 'value': 'alerta_lim_velocidad'}]

model_features = ['modelo_520',
 'color_black',
 'modelo_635',
 'tipo_coche_van',
 'modelo_120',
 'color_grey',
 'modelo_i8',
 'tipo_coche_sedan',
 'modelo_118',
 'gps',
 'modelo_X3',
 'modelo_123',
 'modelo_328',
 'color_orange',
 'modelo_318 Gran Turismo',
 'bluetooth',
 'modelo_M4',
 'modelo_318',
 'modelo_X4',
 'modelo_M3',
 'modelo_520 Gran Turismo',
 'camara_trasera',
 'modelo_425',
 'modelo_216 Gran Tourer',
 'asientos_traseros_plegables',
 'modelo_316',
 'color_silver',
 'tipo_coche_convertible',
 'alerta_lim_velocidad',
 'modelo_X6',
 'tipo_coche_hatchback',
 'modelo_435',
 'modelo_M550',
 'modelo_435 Gran Coupé',
 'modelo_640 Gran Coupé',
 'modelo_523',
 'modelo_650',
 'modelo_135',
 'modelo_325',
 'modelo_640',
 'tipo_gasolina_hybrid_petrol',
 'modelo_M135',
 'modelo_220 Active Tourer',
 'year',
 'antiquity',
 'modelo_ActiveHybrid 5',
 'modelo_335 Gran Turismo',
 'modelo_525',
 'modelo_X5',
 'modelo_430 Gran Coupé',
 'volante_regulable',
 'tipo_gasolina_diesel',
 'modelo_320 Gran Turismo',
 'modelo_430',
 'modelo_730',
 'modelo_535',
 'modelo_225',
 'modelo_530 Gran Turismo',
 'tipo_coche_subcompact',
 'tipo_coche_coupe',
 'modelo_M5',
 'modelo_216',
 'modelo_M235',
 'modelo_218',
 'modelo_535 Gran Turismo',
 'modelo_ Active Tourer',
 'modelo_X6 M',
 'modelo_325 Gran Turismo',
 'color_white',
 'tipo_gasolina_petrol',
 'modelo_i3',
 'modelo_Z4',
 'modelo_220',
 'modelo_X5 M',
 'tipo_coche_estate',
 'modelo_X5 M50',
 'modelo_420 Gran Coupé',
 'aire_acondicionado',
 'modelo_330 Gran Turismo',
 'color_beige',
 'tipo_coche_suv',
 'modelo_418 Gran Coupé',
 'modelo_330',
 'modelo_530',
 'elevalunas_electrico',
 'modelo_114',
 'modelo_218 Gran Tourer',
 'modelo_528',
 'modelo_740',
 'tipo_gasolina_electro',
 'modelo_X1',
 'modelo_116',
 'color_red',
 'modelo_335',
 'scaled_km',
 'modelo_320',
 'modelo_735',
 'modelo_420',
 'modelo_125',
 'modelo_218 Active Tourer',
 'color_blue',
 'modelo_518',
 'modelo_225 Active Tourer',
 'color_green',
 'modelo_214 Gran Tourer',
 'modelo_750',
 'scaled_power',
 'modelo_630',
 'modelo_216 Active Tourer',
 'color_brown']


app.index_string = html_string

app.layout = html.Div([
    html.Button('Estimate Price', id='estimate_price', n_clicks=0),
    html.Div(
        children = [
            html.H2( 'The estimated price is: {} €'.format( 100.000 ))
    ]),

    html.Div( id='dropdowns',
        children = [
        html.Div(
            children = [
            html.H3 (' Model: '),
            dcc.Dropdown(
                id='model_dropdown',
                options = model_options,
                value = model_options[0]['value']
            )
            ], style={'width': '25%', 'display': 'inline-block', 'padding':'10px'}
        ),
        html.Div(
            children = [
            html.H3 (' Type of Fuel: '),
            dcc.Dropdown(
                id='fuel_dropdown',
                options = fuel_options,
                value = fuel_options[0]['value']
            )
            ], style={'width': '25%', 'display': 'inline-block', 'padding':'10px'}
        ),
        html.Div(
            children = [
            html.H3 (' Type of car: '),
            dcc.Dropdown(
                id='typecar_dropdown',
                options = tipo_coche_options,
                value = tipo_coche_options[0]['value']
            )
            ], style={'width': '25%', 'display': 'inline-block', 'padding':'10px'}
        ),
        html.Div(
            children = [
            html.H3 (' Color: '),
            dcc.Dropdown(
                id='color_dropdown',
                options = color_options,
                value = color_options[0]['value']
            )
            ], style={'width': '25%', 'display': 'inline-block', 'padding':'10px'}
        )
    ]),

    html.Div( id='numeric',
        children = [
        html.Div(
            children = [
            html.H3 (' Mileage (km): '),
            dcc.Input(id="input_km", type="number", placeholder="")
            ], style={'width': '33%', 'display': 'inline-block'}
        ),
        html.Div(
            children = [
            html.H3 (' Power (CV): '),
            dcc.Input(id="input_power", type="number", placeholder="")
            ], style={'width': '33%', 'display': 'inline-block'}
        ),
        html.Div(
            children = [
            html.H3 (' Antiquity (years): '),
            dcc.Input(id="input_antiquity", type="number", placeholder="")
            ], style={'width': '33%', 'display': 'inline-block'}
        )
        ]
    ),

    html.H2('Extras:'),
    html.Div(
        dcc.Checklist(
        id = 'checklist1',
        options = extras_options[:4],
        labelStyle={'display': 'inline-block', 'padding':'10px'}
        ),
    ),
    html.Div(
        dcc.Checklist(
        id = 'checklist2',
        options = extras_options[4:],
        labelStyle={'display': 'inline-block', 'padding':'10px'}
        ),
    ),
    html.Div( id = 'seleccion')

])

@app.callback(
    Output('seleccion', 'children'),
    Input('estimate_price', 'n_clicks'),
    State('model_dropdown', 'value'),
    State('fuel_dropdown', 'value'),
    State('typecar_dropdown', 'value'),
    State('color_dropdown', 'value'),
    State('input_km', 'number'),
    State('input_power', 'number'),
    State('input_antiquity', 'number'),
    State('checklist1', 'value'),
    State('checklist2', 'value'),
)
def set_display_children(click, model, fuel, typecar, color, km, power, antiquity, extras1, extras2):
    #print(click)
    x_test = np.array([ 0 for i in range(len( model_features ))], dtype='float64')
    
    # Model
    index = model_features.index(model)
    x_test[index] = 1

    # Fuel
    index = model_features.index(fuel)
    x_test[index] = 1

    # Typecar
    index = model_features.index(typecar)
    x_test[index] = 1

    # Color
    index = model_features.index(color)
    x_test[index] = 1

    # km
    km_scaler = pickle.load(open('km_scaler.pkl','rb'))

    if km is None:
        km_scaled = 0.5
    else:
        km_scaled = km_scaler.transform( np.array(km).reshape(1, -1) )
        
    index = model_features.index('scaled_km')
    x_test[index] = km_scaled

    # Power
    power_scaler = pickle.load(open('power_scaler.pkl','rb'))
    if power is None:
        power = 100

    power_scaled = power_scaler.transform( np.array(power).reshape(1, -1) )
    index = model_features.index('scaled_power')
    x_test[index] = power_scaled

    # Antiquity
    if antiquity is None:
        antiquity = 2

    index = model_features.index('antiquity')
    x_test[index] = antiquity

    # Extras
    if extras1 is not None:
        #print(extras1)
        for extra in extras1:
            index = model_features.index(extra)
            x_test[index] = 1
    if extras2 is not None:
        #print(extras2)
        for extra in extras2:
            index = model_features.index(extra)
            x_test[index] = 1

    print(x_test)

    print(model)
    
    filename = 'bmw_price_prediction_model.sav'
    model = pickle.load(open(filename, 'rb'))
    result = model.predict(x_test.reshape(1, -1))

    print(result[0])

    return 'Estimated Price: {}'.format( str(np.round(result[0])) ) 



if __name__ == '__main__':
    app.run_server(debug = True)

