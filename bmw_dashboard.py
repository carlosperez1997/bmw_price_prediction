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

        #estimate_price {
            padding: 15px;
            border-radius: 15px;
            margin-bottom: 10px;
            font-size: 1.3rem
        }


        body {
            font-family: 'Roboto', sans-serif;
            line-height: 1.3;  
            height: 100%;
        }

        .text-primary {
            color: #3c81af;
        }
        
        /* Intro */
        #intro {
            padding: 0.5rem 1rem;
            width: 70%;
            margin: auto;
        }

        #intro p {
            padding: 0.7rem;   
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
            margin-top: 40px;
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
            <p> This website predicts the selling price of a second hand BMW given features 
                like the model, power, mileage, colour, type of car, type of fuel or even if 
                it has extras (gps, blueetooth, rear camera, air conditioning...) </p>
            <p> The model implemented is a Random Forest Regressor with: n_estimators: 80, max_depth: 50 and min_samples_leaf: 5. You can see the model selection in a notebook in GitHub: <a href="https://github.com/carlosperez1997/bmw_price_prediction/blob/main/price_prediction.ipynb" target="_blank"> Building the Model</a></p>

            <p> The code of this dashboard is in: <a href="https://github.com/carlosperez1997/bmw_price_prediction/blob/main/bmw_dashboard.py" target="_blank"> Building the Dashboard </a>  </p>
        
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

model_options = [{'label': 'i8', 'value': 8},
 {'label': 'M4', 'value': 5},
 {'label': 'X6 M', 'value': 6},
 {'label': 'X5 M50', 'value': 5},
 {'label': 'M5', 'value': 6},
 {'label': '640 Gran Coupé', 'value': 6},
 {'label': 'X5 M', 'value': 5},
 {'label': '740', 'value': 7},
 {'label': '750', 'value': 7},
 {'label': 'M3', 'value': 4},
 {'label': 'M550', 'value': 6},
 {'label': 'X6', 'value': 6},
 {'label': '640', 'value': 6},
 {'label': '435 Gran Coupé', 'value': 4},
 {'label': 'X4', 'value': 4},
 {'label': '435', 'value': 4},
 {'label': '425', 'value': 4},
 {'label': 'X5', 'value': 5},
 {'label': '430', 'value': 4},
 {'label': 'M235', 'value': 3},
 {'label': '430 Gran Coupé', 'value': 4},
 {'label': 'M135', 'value': 2},
 {'label': '330 Gran Turismo', 'value': 3},
 {'label': '535 Gran Turismo', 'value': 5},
 {'label': '335 Gran Turismo', 'value': 3},
 {'label': '420 Gran Coupé', 'value': 4},
 {'label': '420', 'value': 4},
 {'label': '220', 'value': 2},
 {'label': '730', 'value': 7},
 {'label': '535', 'value': 5},
 {'label': '135', 'value': 1},
 {'label': '335', 'value': 3},
 {'label': 'i3', 'value': 3},
 {'label': 'ActiveHybrid 5', 'value': 5},
 {'label': '530 Gran Turismo', 'value': 5},
 {'label': '418 Gran Coupé', 'value': 4},
 {'label': '325 Gran Turismo', 'value': 3},
 {'label': '528', 'value': 5},
 {'label': '520 Gran Turismo', 'value': 5},
 {'label': '530', 'value': 5},
 {'label': '635', 'value': 6},
 {'label': '225 Active Tourer', 'value': 2},
 {'label': '218', 'value': 2},
 {'label': ' Active Tourer', 'value': 1},
 {'label': '225', 'value': 2},
 {'label': 'X3', 'value': 3},
 {'label': '214 Gran Tourer', 'value': 2},
 {'label': '320 Gran Turismo', 'value': 3},
 {'label': '216 Gran Tourer', 'value': 2},
 {'label': '330', 'value': 3},
 {'label': '328', 'value': 3},
 {'label': '518', 'value': 5},
 {'label': '218 Gran Tourer', 'value': 2},
 {'label': '520', 'value': 5},
 {'label': '525', 'value': 5},
 {'label': '218 Active Tourer', 'value': 2},
 {'label': '318 Gran Turismo', 'value': 3},
 {'label': '325', 'value': 3},
 {'label': '216 Active Tourer', 'value': 2},
 {'label': 'X1', 'value': 1},
 {'label': '125', 'value': 1},
 {'label': '120', 'value': 1},
 {'label': '320', 'value': 3},
 {'label': '220 Active Tourer', 'value': 2},
 {'label': '114', 'value': 1},
 {'label': '318', 'value': 3},
 {'label': '630', 'value': 6},
 {'label': '316', 'value': 3},
 {'label': '116', 'value': 1},
 {'label': '118', 'value': 1},
 {'label': 'Z4', 'value': 4},
 {'label': '123', 'value': 1},
 {'label': '650', 'value': 6},
 {'label': '523', 'value': 5},
 {'label': '216', 'value': 2},
 {'label': '735', 'value': 7}]

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

model_features = ['tipo_coche_coupe',
 'modelo_M',
 'camara_trasera',
 'year',
 'gps',
 'tipo_coche_van',
 'aire_acondicionado',
 'scaled_power',
 'tipo_coche_estate',
 'elevalunas_electrico',
 'tipo_coche_convertible',
 'asientos_traseros_plegables',
 'bluetooth',
 'tipo_coche_suv',
 'km',
 'modelo_ordinal',
 'alerta_lim_velocidad',
 'tipo_coche_sedan',
 'antiquity',
 'volante_regulable',
 'tipo_coche_subcompact',
 'tipo_coche_hatchback']

app.index_string = html_string

app.layout = html.Div([
    html.Button('Estimate Price', id='estimate_price', n_clicks=0),
    html.Div(
        id = 'solution'
    ),

    html.Div( id='dropdowns', style={'width': '100%','margin':'10px'},
        children = [
        html.Div(
            children = [
            html.H3 (' Model: '),
            dcc.Dropdown(
                id='model_dropdown',
                options = model_options,
                value = model_options[0]['value']
            )
            ], style={'width': '33%', 'display': 'inline-block', 'padding':'0px 100px'}
        ),
        html.Div(
            children = [
            html.H3 (' Type of Fuel: '),
            dcc.Dropdown(
                id='fuel_dropdown',
                options = fuel_options,
                value = fuel_options[0]['value']
            )
            ], style={'width': '33%', 'display': 'inline-block', 'padding':'0px 100px'}
        ),
        html.Div(
            children = [
            html.H3 (' Type of car: '),
            dcc.Dropdown(
                id='typecar_dropdown',
                options = tipo_coche_options,
                value = tipo_coche_options[0]['value']
            )
            ], style={'width': '33%', 'display': 'inline-block', 'padding':'0px 100px'}
        ),
        #html.Div(
        #    children = [
        #    html.H3 (' Color: '),
        #    dcc.Dropdown(
        #        id='color_dropdown',
        #        options = color_options,
        #        value = color_options[0]['value']
        #    )
        #    ], style={'width': '25%', 'display': 'inline-block', 'padding':'10px'}
        #)
    ]),

    html.Div( id='numeric', style={'width': '100%','margin-bottom':'30px'},
        children = [
        html.Div( children = [
            html.H3 (' Mileage (km): '),
            dcc.Slider(
                id='input_km',
                min=10000,
                max=300000,
                step=10000,
                value=50000,
                marks={
                    10000: '10.000',
                    50000: '50.000',
                    100000: '100.000',
                    150000: '150.000',
                    200000: '200.000',
                    250000: '250.000',
                    300000: '300.000',
                },
            ),
        ], style={'width': '33%', 'display': 'inline-block', 'padding':'0px 40px'}),
        html.Div( children = [
            html.H3 (' Power (CV): '),
            dcc.Slider(
                id='input_power',
                min=100,
                max=400,
                step=20,
                value=180,
                marks={i*20+100: '{}'.format(str(i*20+100)) for i in range( int((400-100)/20)+1 )},
            ),
        ], style={'width': '33%', 'display': 'inline-block', 'padding':'0px 40px'}),
        html.Div( children = [
            html.H3 (' Antiquity (years): '),
            dcc.Slider(
                id='input_antiquity',
                min=1,
                max=10,
                step=1,
                value=2,
                marks={i: '{}'.format(i) for i in range( 11 )},
            ),
        ], style={'width': '33%', 'display': 'inline-block', 'padding':'0px 40px'}),
    ]),

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

])

@app.callback(
    Output('solution', 'children'),
    Input('estimate_price', 'n_clicks'),
    State('model_dropdown', 'value'),
    State('fuel_dropdown', 'value'),
    State('typecar_dropdown', 'value'),
    #State('color_dropdown', 'value'),
    State('input_km', 'value'),
    State('input_power', 'value'),
    State('input_antiquity', 'value'),
    State('checklist1', 'value'),
    State('checklist2', 'value'),
)
def set_display_children(click, model, fuel, typecar, km, power, antiquity, extras1, extras2):
    x_test = np.array([ 0 for i in range(len( model_features ))], dtype='float64')
    
    # Model
    index = model_features.index('modelo_ordinal')
    x_test[index] = model

    # Fuel
    #index = model_features.index(fuel)
    #x_test[index] = 1

    # Typecar
    index = model_features.index(typecar)
    x_test[index] = 1

    # Color
    #index = model_features.index(color)
    #x_test[index] = 1

    # km
    km_scaler = pickle.load(open('km_scaler.pkl','rb'))

    if km is None:
        km = 50000
        km_scaled = 0.5
    else:
        km_scaled = km_scaler.transform( np.array(km).reshape(1, -1) )
        
    #index = model_features.index('scaled_km')
    #x_test[index] = km_scaled

    index = model_features.index('km')
    x_test[index] = int(km)

    # Power
    power_scaler = pickle.load(open('power_scaler.pkl','rb'))

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

    #print(x_test)

    filename = 'bmw_price_prediction_model.sav'
    model = pickle.load(open(filename, 'rb'))
    result = model.predict(x_test.reshape(1, -1))

    #print(result[0])

    return html.H2( 'The estimated price is: {} €'.format( str(np.round(result[0]))  ))


if __name__ == '__main__':
    app.run_server(debug = True)

