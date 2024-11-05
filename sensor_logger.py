import dash
from dash.dependencies import Output, Input
from dash import dcc, html
from datetime import datetime
import json
import plotly.graph_objs as go
from collections import deque
from flask import Flask, request
import time

import numpy as np
from scipy.ndimage.interpolation import shift
from scipy.signal import butter, freqz, filtfilt, firwin, iirnotch, lfilter, find_peaks


external_stylesheets = [
    {
        "href": "https://fonts.googleapis.com/css2?"
        "family=Lato:wght@400;700&display=swap",
        "rel": "stylesheet",
    },
]


server = Flask(__name__)
app = dash.Dash(__name__, server=server,  external_stylesheets=external_stylesheets)


# define maximum number of data points in the queue
# Decrease this number to improve performance
MAX_DATA_POINTS = 1000

# define how often the plot is updated in ms
# Increase this number to improve performance
UPDATE_FREQ_MS = 1000

# Store accelerometer data
accel_time = deque(maxlen=MAX_DATA_POINTS)
accel_x = deque(maxlen=MAX_DATA_POINTS)
accel_y = deque(maxlen=MAX_DATA_POINTS)
accel_z = deque(maxlen=MAX_DATA_POINTS)

# Store uncalibrated accelerometer data
accel_uncali_time = deque(maxlen=MAX_DATA_POINTS)
accel_uncali_x = deque(maxlen=MAX_DATA_POINTS)
accel_uncali_y = deque(maxlen=MAX_DATA_POINTS)
accel_uncali_z = deque(maxlen=MAX_DATA_POINTS)

# Steps
total_steps = 0
step_init = 0
stepvals =[]
available_sensor_list = []
last_update_time = time.time()


app.layout = html.Div(
	[
		html.Div(
			children=[
				html.H1(children="CSCI 4950/6950 - Live Sensor Readings", className="header-title"),
				html.P(children=["Streamed from Sensor Logger: tszheichoi.com/sensorlogger", html.Br(),
					"Refer python code for implementation."], className="header-description"),
			],
			className="header",
		),
		
		html.Div(id="available_sensor_text", className="wrapper",),
		html.Div(id="steps", className="wrapper",),
		html.Div(id="graph_container", className="wrapper",),
		dcc.Interval(id="counter", interval=UPDATE_FREQ_MS),
	]
)

prev_peak_index = None
@app.callback(Output("graph_container", "children"),
Output("available_sensor_text", "children"),
Output("steps", "children"),
Input("counter", "n_intervals"))
def update_graph(_counter):
   
	global total_steps,step_init,stepvals,filtered_signal, prev_peak_index
	
	graphs = []
   



	# Plot accelerometer if available
	if (len(accel_time) > 0):

		data_accel = [
			go.Scatter(x=list(accel_time), y=list(d), name=name)
			for d, name in zip([accel_x, accel_y, accel_z], ["X", "Y", "Z"])
		]

		graphs.append(
			html.Div(
				dcc.Graph(
					id="accel_graph",
					figure={
						"data": data_accel,
						"layout": go.Layout(
							{
								"title": "Accelerometer",
								"xaxis": {"type": "date", "range": [min(accel_time), max(accel_time)]},
								"yaxis": {"title": "Acceleration ms<sup>-2</sup>", "range": [-25,25]},
							}
						)

					}
				),
            	className="card",
			)
		)
		

	# Plot uncalibrated accelerometer if available
	if (len(accel_uncali_time) > 0):

		data_accel_uncali = [
			go.Scatter(x=list(accel_uncali_time), y=list(d), name=name)
			for d, name in zip([accel_uncali_x, accel_uncali_y, accel_uncali_z], ["X", "Y", "Z"])
		]

		graphs.append(
			html.Div(
				dcc.Graph(
					id="accel_uncali_graph",
					figure={
						"data": data_accel_uncali,
						"layout": go.Layout(
							{
								"title": "Uncalibrated Accelerometer",
								"xaxis": {"type": "date", "range": [min(accel_uncali_time), max(accel_uncali_time)]},
								"yaxis": {"title": "Acceleration ms<sup>-2</sup>","range": [-25,25]},
							}
						)

					}
				),
				className="card",
			)
		)

	# Plot filtered_signal if available
	if len(accel_time) > 0:
            accel_magnitude = [((x**2 + y**2 + z**2)**0.5) for x, y, z in zip(accel_x, accel_y, accel_z)]
            data_accel_mag = [
                go.Scatter(x=list(accel_time), y=accel_magnitude, name="Magnitude")
            ]

            graphs.append(
                html.Div(
                    dcc.Graph(
                        id="accel_graph",
                        figure={
                            "data": data_accel_mag,
                            "layout": go.Layout(
                                {
                                    "title": "Accelerometer Magnitude",
                                    "xaxis": {"type": "date", "range": [min(accel_time), max(accel_time)]},
                                    "yaxis": {"title": "Acceleration Magnitude (ms<sup>-2</sup>)", "range": [-25, 25]},
                                }
                            )
                        }
                    ),
                    className="card",
                )
            )

	# Update text for available sensors.
	text_div = html.Div(
		html.P(children="Available Sensors: {}".format(available_sensor_list)),
		className="textcard",
	)




    ### TODOS: Modify this code for step counting using acclerometer data ###################################
    # total_steps = count the total number of steps
    # step_vals is the filtered signal with values only at the positions in which steps are detected
	# modify the accelerometer input to no array for example, np.array(list(accel_x)) before processing

	cutoff_freq = 0.1  # Adjust this value as needed

	# Design the high-pass filter
	b, a = butter(8, cutoff_freq, 'low')

	#prev_peak_index = None
	stepvals = None
	#total_steps = 0
	threshold_value = 0.5

	# Apply the filter to the signal
	if len(accel_magnitude) > 0:
		filtered_signal = filtfilt(b, a, accel_magnitude)
		
		#Find peaks
		peaks, _ = find_peaks(filtered_signal, distance=20)
		
		# Count the number of steps
		if prev_peak_index is not None:
			# Count the number of new peaks since the last update
			new_peaks = [peak for peak in peaks if peak > prev_peak_index and filtered_signal[peak] > threshold_value]
			total_steps += len(new_peaks)
		
		# Update previous peak index for the next iteration
		if len(peaks) > 0:
			prev_peak_index = peaks[-1]
		
		#Update step vals
		stepvals = [filtered_signal[i] if i in peaks else None for i in range(len(filtered_signal))]
	else:
		# If no data available, reset step values
		stepvals = [None] * len(accel_magnitude)
		prev_peak_index = None
	
	# Update total_steps with the current count
	total_steps = total_steps
	
	
	
	
	
	

	#filtered_signal = np.array(list(accel_x)) # placeholder value, modify this
	#steps = [None] * len(filtered_signal) 
	#stepvals = steps
	#total_steps = len(steps)

    #######################################################################################################

	steps_div = html.Div(
		html.P(children="Number of Steps: {}".format(total_steps)),
		className="textcard",
	)


	if (len(filtered_signal) > 0):

		data_accel_uncali = [
			go.Scatter(x=list(accel_uncali_time), y=list(d), name=name, mode = m)
			for d, name,m in zip([  filtered_signal,stepvals ], ["magnitude","steps"],['lines','markers+text'])
		]

		graphs.append(
			html.Div(
				dcc.Graph(
					id="accel_uncali_graph",
					figure={
						"data": data_accel_uncali,
						"layout": go.Layout(
							{
								"title": "Filtered signal",
								"xaxis": {"type": "date", "range": [min(accel_uncali_time), max(accel_uncali_time)]},
								"yaxis": {"title": "Acceleration ms<sup>-2</sup>","range": [-25,25]},
							}
						)

					}
				),
				className="card",
			)
		)

	return html.Div(graphs), text_div, steps_div
	# return text


@server.route("/data", methods=["POST"])

def data():  # listens to the data streamed from the sensor logger
	global last_update_time
	global available_sensor_list

	if str(request.method) == "POST":
		
		# Print received data
		# print(f'received data: {request.data}')

		# reset available sensor after 10 seconds
		if time.time() - last_update_time > 10:
			last_update_time = time.time()
			available_sensor_list = []
		
		# Read in data
		data = json.loads(request.data)
		
		for d in data['payload']:
			
			# Get sensor name
			sensor_name = d.get("name", None)

			if sensor_name not in available_sensor_list:
				available_sensor_list.append(sensor_name)

			# Read accelerometer sensor data value
			# modify to access different sensors
			if (sensor_name == "accelerometer"):  
				ts = datetime.fromtimestamp(d["time"] / 1000000000)
				if len(accel_time) == 0 or ts > accel_time[-1]:
					accel_time.append(ts)
					# modify the following based on which sensor is accessed, log the raw json for guidance
					accel_x.append(d["values"]["x"])
					accel_y.append(d["values"]["y"])
					accel_z.append(d["values"]["z"])


			if (sensor_name == "accelerometeruncalibrated"):  
				ts = datetime.fromtimestamp(d["time"] / 1000000000)
				if len(accel_uncali_time) == 0 or ts > accel_uncali_time[-1]:
					accel_uncali_time.append(ts)
					# modify the following based on which sensor is accessed, log the raw json for guidance
					accel_uncali_x.append(d["values"]["x"])
					accel_uncali_y.append(d["values"]["y"])
					accel_uncali_z.append(d["values"]["z"])


	return "success"


if __name__ == "__main__":
	# app.run_server(port=8000, host="0.0.0.0", debug=True)
	app.run_server(port=8000, host="0.0.0.0")
