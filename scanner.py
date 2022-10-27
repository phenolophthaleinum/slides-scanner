import cv2
import pathlib
import time
import numpy as np
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, no_update, ctx
import json
import figureObject
import dash_bootstrap_components as dbc
from joblib import Parallel, delayed


# TODO: clean up code
imgs = list(pathlib.Path("slides_test").rglob("*"))
imgs = [str(x) for x in imgs]
# processed_imgs = []
output_dir = "slides_processed_test"
print(imgs)
print(f"{len(imgs)} images will be processed")
current_image = []
current_dims = {}
# processed_obj = []
missed_obj = []
current_index = 0
rotate_flag = False


def image_processing(image):
    """
    Parallel processing of input images
    Args:
        image (str): String path to image

    Returns: [str, Figure, list[str]]: `str` is a image name; `Figure` is an object that contains information about
    edited photo; `list` contains names of missed images.
    """

    missed_imgs = []
    im_name = image.split("\\")[-1]
    im = cv2.imread(image, -1)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thres = cv2.threshold(im_gray, 175, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    opening = cv2.morphologyEx(thres, cv2.MORPH_OPEN, kernel, iterations=3)
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    if not cnts:
        return None, None, im_name

    image_number = 0

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        # cv2.rectangle(im, (x, y), (x + w, y + h), (36, 255, 12), 3)
        ROI = im[y:y + h, x:x + w]
        print(ROI.shape)
        print(f"x: {x}, y: {y}, w: {w}, h: {h}")
        if ROI.shape[0] * ROI.shape[1] >= 1_500_000:
            # input_points = np.float32([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
            # output_points = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, y - 1]])
            # matrix = cv2.getPerspectiveTransform(input_points, output_points)
            # imgOutput = cv2.warpPerspective(ROI, matrix, (w, h),
            #                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
            # cv2.imwrite(f"{output_dir}/{im_name}_ROI_{image_number}.png", cv2.rotate(imgOutput, cv2.ROTATE_90_CLOCKWISE))
            current_image = im
            current_dims = {
                "x0": x,
                "x1": x + w,
                "y0": y,
                "y1": y + h
            }
            new_fig = figureObject.Figure(im_name, current_image, current_dims)
            # cv2.imwrite(f"{output_dir}/{im_name}_ROI_{image_number}.png", cv2.rotate(ROI, cv2.ROTATE_90_CLOCKWISE))
            image_number += 1
            # processed_imgs.append(im_name)
            # processed_obj.append(new_fig)
            # print(im_name, new_fig, None)
            return im_name, new_fig, None
        else:
            missed_imgs.append(im_name)
            # current_dims = {
            #     "x0": 0,
            #     "x1": 0 + w,
            #     "y0": 0,
            #     "y1": 0 + h
            # }
            # missed_fig = slides_classes.Figure(im_name, im, current_dims)
            # missed_obj.append(missed_fig)
            continue
    # print(None, None, set(missed_imgs))
    return None, None, next(iter(set(missed_imgs)))

    # processed_obj.extend(missed_obj)


# missed = []
tic = time.perf_counter()
# processed_imgs, processed_obj, missed = Parallel(n_jobs=-1, backend='loky')(delayed(image_processing)(image) for image in imgs)
# print(processed_imgs)
# print(processed_obj)
# print(missed)
r = Parallel(n_jobs=-1, backend='loky')(delayed(image_processing)(image) for image in imgs)
# print(r[0])
processed_imgs, processed_obj, missed = zip(*r)
processed_imgs = [x for x in processed_imgs if x is not None]
processed_obj = [x for x in processed_obj if x is not None]
missed = [x for x in missed if x is not None]

# unprocessed_imgs_names = set(missed) - set(processed_imgs)
for img_name in missed:
    i = cv2.imread(f"slides_test/{img_name}", -1)
    h = i.shape[0]
    w = i.shape[1]
    current_dims = {
        "x0": 0,
        "x1": 0 + w,
        "y0": 0,
        "y1": 0 + h
    }
    missed_obj.append(figureObject.Figure(img_name, i, current_dims))
processed_obj.extend(missed_obj)

print(f"Missed images: \n{set(missed) - set(processed_imgs) if set() else None}")
toc = time.perf_counter()
elapsed_time = toc - tic
print(f"elapsed time: {elapsed_time:0.8f} seconds")
# print(processed_obj)


# image_processing()
fig = px.imshow(processed_obj[0].img)
fig.add_shape(editable=True,
              x0=processed_obj[0].dims["x0"],
              x1=processed_obj[0].dims["x1"],
              y0=processed_obj[0].dims["y0"],
              y1=processed_obj[0].dims["y1"],
              xref='x', yref='y',
              line_color='yellow',
              line_width=5)
# fig.update_layout(dragmode="drawrect")
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
app = Dash(__name__, external_stylesheets=[dbc.themes.ZEPHYR, dbc_css])
app.layout = html.Div(
    [html.H1("Slides scanner"),
     html.H3(f"{processed_obj[current_index].name}", id="fig-name"),
     html.H4(f"{current_index + 1}/{len(processed_obj)}", id="fig-num"),
     dcc.Graph(id="pic", figure=fig, style={'width': '90vh', 'height': '90vh'}),
     html.Button('Previous image', id='prev-fig', n_clicks=0),
     html.Button('Confirm area and save', id='save-fig', n_clicks=0),
     html.Button('Next image', id='next-fig', n_clicks=0),
     dcc.Checklist(options=[
         {'label': 'Rotate image while saving', 'value': "rotate"}
     ], id="rotate-check"),
     html.Pre(id="annotations-data", style={'display': 'none'}),
     html.Pre(id="annotations-data2", style={'display': 'none'}),
     html.Pre(id="annotations-data3", style={'display': 'none'})]
)


@app.callback(
    Output("annotations-data", "children"),
    Input("pic", "relayoutData"),
    prevent_initial_call=True,
)
def on_new_annotation(relayout_data):
    # print(json.dumps(relayout_data))
    # for elem in relayout_data:
    #     print(elem)
    if "shapes[0].x0" in next(iter(relayout_data)):
        processed_obj[current_index].dims["x0"] = int(relayout_data["shapes[0].x0"])
        processed_obj[current_index].dims["x1"] = int(relayout_data["shapes[0].x1"])
        processed_obj[current_index].dims["y0"] = int(relayout_data["shapes[0].y0"])
        processed_obj[current_index].dims["y1"] = int(relayout_data["shapes[0].y1"])
        return json.dumps(relayout_data, indent=2)
    else:
        return no_update


@app.callback(
    Output('annotations-data3', 'children'),
    Input('rotate-check', 'value'),
    prevent_initial_call=True
)
def set_rotate(flag):
    global rotate_flag
    if "rotate" in flag:
        rotate_flag = True
    else:
        rotate_flag = False


@app.callback(
    Output('annotations-data2', 'children'),
    Input('save-fig', 'n_clicks'),
    # Input('rotate-check', 'value'),
    prevent_initial_call=True
)
def save_fig(n_clicks):
    print(current_dims)
    print(rotate_flag)
    final_fig = processed_obj[current_index].img[
                processed_obj[current_index].dims["y0"]:processed_obj[current_index].dims["y1"],
                processed_obj[current_index].dims["x0"]:processed_obj[current_index].dims["x1"]
                ]
    # try:
    #     if "rotate" in rotate:
    #         final_fig = cv2.rotate(final_fig, cv2.ROTATE_90_CLOCKWISE)
    # except:
    #     pass
    if rotate_flag:
        final_fig = cv2.rotate(final_fig, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(f"{output_dir}/{processed_obj[current_index].name}_ROI.png", final_fig)
    return f"Saving to file on {n_clicks}"


@app.callback(
    Output('pic', 'figure'),
    Output('fig-name', "children"),
    Output('fig-num', "children"),
    Input('next-fig', 'n_clicks'),
    Input('prev-fig', 'n_clicks'),
    prevent_initial_call=True
)
def nav_fig(n_fig, p_fig):
    button_id = ctx.triggered_id
    global current_index
    maxlen = len(processed_obj)
    minlen = 0
    if button_id == "next-fig" and current_index + 1 < maxlen:
        current_index += 1
    if button_id == "prev-fig" and current_index - 1 >= minlen:
        current_index -= 1
    fig = px.imshow(processed_obj[current_index].img)
    fig.add_shape(editable=True,
                  x0=processed_obj[current_index].dims["x0"],
                  x1=processed_obj[current_index].dims["x1"],
                  y0=processed_obj[current_index].dims["y0"],
                  y1=processed_obj[current_index].dims["y1"],
                  xref='x', yref='y',
                  line_color='yellow',
                  line_width=5)
    return fig, processed_obj[current_index].name, f"{current_index + 1}/{len(processed_obj)}"


if __name__ == "__main__":
    # TODO cmd interface:
    # - threshold for scanning
    # - input folder
    # - output folder
    # - run gui flag
    app.run_server(debug=True)
