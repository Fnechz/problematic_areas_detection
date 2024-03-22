# from flask import Flask, render_template, request
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
# from vegetative_indices import calculate_vegetation_indices
import os

# my own additions
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib as mpl
from osgeo import gdal, osr

# Create a class for each vegetation_index
class Indexes:
    def __init__(self, img):
        self.img = img
        self.R = self.img[:, :, 2].astype(np.float32)
        self.G = self.img[:, :, 1].astype(np.float32)
        self.B = self.img[:, :, 0].astype(np.float32)

    # All these operations aim to not have ZeroDivisionError

    # Visible Atmospheric Resistant Index
    def VARI(self):
        vari = np.divide((self.G - self.R), (self.G + self.R - self.B + 0.00001))
        return np.clip(vari, -1, 1)

    # Green Leaf Index
    def GLI(self):
        gli = np.divide((2 * self.G - self.R - self.B), (2 * self.G + self.R + self.B + 0.00001))
        return np.clip(gli, -1, 1)

    # Normalized Green Red Difference Index
    def NGRDI(self):  
        v_ndvi = np.divide((self.G - self.R), (self.G + self.R + 0.00001))
        return np.clip(v_ndvi, -1, 1)

    # Normalized Green Blue Difference Index
    def NGBDI(self): 
        ngbdi = (self.G - self.B) / (self.G + self.B + 0.00001) 
        return np.clip(ngbdi, -1, +1)

    # Identification of the Idx object
    def get_index(self, index_name):
        if index_name == 'VARI':
            return self.VARI()
        elif index_name == 'GLI':
            return self.GLI()
        elif index_name == 'NGRDI':
            return self.NGRDI()
        elif index_name == 'NGBDI':
            return self.NGBDI()
        else:
            print('Unknown index')


# Find the real values of the min, max based on the frequency of the vegetation_index histogram
def find_real_min_max(perc, edges, index_clear):
    mask = perc > (0.05 * len(index_clear))
    edges = edges[:-1]
    min_v = edges[mask].min()
    max_v = edges[mask].max()
    return min_v, max_v

# Function that creates the georeferenced VI map
def array_to_raster(output_path, ds_reference, array, name1, name2):
    rows, cols, band_num = array.shape

    driver = gdal.GetDriverByName("GTiff")

    outRaster = driver.Create(os.path.join(output_path, name1+'_'+name2+'.tif'), cols, rows, band_num, gdal.GDT_Byte, options=["COMPRESS=DEFLATE"])
    originX, pixelWidth, b, originY, d, pixelHeight = ds_reference.GetGeoTransform()
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))

    descriptions = ['Red Band', 'Green Band', 'Blue Band', 'Alpha Band', 'Index Array']
    for b in range(band_num):
        outband = outRaster.GetRasterBand(b+1)
        outband.WriteArray(array[:,:,b])
        outband.SetDescription(descriptions[b])
        if b+1==1:
            outRaster.GetRasterBand(1).SetColorInterpretation(gdal.GCI_RedBand)
        elif b+1==2:
            outRaster.GetRasterBand(2).SetColorInterpretation(gdal.GCI_GreenBand)
        elif b+1==3:
            outRaster.GetRasterBand(3).SetColorInterpretation(gdal.GCI_BlueBand)
        elif b+1==4:
            outRaster.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)
    

    outRasterSRS = osr.SpatialReference(wkt=prj)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    driver = None
    outband.FlushCache()

    # print('Georeferenced {} map was extracted!'.format(index_name))

    return outRaster

# Function to calculate vegetation indices
def calculate_vegetation_indices(input_image_path, output_dir, selected_indices):
    img = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
    h, w, ch = img.shape

    if ch > 3:
        image = img[:, :, :3].astype(float)
        image[img[:, :, 3] == 0] = np.nan
        empty_space = img[:, :, 3] == 0
    else:
        image = img

    Idx = Indexes(image)
    results = {}

    for index_name in selected_indices:
        idx = Idx.get_index(index_name)

        index_clear = idx[~np.isnan(idx)]

        perc, edges, _ = plt.hist(index_clear, bins=100, range=(-1, 1), color='darkcyan', edgecolor='black')
        plt.close()

        lower, upper = find_real_min_max(perc, edges, index_clear)
        index_clipped = np.clip(idx, lower, upper)

        cm = plt.get_cmap('RdYlGn')
        cNorm = mpl.colors.Normalize(vmax=upper, vmin=lower)
        colored_image = cm(cNorm(index_clipped))

        img = Image.fromarray(np.uint8(colored_image * 255), mode='RGBA')
        
        rgba = np.array(img, dtype=np.float32)

        ds = gdal.Open(input_image_path, gdal.GA_ReadOnly)
        prj = ds.GetProjection()

        if prj: 
            array_to_raster(output_dir, ds, rgba, os.path.splitext(os.path.basename(input_image_path))[0], index_name)    
        else:
            img.save(os.path.join(output_dir, '{}_{}.tif'.format(os.path.splitext(os.path.basename(input_image_path))[0], index_name)))
            print('Non georeferrenced {} map was extracted!'.format(index_name))

        results[index_name] = index_clipped

        # do the npy stuff (saving.. for use latter)
        # np.save(os.path.join(output_dir,'{}_{}.npy'.format(os.path.splitext(os.path.basename(input_image_path))[0], index_name), index_clipped))
        np.save('{}/{}_{}.npy'.format(output_dir, os.path.splitext(os.path.basename(input_image_path))[0], index_name), index_clipped)


    print('Done!')

    return results


# end of my own additions


# start of problematic areas detection logic
import numpy as np
import cv2
from scipy import io
import os
import glob
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import morphology, measure, filters
import re
import argparse
from osgeo import gdal, osr
import json
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score

def threshold_index(index, tresh_value):
    _, areas_mask = cv2.threshold(index, tresh_value, 1, cv2.THRESH_BINARY_INV)
    areas_mask[areas_mask > 0] = 255
    return areas_mask

def find_optimal_K(centers, num_pixels):
	metric = 'calinski'
	scores = []
	if np.shape(centers)[0] < 10:
		clusters = np.arange(2, np.shape(centers)[0])
	else:
		clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
	for n_clusters in clusters:
		cluster = KMeans(n_clusters=n_clusters, random_state=10)
		cluster_labels = cluster.fit_predict(centers, sample_weight=num_pixels[1:])
		if metric == 'silhouette':
			scores.append(silhouette_score(centers, cluster_labels))
		elif metric == 'calinski':
			scores.append(calinski_harabasz_score(centers, cluster_labels))
		else:
			print('Unknown score')

	opt_K = clusters[np.argmax(scores)]
	return opt_K

def find_areas(index, img_path, save_dir):
    index_array = np.load(index)
    index = index_array.astype(np.float32)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    h, w, _ = img.shape
    empty_space = np.isnan(index)

    index_clear = index[~np.isnan(index)]
    lower = np.min(index_clear)
    upper = np.max(index_clear)

    mask = threshold_index(index, (lower + (upper + lower) / 2) / 2)
    mask = (mask / 255).astype(np.uint8)

    # mask = ndimage.morphology.binary_closing(mask, np.ones((5, 5)), iterations=1).astype(np.uint8)
    # mask = ndimage.morphology.binary_opening(mask, np.ones((5, 5)), iterations=1).astype(np.uint8)

    mask = ndimage.binary_closing(mask, np.ones((5, 5)), iterations=1).astype(np.uint8)
    mask = ndimage.binary_opening(mask, np.ones((5, 5)), iterations=1).astype(np.uint8)
  




    mask_white = np.copy(mask)
    mask_white[empty_space] = 1

    labels, nlabels = measure.label(mask, connectivity=2, background=0, return_num=True)
    centers = np.array([ndimage.center_of_mass(mask, labels, i + 1) for i in range(nlabels)])
    _, num_pixels = np.unique(labels, return_counts=True)
    
    opt_K = find_optimal_K(centers, num_pixels)
    kmeans = KMeans(n_clusters=opt_K, random_state=0).fit(centers, sample_weight=num_pixels[1:])
    centers_cluster = kmeans.cluster_centers_

    spot_size = np.sqrt(np.shape(index)[0] ** 2 + np.shape(index)[1] ** 2)

    # plot had ugly dots had to edit:
    spot_size = np.sqrt(np.shape(index)[0] ** 2 + np.shape(index)[1] ** 2)
    f = plt.figure()
    f.set_figheight(index.shape[0] / f.get_dpi())
    f.set_figwidth(index.shape[1] / f.get_dpi())
    ax = plt.Axes(f, [0., 0., 1., 1.])
    ax.set_axis_off()
    f.add_axes(ax)
    ax.imshow(np.clip(index, lower, upper), cmap="RdYlGn", aspect='auto')
    ax.scatter(centers_cluster[:, 1], centers_cluster[:, 0], s=0.5 * spot_size, c='dodgerblue', edgecolors='black', linewidth=5)
    f.savefig(os.path.join(save_dir, 'centers.png'), transparent=True)

    # plt.figure()
    # plt.imshow(np.clip(index, lower, upper), cmap="RdYlGn", aspect='auto')
    # plt.scatter(centers_cluster[:, 1], centers_cluster[:, 0], s=0.5 * spot_size, c='dodgerblue', edgecolors='black', linewidth=5)
    # plt.savefig(os.path.join(save_dir, 'centers.png'), transparent=True)
    # plt.close()

    return centers_cluster


def find_Lat_Lon(raster, centers):
    xoff, a, b, yoff, d, e = raster.GetGeoTransform()
    crs = osr.SpatialReference()
    crs.ImportFromWkt(raster.GetProjectionRef())
    
    crsGeo = osr.SpatialReference()
    crsGeo.ImportFromProj4('+proj=longlat +datum=WGS84 +no_defs')
    transform = osr.CoordinateTransformation(crs, crsGeo)

    centers_lat_lon = []
    for x_pixel, y_pixel in centers:
        xp = a * x_pixel + b * y_pixel + a * 0.5 + b * 0.5 + xoff
        yp = d * x_pixel + e * y_pixel + d * 0.5 + e * 0.5 + yoff
        (lat, lon, z) = transform.TransformPoint(xp, yp)
        centers_lat_lon.append([lat, lon])

    centers_geo = np.array(centers_lat_lon)
    centers = np.fliplr(centers_geo)
    return centers

def process_problematic_areas(input_image, index, output_dir):
    ds = gdal.Open(input_image, gdal.GA_ReadOnly)
    centers_clusters = find_areas(index, input_image, output_dir)

    data = []
    prj = ds.GetProjection()

    if prj: 
        centers_geo = find_Lat_Lon(ds, centers_clusters)
        for center in centers_geo:
            data.append({"Lattitude": center[0], "Longtitude": center[1]})
    else:
        centers_pixel_level = centers_clusters
        for center in centers_pixel_level:
            data.append({"X_pixel": center[1], "Y_pixel": center[0]})

    with open(os.path.join(output_dir, 'result.json'), "w") as file:
        json.dump(data, file, indent=4)

    print('Done!')

# Example usage
# process_problematic_areas('path/to/image.tif', 'path/to/index.npy', 'path/to/output')
    
# print("testing 123.....")

# index_path = "/Users/icom/Desktop/temp_dir/EP-11-29590_0007_0013_VARI.npy"
# image_path = "/Users/icom/Desktop/temp_dir/EP-11-29590_0007_0013_VARI.tif"
# save_directory = "/Users/icom/Desktop/test_prob"

# find_areas(index_path, image_path,save_directory)

# print("Done!")
# end of problematic areas detection logic














app = Flask(__name__)
mpl.use('agg')
# Set the upload folder and allowed extensions
# UPLOAD_FOLDER = 'uploads'
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_file():

#     if 'file' not in request.files:
#         return redirect(request.url)
#     file = request.files['file']
#     if file.filename == '':
#         return redirect(request.url)
#     # Check if the post request has the file part
#     # if 'file' not in request.files:
#     #     flash('No file part')
#     #     return redirect(request.url)
    
#     # file = request.files['file']
    
#     # If user does not select file, browser also submit an empty part without filename
#     if file.filename == '':
#         flash('No selected file')
#         return redirect(request.url)
    
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#         return render_template('index.html', filename=filename)

# @app.route('/process', methods=['POST'])
# def process():
#     filename = request.form['filename']
#     indices = request.form.getlist('indices')
#     input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     output_dir = 'results'
#     results = calculate_vegetation_indices(input_path, output_dir, indices)
#     return render_template('results.html', results=results)

# if __name__ == '__main__':
#     app.run(debug=True)











# from flask import Flask, render_template, request, redirect, url_for, flash
# import os
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from PIL import Image
# import matplotlib as mpl
# from osgeo import gdal, osr

# mpl.use('agg')

# app = Flask(__name__)
# app.secret_key = 'farainechikwira43'

# # Create a class for each vegetation_index
# class Indexes:
#     def __init__(self, img):
#         self.img = img
#         self.R = self.img[:, :, 2].astype(np.float32)
#         self.G = self.img[:, :, 1].astype(np.float32)
#         self.B = self.img[:, :, 0].astype(np.float32)

#     # Visible Atmospheric Resistant Index
#     def VARI(self):
#         vari = np.divide((self.G - self.R), (self.G + self.R - self.B + 0.00001))
#         return np.clip(vari, -1, 1)

#     # Green Leaf Index
#     def GLI(self):
#         gli = np.divide((2 * self.G - self.R - self.B), (2 * self.G + self.R + self.B + 0.00001))
#         return np.clip(gli, -1, 1)

#     # Normalized Green Red Difference Index
#     def NGRDI(self):
#         v_ndvi = np.divide((self.G - self.R), (self.G + self.R + 0.00001))
#         return np.clip(v_ndvi, -1, 1)

#     # Normalized Green Blue Difference Index
#     def NGBDI(self):
#         ngbdi = (self.G - self.B) / (self.G + self.B + 0.00001)
#         return np.clip(ngbdi, -1, +1)

#     # Get the desired index
#     def get_index(self, index_name):
#         if index_name == 'VARI':
#             return self.VARI()
#         elif index_name == 'GLI':
#             return self.GLI()
#         elif index_name == 'NGRDI':
#             return self.NGRDI()
#         elif index_name == 'NGBDI':
#             return self.NGBDI()
#         else:
#             print('Unknown index')


# # Function to process uploaded image and calculate indices
# def process_image(file, output_path, indices):
#     img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
#     h, w, ch = img.shape

#     if ch > 3:
#         image = img[:, :, :3].astype(float)
#         image[img[:, :, 3] == 0] = np.nan
#         empty_space = img[:, :, 3] == 0
#     else:
#         image = img

#     Idx = Indexes(image)

#     for index_name in indices:
#         idx = Idx.get_index(index_name)

#         # Your calculation logic and saving here
#         # For example:
#         # Calculate index histogram
#         perc, edges, _ = plt.hist(idx[~np.isnan(idx)], bins=100, range=(-1, 1), color='darkcyan', edgecolor='black')
#         plt.close()

#         # Find the real min, max values of the vegetation_index
#         lower, upper = perc[0], perc[-1]
#         index_clipped = np.clip(idx, lower, upper)

#         # Convert to RGBA image
#         cm = plt.get_cmap('RdYlGn')
#         cNorm = mpl.colors.Normalize(vmax=upper, vmin=lower)
#         colored_image = cm(cNorm(index_clipped))
#         img_rgba = Image.fromarray(np.uint8(colored_image * 255), mode='RGBA')

#         # Save the image
#         img_name = os.path.splitext(file.filename)[0]
#         output_filename = f"{img_name}_{index_name}.tif"
#         output_filepath = os.path.join(output_path, output_filename)
#         img_rgba.save(output_filepath)


# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             flash('No file part')
#             return redirect(request.url)

#         file = request.files['file']
#         indices = request.form.getlist('indices')
#         output_path = request.form['output_path']

#         if file.filename == '':
#             flash('No selected file')
#             return redirect(request.url)

#         if file and indices and output_path:
#             # process_image(file, output_path, indices)
#             print("at least we reached this logic")
#             print(file)
#             print("####")
#             print(file.filename)
#             calculate_vegetation_indices(file, output_path, indices)
#             flash('Indices calculated and saved successfully')
#             return redirect(url_for('index'))

#     return render_template('index.html')


# if __name__ == '__main__':
#     app.run(debug=True)


# def process():
#     filename = request.form['filename']
#     indices = request.form.getlist('indices')
#     input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     output_dir = 'results'
#     results = calculate_vegetation_indices(input_path, output_dir, indices)
#     return render_template('results.html', results=results)
# from flask_ngrok import run_with_ngrok
from flask_cors import CORS

app = Flask(__name__)
app.secret_key = 'oianjdfkmroapoaok300540943jr0fmrjg0erjkk04lkfdkvei0394494950w30fdir0303id0'
CORS(app, origins=['http://localhost:8000'])
# run_with_ngrok(app)

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         flash('No file part')
#         return redirect(request.url)
#     file = request.files['file']
#     if file.filename == '':
#         flash('No selected file')
#         return redirect(request.url)
#     if file:
#         # Instead of saving the file, directly process it
#         selected_indices = request.form.getlist('index')
#         output_dir = request.form['output_dir']
#         img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#         file.save(img_path)
#         results = calculate_vegetation_indices(img_path, output_dir, selected_indices)
#         return render_template('results.html', results=results)

# if __name__ == '__main__':
#     app.config['UPLOAD_FOLDER'] = os.getcwd()  # Set the upload folder to the current working directory
#     app.run()

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' in request.files:  # Vegetative indices calculation form submitted
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            selected_indices = request.form.getlist('index')
            output_dir_vegetative = request.form['output_dir']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(img_path)
            results_vegetative = calculate_vegetation_indices(img_path, output_dir_vegetative, selected_indices)
            return render_template('results.html', results=results_vegetative)
    
    elif 'input_image' in request.files:  # Problematic areas detection form submitted
        if 'input_image' not in request.files or 'index_file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        input_image = request.files['input_image']
        index_file = request.files['index_file']
        if input_image.filename == '' or index_file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if input_image and index_file:
            output_dir_problematic = request.form['output_dir']
            input_image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(input_image.filename))
            index_file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(index_file.filename))
            input_image.save(input_image_path)
            index_file.save(index_file_path)
            find_areas(index_file_path, input_image_path, output_dir_problematic)
            # return redirect(url_for('show_results'))  # Assuming there's a route for showing results
    
    flash('Error uploading files')
    return "Done"

@app.route('/results')
def show_results():
    # Render the results page
    return render_template('results.html')
    
if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = os.getcwd()  # Set the upload folder to the current working directory
    app.run(debug=True)
