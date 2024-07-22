import string
import random
from flask_sqlalchemy import SQLAlchemy
from flask import Flask, render_template, request
from flask_cors import CORS
from flask import Flask, request, send_from_directory, render_template, jsonify
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
from datetime import datetime
matplotlib.use('SVG')

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:secretrootpassword@localhost/kmeans'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

CORS(app)
db = SQLAlchemy(app)


class KmeansHistory(db.Model):
    __tablename__ = 'kmeans_history'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    file_name = db.Column(db.String(255), nullable=False)
    kota = db.Column(db.String(255), unique=True, nullable=False)
    clustering_date = db.Column(db.Date)
    index = db.Column(db.String(255), unique=True, nullable=False)
    nama = db.Column(db.String(255), unique=True, nullable=False)
    cluster = db.Column(db.String(255), unique=True, nullable=False)
    umur = db.Column(db.String(255), unique=True, nullable=False)
    ijazah_tertinggi = db.Column(db.String(255), unique=True, nullable=False)
    jenis_cacat = db.Column(db.String(255), unique=True, nullable=False)
    status_pekerjaan = db.Column(db.String(255), unique=True, nullable=False)
    jumlah_jam_kerja = db.Column(db.String(255), unique=True, nullable=False)
    lapangan_usaha = db.Column(db.String(255), unique=True, nullable=False)
    penyakit_kronis = db.Column(db.String(255), unique=True, nullable=False)


dataset = []
dataset_labels = []
label_desa = []
columns = []
rows = None
cluster = None
max_iteration = None
kmeans = None
dictionary_db = {}
uploadedFileName = ''


@app.route('/')
def index():
    return render_template('index.html')


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.float64):
        return float(obj)
    else:
        return obj


class ClusteringResult():
    def __init__(self) -> None:
        self.iteration = 0
        self.centroids = []
        self.euclideans = []
        self.clusters = []

    def to_dict(self):
        return {
            'iteration': int(self.iteration),
            'centroids': convert_to_serializable(self.centroids),
            'euclideans': convert_to_serializable(self.euclideans),
            'clusters': convert_to_serializable(self.clusters)
        }


class KMeans():

    def __init__(self, data, k, labels):
        self.data = data
        self.k = k
        self.labels = labels
        self.assignment = [-1 for _ in range(len(data))]
        self.results = []

    def _is_unassigned(self, i):
        return self.assignment[i] == -1

    def _unassign_all(self):
        self.assignment = [-1 for _ in range(len(self.data))]

    def _is_centers_diff(self, c1, c2):
        for i in range(self.k):
            if self.dist(c1[i], c2[i]) != 0:
                return True
        return False

    def dist(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    def kmeans_plusplus(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        centroids = self.data[np.random.choice(range(len(self.data)), size=1)]
        for _ in range(1, self.k):
            min_sq_dist = [min([self.dist(c, x) ** 2 for c in centroids])
                           for x in self.data]
            prob = min_sq_dist / sum(min_sq_dist)
            centroids = np.append(centroids, self.data[np.random.choice(
                range(len(self.data)), size=1, p=prob)], axis=0)
        return centroids

    def assign(self, centers):
        temp = ClusteringResult()
        temp.iteration = self.iteration + 1
        temp.centroids = centers
        for i in range(len(self.data)):
            min_dist = float('inf')
            eucTemp = []
            for j in range(self.k):
                d = self.dist(self.data[i], centers[j])
                eucTemp.append(d)
                if d < min_dist:
                    min_dist = d
                    self.assignment[i] = j
            eucTemp.insert(0, self.labels[i])
            temp.euclideans.append(eucTemp)

        for i in range(len(self.data)):
            temp.clusters.append([self.labels[i], self.assignment[i]+1])

        self.results.append(temp)

    def compute_centers(self):
        centers = []
        for j in range(self.k):
            cluster = np.array([self.data[k] for k in filter(lambda x: x >= 0,
                                                             [i if self.assignment[i] == j else -1 for i in range(len(self.data))])])
            centers.append(np.mean(cluster, axis=0))
        return np.array(centers)

    def lloyds(self, max_iter=100, seed=None):
        self.iteration = 0
        centers = self.kmeans_plusplus(seed=seed)
        self.assign(centers)
        new_centers = self.compute_centers()
        while self._is_centers_diff(centers, new_centers) and self.iteration < max_iter:
            self._unassign_all()
            self.iteration += 1
            centers = new_centers
            self.assign(centers)
            new_centers = self.compute_centers()

    def printResults(self):
        for x in self.results:
            print(x.iteration)
            print(x.centroids)
            print(x.euclideans)
            print(x.clusters)

    def plotElbow(self, max_k):
        inertias = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(self.data, k, self.labels)
            kmeans.lloyds()
            inertia = 0
            for result in kmeans.results:
                inertia += sum(min(euc[1:]) ** 2 for euc in result.euclideans)
            inertias.append(inertia)

        plt.plot(range(1, max_k + 1), inertias, marker='o')
        plt.title('Elbow Method')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.savefig('images/elbow.png')
        plt.close()

    def plot_clusters(self):
        # Get the last clustering result
        last_result = self.results[-1]
        centroids = last_result.centroids
        assignments = np.array(self.assignment)

        # Assign colors to clusters
        colors = plt.cm.get_cmap('tab10', self.k)

        if self.data.shape[1] >= 3:
            # 3D plot for the first three features
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for i in range(self.k):
                cluster_data = self.data[assignments == i]
                ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], s=30, c=[
                           colors(i)], label=f'Cluster {i+1}')
            ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                       s=200, c='black', marker='X', label='Centroids')
            ax.set_title('Cluster Assignments and Centroids')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_zlabel('Feature 3')
            plt.legend()
            plt.savefig('images/cluster.png')
            plt.close()
        else:
            # 2D plot for the first two features
            for i in range(self.k):
                cluster_data = self.data[assignments == i]
                plt.scatter(cluster_data[:, 0], cluster_data[:, 1], s=30, c=[
                            colors(i)], label=f'Cluster {i+1}')
            plt.scatter(centroids[:, 0], centroids[:, 1],
                        s=200, c='black', marker='X', label='Centroids')
            plt.title('Cluster Assignments and Centroids')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.legend()
            plt.savefig('images/cluster.png')
            plt.close()


@app.route('/upload', methods=['POST'])
def upload_file():
    global dataset, dataset_labels, columns, rows, kmeans, label_desa, dictionary_db, uploadedFileName

    file = request.files['file']
    rd = pd.read_excel(file)

    uploadedFileName = file.filename

    # Clean column names by stripping whitespace and replacing multiple spaces with a single space
    rd.columns = rd.columns.str.strip().str.replace(r'\s+', ' ', regex=True)

    # update this
    prev_column = rd.columns.tolist()
    prev_dataset = rd.head().to_dict(orient='records')
    prev_len = len(rd)

    # Keep only the specified columns
    rd = rd[['KDDESA', 'Nama', 'Umur', 'Ijazah_tertinggi', 'Jenis_cacat',
             'Status_pekerjaan', 'Jumlah_jamkerja', 'Lapangan_usaha', 'Penyakit_kronis']]

    # Convert all columns except the first one to numerical values, filling non-numeric with NaN and then replacing NaN with 0
    rd.iloc[:, 2:] = rd.iloc[:, 2:].apply(
        pd.to_numeric, errors='coerce').fillna(0)

    # Apply the filters
    rd = rd[
        (rd['Umur'] > 50) &
        (rd['Ijazah_tertinggi'] < 4) &
        (rd['Jenis_cacat'] > 0) &
        (rd['Status_pekerjaan'] > 0) &
        (rd['Penyakit_kronis'] > 0) &
        (rd['Lapangan_usaha'] > 0)
    ]

    dataset = rd.iloc[:, 2:].values
    rd.iloc[:, 1] = rd.iloc[:, 1].str.strip()
    dataset_labels = rd.iloc[:, 1].values
    label_desa = rd.iloc[:, 0].values
    columns = rd.columns
    rows = len(dataset)

    kmeans = KMeans(dataset, 4, dataset_labels)

    dtn = []
    lbn = []
    counter = 0
    for l, v in zip(dataset_labels.tolist(), dataset.tolist()):
        if counter < 14:
            lbn.append(l)
            dtn.append(v)
            counter += 1

        dictionary_db[l] = v

    return jsonify({
        "previous_columns": prev_column,
        "previous_dataset": prev_dataset,
        "previous_length": prev_len,
        "dataset": dtn,
        "labels": lbn,
        "columns": columns.to_list()[1:],
        "rows": rows
    }), 200


@app.route('/cluster', methods=['POST'])
def cluster_config():
    global kmeans, dataset, dataset_labels, max_iteration, cluster
    if len(dataset) == 0:
        return jsonify({'message': 'Please upload your dataset first'}), 400

    data = request.json

    max_iteration = data['iteration']
    cluster = data['cluster']

    return jsonify({'message': 'Success set cluster'})


@app.route('/k-means', methods=['GET'])
def kmeans_process():
    global kmeans, dataset, dataset_labels, cluster, dictionary_db, uploadedFileName
    if len(dataset) == 0:
        return jsonify({'message': 'Please upload your dataset first'}), 400

    kmeans = KMeans(dataset, cluster, dataset_labels)
    kmeans.lloyds(max_iteration, 42)

    a = []
    b = []

    for result in kmeans.results:
        each = result.to_dict()
        cl = each['clusters']

        temp = []
        for ecl in cl:
            name = ecl[0]
            # looking up to dict
            if name not in dictionary_db:
                continue

            dict_db = dictionary_db[name]
            temp.append(ecl + dict_db)

        b.append(temp)
        a.append(each)

    for i in range(len(a)):
        a[i]['clusters'] = b[i]

    # save last result to db
    current_date = datetime.now()
    namaKota = ''
    if 'JakBar' in uploadedFileName:
        namaKota = 'Jakarta Barat'
    elif 'JakPus' in uploadedFileName:
        namaKota = 'Jakarta Pusat'
    elif 'JakSel' in uploadedFileName:
        namaKota = 'Jakarta Selatan'
    elif 'JakTim' in uploadedFileName:
        namaKota = 'Jakarta Timur'
    elif 'JakUt' in uploadedFileName:
        namaKota = 'Jakarta Utara'
    elif 'KepRi' in uploadedFileName:
        namaKota = 'Kepulauan Seribu'

    try:
        random_string = ''.join(random.choices(string.digits, k=6))

        for i in a[2]['clusters']:
            history = KmeansHistory(
                file_name=uploadedFileName, kota=namaKota, clustering_date=current_date, index=random_string, nama=i[0], cluster=i[1], umur=i[2], ijazah_tertinggi=i[3], jenis_cacat=i[4], status_pekerjaan=i[5], jumlah_jam_kerja=i[6], lapangan_usaha=i[7], penyakit_kronis=i[8],)
            db.session.add(history)

        db.session.commit()

        return jsonify({'result': a})
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': e}), 409

    # return jsonify({'result': [result.to_dict() for result in kmeans.results]})


@app.route('/history', methods=['GET'])
def get_history():
    try:
        distinct_indexes = db.session.query(
            KmeansHistory.index).distinct().all()
        distinct_indexes = [index[0] for index in distinct_indexes]

        results = []
        for x in distinct_indexes:
            history_records = KmeansHistory.query.filter_by(
                index=x).first()
            results.append({
                'id': history_records.id,
                'file_name': history_records.file_name,
                'kota': history_records.kota,
                'clustering_date': history_records.clustering_date,
                'index': history_records.index,
            })

        # Return distinct indexes and filtered records
        return jsonify({
            'distinct_indexes': distinct_indexes,
            'records': results
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/history-detail', methods=['GET'])
def get_history_detail():
    id = request.args.get('id')

    try:
        results = []
        history_records = KmeansHistory.query.filter_by(
            index=id).all()
        for x in history_records:
            results.append({
                'id': x.id,
                'file_name': x.file_name,
                'kota': x.kota,
                'clustering_date': x.clustering_date,
                'index': x.index,
                'nama': x.nama,
                'cluster': x.cluster,
                'umur': x.umur,
                'ijazah_tertinggi': x.ijazah_tertinggi,
                'jenis_cacat': x.jenis_cacat,
                'status_pekerjaan': x.status_pekerjaan,
                'jumlah_jam_kerja': x.jumlah_jam_kerja,
                'lapangan_usaha': x.lapangan_usaha,
                'penyakit_kronis': x.penyakit_kronis
            })

        return jsonify({
            'records': results
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/elbow', methods=['POST'])
def plot_elbow():
    global kmeans, dataset
    if len(dataset) == 0:
        return jsonify({'message': 'Please upload your dataset first'}), 400

    data = request.json
    kmeans.plotElbow(data['max_cluster'])
    return jsonify({"path": 'http://' + request.host + '/images/elbow.png'})


@app.route('/images/<path:filename>', methods=['GET'])
def serve_image(filename):
    return send_from_directory('images', filename)


@app.route('/cluster', methods=['GET'])
def plot_clusters():
    global kmeans, dataset, dataset_labels, cluster, label_desa
    if len(dataset) == 0:
        return jsonify({'message': 'Please upload your dataset first'}), 400

    if cluster == None:
        return jsonify({'message': 'Please complete the K-Means processing first.'}), 400

    kmeans = KMeans(dataset, cluster, dataset_labels)
    kmeans.lloyds(max_iteration, 42)
    iterations = [x.to_dict() for x in kmeans.results][-1]

    for x in range(len(iterations['clusters'])):
        iterations['clusters'][x].insert(0, label_desa[x])

    kmeans.plot_clusters()
    clusterPath = 'http://' + request.host + '/images/cluster.png'

    return jsonify({'iterations': iterations, 'clusterPath': clusterPath})


if __name__ == '__main__':
    app.run()
