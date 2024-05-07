from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from config.constants import DATA_FOLDER, IMAGES_FOLDER
from dataset.extract_data.extract_arffs import _load_arffs


def do_the_plot(x, y, z, c, name: str):
    ax = plt.axes(projection='3d')
    ax.scatter3D(-x, y, z, c=c, alpha=0.4, s=4, marker='.')
    ax.view_init(190, -10, 0)
    IMAGES_FOLDER.mkdir(exist_ok=True)
    plt.savefig(IMAGES_FOLDER / name, dpi=1200, bbox_inches='tight')


if __name__ == '__main__':
    path_1fbl = DATA_FOLDER / "raw/joined/b210/1fbl"
    df = _load_arffs(path_1fbl)[0]
    df["@@class@@"] = df["@@class@@"] == b'1'

    do_the_plot(df["xyz.x"],
                df["xyz.y"],
                df["xyz.z"],
                ["red" if x else "green" for x in df["@@class@@"]],
                "point_cloud_class.png")
    pca = PCA(3)
    features = df.drop(["xyz.x", "xyz.y", "xyz.z", "@@class@@"], axis=1).to_numpy()
    print(f"Features shape: {features.shape}")
    c = pca.fit_transform(features)

    scaler = MinMaxScaler(feature_range=(0, 1))
    c = scaler.fit_transform(c)
    do_the_plot(df["xyz.x"],
                df["xyz.y"],
                df["xyz.z"],
                c,
                "point_cloud_pca.png")
