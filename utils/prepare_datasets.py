import os
import pickle

import numpy as np

from extract_surroundings import extract_surroundings, get_complete_dataset


def main():
    # print("Extracting chen11...")
    # extract_arffs("David_arffs/chen11", "chen11.pckl")
    #
    # print("chen11 extracted succesfully, extracting holo4k...")
    # extract_arffs("David_arffs/holo4k", "holo4k.pckl")

    print("holo4k extracted succesfully, extracting chen11 surroundings...")
    checkpoints_folder = "checkpoints"
    if not os.path.exists(checkpoints_folder):
        os.mkdir(checkpoints_folder)
    with open("chen11.pckl", "rb") as file:
        chen11 = pickle.load(file)
    chen11_surroundings, chen11_labels = extract_surroundings(chen11[:300], 30,
                                                              os.path.join(checkpoints_folder, "chen11_surroundings"))
    # save surroundings and labels to .npy files
    np.save("chen11_surroundings.npy", chen11_surroundings)
    np.save("chen11_labels.npy", chen11_labels)
    return

    print("chen11 surroundings extracted succesfully, extracting holo4k surroundings...")
    with open("holo4k.pckl", "rb") as file:
        holo4k = pickle.load(file)
    holo4k_surroundings = extract_surroundings(holo4k[:30], 30,
                                               os.path.join(checkpoints_folder, "holo4k_surroundings"),
                                               get_complete_dataset)
    with open("holo4k_surroundings.pckl", "wb") as file:
        pickle.dump(holo4k_surroundings, file)
    print("holo4k surroundings extracted succesfully, all done!")


if __name__ == '__main__':
    main()
