import argparse
import os

import tiledb

from scimilarity.knn_backends import KNN_BACKENDS

cfg = tiledb.Config()
cfg["sm.mem.total_budget"] = 50000000000  # 50G

def main():
    parser = argparse.ArgumentParser(description="Build cellsearch knn from embeddings tiledb")
    parser.add_argument("-m", type=str, help="model path")
    parser.add_argument("--cellsearch", type=str, default="cellsearch", help="relative path to the cellsearch folder")
    parser.add_argument("--embeddings", type=str, default="cell_embedding", help="relative path to the cell embeddings folder")
    parser.add_argument("--knn_filename", type=str, default="full_kNN.bin", help="knn filename")
    parser.add_argument("--knn_type", type=str, default="tiledb_vector_search", help="Type of knn: ['hnswlib', 'tiledb_vector_search']")
    parser.add_argument("--ef_construction", type=int, default=1000, help="hnswlib ef construction parameter")
    parser.add_argument("--M_construction", type=int, default=80, help="hnswlib M construction parameter")
    args = parser.parse_args()
    print(args)

    model_path = args.m
    knn_filename = args.knn_filename
    knn_type = args.knn_type
    ef_construction = args.ef_construction
    M = args.M_construction
 
    # embeddings
    cellsearch_path = os.path.join(model_path, args.cellsearch)
    embedding_tdb = tiledb.open(os.path.join(cellsearch_path, args.embeddings), "r", config=cfg)
    attr = embedding_tdb.schema.attr(0).name
    embeddings = embedding_tdb[:][attr]
    embedding_tdb.close()

    # build knn
    knn_fullpath = os.path.join(cellsearch_path, knn_filename)

    if knn_type not in KNN_BACKENDS:
        raise ValueError(
            f"Unknown knn_type {knn_type!r}. "
            f"Available backends: {sorted(KNN_BACKENDS)}"
        )

    backend_cls = KNN_BACKENDS[knn_type]
    backend = backend_cls()

    if knn_type == "hnswlib":
        backend.build(
            embeddings,
            ef_construction=ef_construction,
            M=M,
        )
        backend.save(knn_fullpath)
    elif knn_type == "tiledb_vector_search":
        backend.build(embeddings, index_uri=knn_fullpath)
        print("TileDB index built at:", knn_fullpath)

if __name__ == "__main__":
    main()
