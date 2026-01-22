from scipy.spatial.distance import cosine

def top_k_similar(query_vec, db_dict, k=5):
    sims = []
    for product_link, stored_vec in db_dict.items():
        score = 1.0 - cosine(query_vec, stored_vec)
        sims.append((product_link, float(score)))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:k]
